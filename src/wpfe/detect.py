"""Open-vocabulary detector wrappers with one shared output format.

Every wrapper returns, per image, a list of dicts:
    {"class_name", "phrase", "score", "x_min", "y_min", "x_max", "y_max"}
in original-image pixel coordinates (same xyxy convention as the manifests).
Scores are model-specific and not calibrated probabilities.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from PIL import Image


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def phrase_to_class(prompt_set: dict[str, list[str]]) -> dict[str, str]:
    """Invert class -> phrases. Fails if a phrase would map to two classes."""
    mapping: dict[str, str] = {}
    for class_name, phrases in prompt_set.items():
        for phrase in phrases:
            key = phrase.strip().lower()
            if key in mapping and mapping[key] != class_name:
                raise ValueError(f"phrase {phrase!r} maps to both {mapping[key]} and {class_name}")
            mapping[key] = class_name
    return mapping


class GroundingDino:
    def __init__(self, hf_id: str, revision: str, prompt_set: dict, device: str):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(hf_id, revision=revision)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(hf_id, revision=revision).to(device).eval()
        self.phrase_map = phrase_to_class(prompt_set)
        self.phrases = list(self.phrase_map)
        # Grounding DINO takes one lowercase string with phrases separated by " . "
        self.text = " . ".join(self.phrases) + " ."
        self.positive_map = self._build_positive_map()

    def _build_positive_map(self) -> torch.Tensor:
        """[n_phrases, n_tokens] matrix; row p averages the token scores of phrase p.

        Same idea as the official Grounding DINO COCO evaluation. We do not use
        the library's text-threshold label decoding: at a low threshold it glues
        tokens from several phrases into one label ("pallet forklift ..."), and
        phrases sharing tokens ("pallet" / "pallet truck") need the mean, not the
        max, to stay separable.
        """
        enc = self.processor.tokenizer(self.text, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        pos = torch.zeros(len(self.phrases), len(offsets))
        start = 0
        for p, phrase in enumerate(self.phrases):
            begin = self.text.index(phrase, start)
            end = begin + len(phrase)
            start = end
            tokens = [t for t, (a, b) in enumerate(offsets) if b > a and a >= begin and b <= end]
            pos[p, tokens] = 1.0 / len(tokens)
        return pos

    @torch.inference_mode()
    def predict(self, image: Image.Image, score_floor: float) -> list[dict]:
        inputs = self.processor(images=image, text=self.text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        n_tokens = self.positive_map.shape[1]
        token_probs = outputs.logits[0, :, :n_tokens].sigmoid().cpu()  # [queries, tokens]
        phrase_scores = token_probs @ self.positive_map.T  # [queries, phrases]
        scores, best = phrase_scores.max(dim=1)  # one class per query

        # pred_boxes are normalized (cx, cy, w, h) relative to the input image.
        cx, cy, w, h = outputs.pred_boxes[0].cpu().unbind(-1)
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
        boxes = boxes * torch.tensor([image.width, image.height, image.width, image.height])

        keep = scores >= score_floor
        return [
            _row(self.phrase_map[self.phrases[i]], self.phrases[i], s, b)
            for i, s, b in zip(best[keep].tolist(), scores[keep].tolist(), boxes[keep].tolist())
        ]


class Owlv2:
    def __init__(self, hf_id: str, revision: str, prompt_set: dict, device: str):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.device = device
        self.processor = Owlv2Processor.from_pretrained(hf_id, revision=revision)
        self.model = Owlv2ForObjectDetection.from_pretrained(hf_id, revision=revision).to(device).eval()
        self.phrase_map = phrase_to_class(prompt_set)
        self.phrases = list(self.phrase_map)

    @torch.inference_mode()
    def predict(self, image: Image.Image, score_floor: float) -> list[dict]:
        inputs = self.processor(text=[self.phrases], images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # OWLv2 pads the image to a square (bottom/right) before resizing, so raw
        # boxes are relative to the padded square. transformers>=5 rescales by
        # max(height, width) internally, so we pass the original size here.
        # Padding is bottom/right only, so the origin never shifts.
        result = self.processor.post_process_grounded_object_detection(
            outputs, threshold=score_floor, target_sizes=[(image.height, image.width)]
        )[0]
        return [
            _row(self.phrase_map[self.phrases[i]], self.phrases[i], score, box)
            for i, score, box in zip(result["labels"].tolist(), result["scores"].tolist(), result["boxes"].tolist())
        ]


def _row(class_name: str | None, phrase: str, score: float, box: list[float]) -> dict:
    x_min, y_min, x_max, y_max = box
    return {
        "class_name": class_name,  # None when the model returns a phrase we cannot map
        "phrase": phrase,
        "score": float(score),
        "x_min": float(x_min),
        "y_min": float(y_min),
        "x_max": float(x_max),
        "y_max": float(y_max),
    }


MODEL_CLASSES = {"grounding_dino_tiny": GroundingDino, "owlv2_base": Owlv2}


def load_detector(model_key: str, prompt_set_key: str, cfg_detector: dict, device: str):
    spec = cfg_detector["models"][model_key]
    prompt_set = cfg_detector["prompt_sets"][prompt_set_key]
    return MODEL_CLASSES[model_key](spec["hf_id"], spec["revision"], prompt_set, device)
