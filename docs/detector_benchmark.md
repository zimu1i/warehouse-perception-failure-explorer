# Zero-shot detector benchmark (dev sample)

Run 2026-10-28 on 50 annotated dev images (seed 0, spread evenly across subsets 2, 3, 5; 1,779 GT boxes), Apple M3 / 16 GB, MPS backend, torch 2.15.0, transformers 5.21.0. Metrics from pycocotools (COCO AP, AP50, recall at IoU 0.5 with up to 500 detections per image). Predictions stored down to score 0.05. Commands: `uv run wpfe benchmark --model <model> --prompts <set>`.

## Models

| Model | Checkpoint revision | License | s/image (median, MPS) | MPS memory |
|---|---|---|---|---|
| Grounding DINO tiny | `IDEA-Research/grounding-dino-tiny@a2bb814` | Apache-2.0 | 1.01 | 4.6 GB |
| OWLv2 base | `google/owlv2-base-patch16-ensemble@cfd3195` | Apache-2.0 | 0.71 | 1.1 GB |

Prompt sets (`configs/default.yaml`): `literal` uses the class names; `descriptive` uses "wooden pallet", "hand pallet jack", "plastic storage crate", "metal wire mesh cage container", "forklift".

## Recall at IoU 0.5 (all predictions >= 0.05)

| Class (GT boxes) | GDINO literal | GDINO descriptive | OWLv2 literal | OWLv2 descriptive |
|---|---|---|---|---|
| pallet (1,411) | 0.05 | 0.11 | **0.25** | 0.16 |
| small load carrier (292) | 0.05 | 0.21 | 0.00 | **0.32** |
| stillage (44) | 0.05 | **0.77** | 0.52 | 0.55 |
| pallet truck (25) | **0.56** | 0.44 | 0.36 | 0.32 |
| forklift (7) | **1.00** | **1.00** | 0.57 | 0.86 |

AP50 is below 0.05 for pallets and small load carriers in every configuration.

## Per-image recall (preview, rough greedy matcher)

OWLv2 literal: at score >= 0.1, median image recall 0.14 and 90% of images have recall < 0.5; at score >= 0.2, median 0.03 and 96% below 0.5. The descriptive set is similar.

## Findings

1. **Granularity mismatch.** LOCO annotates each pallet and crate in a stack individually. Both models tend to box a whole stack as one object, so most stacked pallets are unmatched at IoU 0.5. Visual inspection confirmed predicted boxes are aligned with the image (e.g. forklift and stillage boxes overlap their GT), so this is not a coordinate bug. Median predicted box width was 225 px vs 70 px for GT pallets (Grounding DINO).
2. **Vocabulary.** "small load carrier" is not understood by OWLv2 (recall 0.00); "plastic storage crate" works better (0.32). "stillage" works only for some phrasings.
3. **False positives on scene structure.** Loading-dock doors and wall fixtures are frequently labeled "pallet".
4. **Implementation notes.** Grounding DINO's library label decoding merges tokens from several phrases at a low text threshold; class assignment instead uses the mean token score per phrase (positive map). OWLv2 boxes are rescaled by max(height, width) because of square padding; transformers 5 does this internally.

## Consequence for the experiment

With zero-shot detectors, almost every image would count as a severe failure under any usable operating threshold, which leaves little headroom for review ranking to beat random selection. See decision D5.
