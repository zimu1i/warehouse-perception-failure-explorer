# Warehouse Perception Failure Explorer

Robots and camera rigs collect far more images than anyone has time to label. If a reviewer can only check 50 images, which 50 should they look at?

In this project I tested one answer to that question. I fine-tuned an object detector on three warehouses, ran it on two warehouses it had never seen, and ranked those images by **flip instability**: I run the detector on each image twice, once normally and once mirrored, and measure how much its two answers disagree. The idea is that if mirroring an image changes what the model sees, the model is probably unsure about that image, and unsure images are where mistakes hide.

The ranking never looks at the labels. I only used the labels afterwards, to count how many real failures each method found.

![Failures found by budget](docs/figures/failures_by_budget.png)

## What I found

The pool had 2,277 images from the two unseen warehouses, and 355 of them (15.6%) were severe failures, meaning the detector found fewer than 40% of the objects in the image.

- The 50 most unstable images contained 32 failures. Random batches of 50 contained about 8 on average, and never more than 13 in 1,000 tries. The result held at both warehouses and for batches of 25, 50 and 100.
- When I looked at which images the ranking picked, most were nearly empty scenes with only two or three objects. Those images fail more often anyway, because missing two out of three objects already counts as a failure. A much simpler rule, "pick the images with the fewest detections", got 28 failures out of 50. So a good part of the 4x gain comes from scene density rather than from the instability score itself.
- When I compared against random batches with the same mix of sparse and busy scenes, instability still found about 1.6x more failures. The signal is real, just smaller than the headline number.
- About one in five failures are images where the detector is wrong in the same way in both views. Flip instability can't catch those, and I show an example in the report.

The full write-up, including per-site results and limitations, is in **[docs/report.md](docs/report.md)**.

To be clear about scope: this is about prioritizing images for human review. The detector is frozen before any images are selected and is never retrained on them, so this is not active learning.

## How it works

```
LOCO images ──► ingest ──► validate ──► train ──► run on original + mirrored images ──► instability score
                tables     quality      Faster    predictions saved, labels never read
                           checks       R-CNN
                                                               │
               score threshold and failure rule set on a ◄────┤
               separate tuning warehouse, then frozen          ▼
                                     pick images: random / instability / instability + spacing
                                                               │
                                    labels opened only here ───▼
                                    count failures in each batch ──► report and figures
```

I was careful about leakage because it would be easy to fool myself here:

- Train, tune and review data come from different warehouses, so near-identical frames can't end up on both sides.
- I chose the failure rule and the score threshold on the tuning warehouse, and committed them along with the experiment design before reading a single review label.
- The selection functions only receive a table without labels, so they can't use them even by accident.
- I ran the experiment once.

Every design decision, with the alternatives I considered and what each one costs, is written up in [docs/decision_log.md](docs/decision_log.md).

## Data

I used [LOCO (Logistics Objects in Context)](https://github.com/tum-fml/loco) from TUM (Mayershofer et al., ICMLA 2020, CC0-1.0). It has 5,097 labelled images with 151,428 boxes across five classes: pallet, small load carrier, stillage, pallet truck and forklift. The images were taken by people walking through warehouses with handheld cameras. More detail is in [docs/dataset_card.md](docs/dataset_card.md). The images themselves are not included in this repository.

## Running it

You need Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). I built and ran everything on an M3 MacBook (MPS backend); CUDA and CPU should also work.

First download `dataset.zip` (linked from the LOCO repository) and extract it to `data/raw/loco_v1/`, along with the annotation JSON files from the repository's `rgb/` folder. Then:

```bash
uv sync
uv run wpfe ingest        # build image and annotation tables (Parquet)
uv run wpfe validate      # data quality checks
uv run wpfe train         # about 3.5 hours on an M3, can resume if interrupted
uv run wpfe infer         # about 40 minutes, original and mirrored images
uv run wpfe experiment    # a few seconds, uses the saved predictions
uv run wpfe figures
uv run pytest             # 79 tests on small synthetic examples, no dataset needed
```

To browse the results, start the viewer:

```bash
uv run --group viewer streamlit run src/wpfe/viewer.py
```

It has two modes. **Review queue** shows what a reviewer would see: images ranked by a selection method, with the boxes from the original and mirrored images drawn on top and the disagreeing boxes highlighted. It never loads the labels. **Results** opens the labels and lets you step through caught failures, false alarms and consistent misses.

`uv run wpfe benchmark --model owlv2_base --prompts descriptive` reruns the zero-shot detector comparison I did before deciding to train my own model ([docs/detector_benchmark.md](docs/detector_benchmark.md)).

All settings, including the frozen failure rule, are in [configs/default.yaml](configs/default.yaml).

## Code layout

| Path | What it does |
|---|---|
| `src/wpfe/ingest.py`, `validate.py` | build the data tables and run quality checks |
| `src/wpfe/detect.py`, `benchmark.py`, `coco_eval.py` | zero-shot detector benchmark |
| `src/wpfe/train.py`, `infer.py` | fine-tuning, and saved predictions on original and mirrored images |
| `src/wpfe/boxes.py`, `matching.py` | box mirroring, IoU, and matching predictions to labels |
| `src/wpfe/instability.py`, `select.py` | the instability score and the image selection methods |
| `src/wpfe/experiment.py`, `figures.py` | the experiment and the figures |
| `src/wpfe/viewer.py` | Streamlit viewer for the review queue and the results |
| `docs/` | report, decision log, dataset card, detector benchmark |

## Limitations

- There are only two review warehouses and eight camera sequences, and nearby frames look alike, so the image-level p-values look more certain than they really are.
- My failure rule is based on recall per image, which is coarse for images with very few objects. That is where the density effect above comes from.
- I tested one detector and one kind of change to the image (a horizontal flip).
- This is an offline analysis project. It doesn't make any claims about real deployed systems or safety.
