# Report: choosing warehouse images for human review without looking at labels

## The question

If a reviewer only has time to check a small batch of images, can a ranking that never looks at labels choose images with more object detection failures than random sampling would?

## Short answer

Yes, though by less than the first number suggests. Ranking images by flip instability found about four times as many failures as random sampling, at both unseen warehouses and for every batch size I tried. When I checked why, I found that a large part of this comes from the ranking preferring sparse scenes, and a simple "fewest detections" rule gets most of the same benefit. With scene density held constant, instability still finds about 1.6 times as many failures as random. Roughly a fifth of all failures are cases where the detector is consistently wrong, which this method cannot detect.

The detector is frozen before any images are selected and is never retrained on them. This is review prioritization, not active learning.

## 1. Data and split

I used LOCO (Logistics Objects in Context, TUM, CC0-1.0). The images were taken by people walking through warehouses with handheld cameras, so they are not robot-mounted camera data. There are five classes: pallet, small load carrier, stillage, pallet truck and forklift. Details are in `docs/dataset_card.md`.

LOCO is organized into five subsets, each from a different site. I split by site so that near-identical frames from the same walk could never end up in both training and evaluation (decisions D3 and D6):

| Role | Subsets | Labelled images | Used for |
|---|---|---|---|
| Training | 2, 5 | 1,947 | fine-tuning the detector |
| Tuning | 3 | 873 | choosing the score threshold and the failure cutoff |
| Review pool | 1, 4 | 2,277 | the selection experiment only |

No code read the review pool's labels until the evaluation step. The failure rule and the experiment design were committed first (commit `4f888ed`), and the evaluation code came after (commit `c7e18f1`).

## 2. Detector

I first tried two zero-shot detectors that take text prompts, Grounding DINO tiny and OWLv2 base (`docs/detector_benchmark.md`). Both tended to draw one box around a whole stack of pallets, while LOCO labels every pallet separately. With either model, about 90% of images would have counted as failures, which leaves nothing for a ranking method to improve on (D5).

So I fine-tuned a COCO-pretrained Faster R-CNN (torchvision, ResNet-50 FPN v2) on subsets 2 and 5. I trained for 6 epochs with images resized to 640 px on the short side and random horizontal flips, using a fixed schedule and the last epoch, on an M3 MacBook. Training took about 3.5 hours. On the tuning site, at the operating point I chose (score of at least 0.4, IoU 0.5), precision was 0.47 and recall 0.42.

One thing that stood out: the model barely recognized pallet trucks at the tuning site (recall 0.15 on the monitoring sample), even though it had 1,366 pallet truck boxes in training. Most of those came from a single site, so the model seems to have learned what that site's pallet trucks look like rather than pallet trucks in general.

## 3. What counts as a failure

- A predicted box matches a labelled box if they have the same class and an IoU of at least 0.5. Each prediction can match at most one label, and the most confident predictions pick first. This is the same rule pycocotools uses (D8).
- An image is a **severe failure** if fewer than 40% of its labelled objects are matched by predictions with a score of at least 0.4 (D1, D10).
- I chose the score threshold as the one with the best F1 score on the tuning site. I chose the 40% cutoff so that a reasonable share of tuning images (43%) counted as failures.

## 4. How images are selected

None of these methods sees labels. They only receive a table with each image's id, camera sequence, frame position and instability score.

- **Random:** 1,000 random batches per batch size, to show what luck alone produces.
- **Flip instability (D9):** I run the detector on each image and on its mirror image, map the mirrored boxes back, and pair up boxes between the two views using the matching rule above. The score is the share of total confidence that sits on boxes with no partner in the other view. 0 means the two views agree completely; 1 means nothing matches. I take the top-scoring images.
- **Instability with spacing (D11):** the same ranking, but an image is skipped if it is within 2 frames of an image already picked from the same camera. Before running the experiment I checked, without labels, that 42 of the top 100 images were within 2 frames of another pick.

## 5. Results

I ran the experiment once, as designed. The review pool has 2,277 images, and 355 of them are severe failures (15.6%). That is lower than the 43% at the tuning site, so the review warehouses turned out to be easier for the model.

![Failures found by batch size](figures/failures_by_budget.png)

Failures found per batch, whole review pool:

| Batch size | Random, average (95% range) | Instability | Ratio | Instability with spacing |
|---|---|---|---|---|
| 25 | 3.8 (1 to 8) | 17 | 4.4x | 16 |
| 50 | 7.8 (3 to 13) | 32 | 4.1x | 32 |
| 100 | 15.4 (9 to 23) | 52 | 3.4x | 54 |

None of the 1,000 random batches did as well as instability at any batch size.

Looking at each warehouse separately with batches of 50, instability found 29 failures versus 8.8 at subset 1, and 30 versus 6.9 at subset 4. The pool only has 8 camera sequences and nearby frames are similar, so I treat agreement between the two sites as the main evidence rather than the p-values.

As a sensitivity check I loosened the matching rule to IoU 0.3. The failure rate drops to 8.4%, and instability finds 30 failures in 50 versus 4.3 for random, so the conclusion doesn't depend on that setting.

Spacing out the picks didn't help consistently. It helped slightly at subset 4 for small batches (19 versus 16 with batches of 25) and hurt at subset 1 (23 versus 29 with batches of 50).

## 6. Checking for a confound

This part was not in the original plan. I added it after seeing the results, so I treat it as exploratory.

The 50 most unstable images had a median of 2.5 labelled objects, compared with 21 across the whole pool. That matters for two reasons. With a recall-based rule, sparse images fail more easily: 26% of images with five or fewer objects are failures, against 15.6% overall. The instability score is also jumpier when an image has only a few boxes. So both the score and the failure label rise on sparse images, partly for reasons that have nothing to do with the model being unsure.

To measure this, I compared instability with two other ways of choosing images that also don't use labels:

| Batch size | Instability | Fewest detections | Random, same mix of sparse and busy scenes (average) |
|---|---|---|---|
| 25 | 17 | **20** | 10.9 |
| 50 | **32** | 28 | 19.2 |
| 100 | **52** | 45 | 33.6 |

Two things follow from this. First, instability still beats random batches with the same density mix (p = 0.007 at 25, p < 0.001 at 50 and 100), so the score does pick up something beyond sparsity. Second, simply choosing the images with the fewest detections captures most of the benefit, and does better than instability for batches of 25.

## 7. What instability misses

67 of the 355 failures (19%) had below-median instability, and 10 failures had no detections at all at the operating point. If the detector misses an object in the original image and also in the mirrored one, the two views agree, and the score sees nothing wrong.

![Examples](figures/examples.jpg)

*Left: an image the ranking picked that really was a failure (0 of 10 objects found). Middle: an image it picked that was fine. Right: a failure the ranking scored as stable. White boxes are labels, blue boxes are predictions on the original image, and orange boxes are predictions on the mirrored image (score of at least 0.4).*

## 8. Limitations

- The review pool's failure rate (15.6%) ended up below the 20 to 50% range I aimed for on the tuning site, because the review warehouses were easier.
- The recall-based failure rule is coarse for images with few objects, and that explains much of the measured gain (section 6).
- Two review sites and 8 camera sequences are not many independent samples, so image-level p-values overstate how certain these results are.
- I only tested one detector, trained with flip augmentation, and only one kind of image change.
- The matching rule is greedy rather than optimal. I kept it so my numbers line up with the standard evaluator.

## 9. What I would try next

- Combine signals, for example ranking by instability within groups of similar scene density, or adding disagreement between two different detectors, which could catch some of the consistent misses.
- Keep a small random sample in every review batch, so the reviewer still sees the kinds of failures the ranking can't detect.
- Define failures in a way that is less sensitive to object count, and rerun the experiment on a newly frozen split.

## Reproducing the results

The README explains the setup. Every number in this report comes from `uv run wpfe experiment` using the saved predictions, and the figures come from `uv run wpfe figures`.
