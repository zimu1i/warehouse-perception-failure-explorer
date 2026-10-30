# Decision log

Each entry: date, decision, alternatives, rationale, costs, status.

## D1. Primary failure event: severe-failure binary event (2026-10-26)

- **Options considered:**
  - A. Image has >= 1 unmatched GT or prediction. Rejected: LOCO is dense (median 20 boxes/image, 27% of boxes < 32x32 px), so this event is expected to be true for nearly every image, leaving no headroom for any ranking to beat random.
  - B. Severe failure: image-level threshold, fixed on the dev split only to target a 20-50% base rate, then frozen before the review pool is evaluated. **Chosen.**
  - C. Per-image error count. Kept as a secondary metric; as a primary metric it rewards dense scenes, which also cost more review time.
- **Definition:** an image is a severe failure if its recall (matched GT / total GT) is below tau at the frozen operating point. The alternative, >= k errors (FN + FP), reintroduces density bias.
- **Rationale:** missed objects are the failures that matter most in a warehouse setting, so the event prioritizes misses over false alarms. Recall is also normalized by GT count, which removes the density bias of count-based events.
- **Costs accepted:** recall ignores false positives, which still matter operationally (phantom obstacles cause stops and lost throughput), so they are reported as a secondary metric. Recall is coarse on images with few GT boxes. Image recall depends on the confidence threshold, so tau and the score threshold are frozen together. This is an offline review tool: the safety motivation explains the metric choice but makes no safety claim.
- **Status:** definition chosen; tau to be fixed on dev split only, then frozen.

## D2. The 496 archive images without annotation entries (2026-10-26)

- **Options considered:** keep them as unscored selector candidates (with best/worst-case bounds), or exclude them from the primary experiment.
- **Decision:** record them in the image manifest with status `unannotated`. Exclude them from the primary selection experiment (candidate pool = annotated images only, so every pick is scorable). Optional side analysis: run selectors over the full pool and report the unannotated share of each batch.
- **Rationale:** the images were left unlabeled for an unknown, possibly non-random reason (e.g. blur, empty frames). Instability-based selectors may prefer such images, and scoring them as "no failure" would penalize those selectors for missing labels, not ranking quality. They cannot be treated as zero-object negatives either.
- **Status:** final.

## D3. Split strategy: environment-level split (2026-10-27)

- **Options considered:** A environment-level split; B session-level split within each subset; variant of A with a single-subset dev set.
- **Decision:** dev = subsets 2, 3, 5 (2,820 annotated images); review pool = subsets 1, 4 (2,277 annotated images). This matches the repository's train/val file naming. No final test set, because no retraining claims are planned.
- **Rationale:** the review experiment should surface real problems rather than look-alikes of what was tuned on. Whole environments on each side rule out cross-split near-duplicate leakage by construction, and test whether dev-chosen thresholds transfer to unseen sites.
- **Costs accepted:** review pool is class-imbalanced (124 forklifts; subset 4 has no small load carriers), so per-class review-pool results will be thin. Frames inside the review pool remain correlated, so uncertainty must be grouped by session, not by image.
- **Caveat:** "subset = environment" is not yet verified; the leakage guarantee holds at the subset level regardless.
- **Status:** final. If retraining is added later, a test set must be carved out before any review-pool labels are used for training decisions.

## D4. Box edge rules: relative tolerance with escalation (2026-10-27)

- **Options considered:** A fixed 1 px tolerance; B zero tolerance; C relative tolerance with escalation.
- **Decision (C):** overshoot past an image edge is measured per axis as a fraction of that axis (x vs width, y vs height). <= 1%: OK. > 1% to 10%: `warning`. > 10%, or no overlap with the image at all: `error`.
- **Rationale:** the tolerance adapts to each image's own scale instead of one fixed pixel value; LOCO mixes five resolutions from 640x480 to 1920x1080.
- **Costs accepted:** three thresholds to justify and test; the > 10% escalation partly overlaps the image-level `size_mismatch` check. On LOCO v1 no box overshoots by more than 0.5 px, so these rules guard against future data or code bugs rather than changing current results.
- **Status:** final; validation flags only and never clips. Any clipping for evaluation is decided and documented in the evaluation step.

## D5. Detector setup: fine-tune on dev environments (2026-10-28)

- **Options considered:** A zero-shot OWLv2 with the best prompt per class; B fine-tune a closed-vocabulary detector on dev subsets only; C zero-shot, scored on stillages, forklifts and pallet trucks only.
- **Decision (B):** fine-tune a COCO-pretrained detector on part of the dev split, choose the operating threshold and tau on a held-out dev subset, and freeze the model before any review-pool inference.
- **Evidence:** zero-shot benchmark (`docs/detector_benchmark.md`): pallet AP50 below 0.05 for both models; with OWLv2 at score >= 0.1, about 90% of dev images have recall below 0.5, so the failure event would be saturated and leave no headroom for ranking.
- **Rationale:** a detector trained on some sites and deployed to new ones is the realistic setting for review prioritization. Its failures on unseen environments are informative rather than dominated by one systematic labeling-granularity mismatch, which a flip-instability signal cannot see.
- **Costs accepted:** roughly half a day of schedule; training on MPS; a training dependency and its license; the dev split now serves both training and calibration, so they must use different subsets.
- **Unchanged:** the model is frozen before selection, the review pool stays unseen, and the project makes no retraining or active-learning claim. Zero-shot results stay in the report as the reason for training.
- **Status:** chosen; training library and calibration subset pending (D6).

## D6. Training setup and dev re-split (2026-10-28)

- **Options considered:** tuning subset W3, W5 or W2 (the other two train); torchvision Faster R-CNN ResNet50-FPN v2 or MobileNetV3-FPN.
- **Decision:** train on subsets 2 and 5 (1,947 annotated images); choose the operating threshold and tau on subset 3 (873 images). Model: torchvision `fasterrcnn_resnet50_fpn_v2`, COCO_V1 weights (BSD-3), new 5-class head, shorter side 640 px, 6 epochs, SGD lr 0.005, fixed schedule.
- **Rationale:** subset 3 contains every class (402 stillages); subset 5 has only 1 stillage and could not check that class. ResNet50-FPN handles small, dense objects better than MobileNet.
- **Measured cost:** on the M3, 1.55 s per batch of 2 at 640 px (about 25 min per epoch, 5.8 GB); 800 px was 1.5x slower. No MPS operations fell back to CPU.
- **Guards:** `check_training_split` rejects overlapping train/tune subsets or any review subset (tested). The last epoch is used downstream; the tune subset is never used to pick an epoch.
- **Status:** final. Flip augmentation pending (D7).

## D7. Flip augmentation during training (2026-10-28)

- **Options considered:** horizontal flip with p = 0.5; no flip augmentation.
- **Decision:** p = 0.5, applied to image and boxes together at load time (raw data untouched).
- **Rationale:** standard practice and a better detector. It is also the harder test for flip instability: the model sees mirrored scenes during training, so remaining instability on test images is less likely to reflect unfamiliarity with mirrored views. Training without flips could inflate the signal and would look like tuning the setup in the method's favor.
- **Cost accepted:** the flip-instability signal may be weaker than with an unaugmented model.
- **Status:** final.

## D8. Matching rules for the failure event (2026-10-28)

- **Options considered (IoU threshold):** 0.5, 0.3, 0.75.
- **Decision:** class-aware, greedy one-to-one matching by descending score (the pycocotools rule); a GT box counts as found if matched at IoU >= 0.5.
- **Rationale:** 0.5 is the standard threshold (PASCAL VOC, COCO AP50) and easy to compare. One-to-one matching stops one large box over a stack from counting as several found pallets, which is the dominant zero-shot failure.
- **Costs accepted:** 31% of dev pallets are under 32x32 px, where a few pixels of offset fall below 0.5 IoU; object size is reported as a slice and results are re-run at IoU 0.3 as a sensitivity check. Greedy matching is not globally optimal (tested), but it matches the standard evaluator.
- **Status:** final.

## D9. Flip-instability score (2026-10-28)

- **Options considered:** A count of unpaired boxes; B fraction of unpaired boxes; C score-weighted fraction of unpaired boxes.
- **Decision (C):** match original-view boxes to flipped-view boxes (mapped back) with the D8 rule; instability = score mass of unpaired boxes in both views / total score mass in both views. Images with no boxes in either view score 0.
- **Rationale:** normalized like the D1 recall event, so dense scenes are not favored; a confident box that disappears under mirroring counts far more than flicker near the 0.05 score floor; needs no extra threshold.
- **Costs accepted:** an image where the detector finds nothing in either view looks perfectly stable, so consistent misses are invisible to this signal and are reported separately. Matching runs with original-view boxes as the reference and flipped-view scores setting the order, which is not perfectly symmetric.
- **Status:** final.

## D10. Frozen operating point and tau (2026-10-29)

- **Measured on W3 only (873 annotated images, original view):** F1 is flat at 0.43-0.44 for score thresholds 0.3-0.5. At threshold 0.4: precision 0.47, recall 0.42, median image recall 0.43. Severe-failure share at tau 0.3 / 0.4 / 0.5: 23% / 43% / 64%.
- **Options considered (tau, at threshold 0.4):** 0.4 (43%); 0.3 (23%); also tau 0.5 at threshold 0.2 (46%).
- **Decision:** score threshold 0.4 (maximum F1 on W3; the lower of the tied values, to favor recall); tau 0.4. An image is a severe failure if fewer than 40% of its GT boxes are matched at IoU 0.5 by predictions with score >= 0.4.
- **Rationale:** the threshold is set by a rule rather than by eye. tau 0.4 sits inside the 20-50% target band with room to fall if the review sites are easier (their median instability is lower than W3's), and keeps enough positive images for stable comparisons.
- **Costs accepted:** with a 43% base rate, a perfect selector can beat random by at most about 2.3x. Review-pool base rate is unknown until the experiment runs and may differ from W3.
- **Status:** frozen in `configs/default.yaml` (`failure_event`) before reading any review-pool labels.

## D11. Selection experiment design (2026-10-29)

- **Pool:** 2,277 annotated review images (subset 1: 1,075; subset 4: 1,202). Unannotated images are excluded (D2).
- **Budgets:** 25, 50, 100 images.
- **Selectors (label-blind; read predictions and file paths only):** random (1,000 seeds); top-k by flip instability (D9); instability with diversity, which walks the instability ranking and skips any image within +-2 frames (filename order) of an earlier pick in the same camera sequence.
- **Evidence for the diversity rule (label-blind check):** in the plain top 100, 42 picks are within 2 frames of another pick, and 73 come from subset 1 (47% of the pool). Filename order is used because the timestamps mix decimal separators, units and device clocks.
- **Outcome measures:** severe failures found (D10 rule), lift over the random mean, and p = share of random batches with at least as many failures. Reported for the whole pool and per subset.
- **Uncertainty:** the pool has only 8 camera sequences and neighboring frames are correlated, so consistency across the two review sites is the main robustness evidence; no sequence-level bootstrap is attempted with 8 clusters.
- **Also reported:** consistent misses (failure images ranked low by instability), with examples.
- **Status:** frozen and committed before the evaluation code is written or any review-pool label is read; run once.
