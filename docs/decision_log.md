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
