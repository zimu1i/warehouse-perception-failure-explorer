# Dataset card: LOCO (Logistics Objects in Context), annotated release v1

Status: draft. Facts below were verified locally on 2026-10-26 unless marked otherwise.

## Source and provenance

- Dataset: LOCO, Technical University of Munich (Mayershofer, Holm, Molter, Fottner, ICMLA 2020). Not MVTec LOCO AD.
- Repository: https://github.com/tum-fml/loco (last push 2024-05-08).
- Images: `dataset.zip` from https://go.mytum.de/239870, which redirects to TUM webdisk `LOCO\v1\dataset.zip`.
  - Size 769,055,500 bytes; SHA-256 `f3629d989071b824edc63e000f45ccc5826cfbb201a338f16ed87ab3e10bcdcf`.
- Annotations: `rgb/loco-all-v1.json` plus five per-subset files from the GitHub repository (COCO format).
- Capture: images were taken while walking through logistics environments with several low-cost cameras. They are **not** robot-mounted camera data.

## License

CC0-1.0 (public domain dedication) per the repository LICENSE and GitHub metadata. The authors request citation of the ICMLA 2020 paper. Not legal advice; re-check before redistributing images.

## Local layout

- `data/raw/dataset.zip`: original archive (kept).
- `data/raw/loco_v1/`: extracted archive plus annotation JSONs, made read-only. Raw inputs are never modified.

## Verified counts

| Item | Value |
|---|---|
| Image files in archive | 5,593 (all `.jpg`) |
| Images in `loco-all-v1.json` | 5,097 |
| Annotations | 151,428 boxes (all `iscrowd = false`) |
| Annotated images missing on disk | 0 |
| Archive images with no entry in the JSON | 496 (subset-3: 234, subset-5: 129, subset-1: 65, subset-2: 39, subset-4: 29) |
| Unreadable annotated images | 0 |
| Width/height mismatches vs JSON | 0 |
| Duplicate image or annotation IDs | 0 |
| Out-of-bounds or non-positive boxes | 0 |

README states 5,593 annotated images and 152,421 annotations. The archive matches the 5,593 file count, but the released JSON covers only 5,097 images and 151,428 boxes. Cause unknown; see "Open questions".

## Classes (COCO category id: name, box count)

- 7 pallet: 120,445
- 3 small_load_carrier: 22,151
- 10 stillage: 5,407
- 11 pallet_truck: 2,827
- 5 forklift: 598 (on 449 images; only 5 boxes in subset-1)

## Structure available for grouping

- `path` encodes subset, capture session (date_hour) and camera, e.g. `subset-4/2020-01-16_17/cam2/<timestamp>.jpg`.
- subset-1 has no session or camera folders.
- File names are Unix timestamps, giving temporal order within a session.
- Repository file names label subsets 2, 3, 5 as train and 1, 4 as val.
- "Subset = environment" is plausible from the paper's description (5 environments, 5 subsets) but **not yet verified**.

## Image properties

- Five resolutions: 1920x1080 (1,655), 1280x720 (1,460), 640x480 (805), 1280x960 (617), 1280x800 (560).
- Dense scenes: median 20 boxes/image, max 345; 27% of boxes are smaller than 32x32 px, 75% smaller than 96x96 px.

## Open questions

- Are the 496 unannotated archive images unlabeled, deliberately excluded, or images with zero objects? The JSON cannot tell us, so they must not be treated as "no objects present". Handling: see decision D2 (kept in manifest as `unannotated`, excluded from scoring).
