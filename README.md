# Chlorella Sister-Strain Colony Classification

**Question:** Given cell-phone photos of streak-isolation plates for four Chlorella
strains (ALE, HMF, HS19, HS2) — all four derived from the same mother strain, **HS2**
— can single colonies be automatically identified on the plate, and can an
image-based model correctly classify which strain a given colony belongs to?

This repo tracks the first phase of that project: turning raw plate photos into a
clean, single-colony image dataset, and an honest first look at whether any
measurable signal separates the strains at all before committing to a CNN.

## Data

- 24 raw plate photos (not included here — unpublished lab photos): 4 strains ×
  3 physical plates × 2 photo angles (front/back of the dish).
- Each plate is a streak-isolation plate: dense streak lines plus a scatter of
  well-separated single colonies in the sparser zones.

## Pipeline (`scripts/segment_colonies.py`)

1. **Locate the dish**: separate the pale blue-gray background from the dish
   interior (HSV threshold), erode inward to avoid rim glare.
2. **Threshold colonies**: saturated green pixels in HSV space (colonies are
   strongly green; agar and background are desaturated).
3. **Connected components** on the green mask → candidate blobs.
4. **Shape filters**: plausible area, high circularity + solidity, bounded
   aspect ratio. This is what rejects handwritten labels and streak fragments
   — pen strokes are elongated or low-solidity, unlike a round colony.
5. **Crop-frame isolation check**: compute the exact crop window for a
   candidate, then reject it if *any other* detected blob (a neighboring
   colony, a streak fragment, or a stray ink mark) falls inside that window.
   This directly enforces "the saved image shows exactly one colony," rather
   than relying on an arbitrary isolation distance.
6. **QC overlays** (`qc_samples/`): every plate photo gets an overlay image
   with green boxes for accepted colonies and red boxes (labeled with the
   rejection reason) for everything discarded, so every automatic call is
   visually auditable.

A handful of remaining false positives (streak fragments, plate-edge shadows)
were caught by eye in the QC overlays and manually removed — see commit
history / `data/colony_manifest.csv`.

**Result: 53 clean single-colony crops** — HS2 (mother strain) = 9, ALE = 15,
HMF = 7, HS19 = 22 — from 24 source photos. See `figures/contact_sheet.jpg`
for a montage of every crop, and `qc_samples/` for example overlays.

## First look: is there any separable signal at all?

Before training any classifier, `scripts/analyze_features.py` and
`scripts/plot_features.py` compute simple, interpretable features per colony
(equivalent diameter, circularity, mean hue/saturation/brightness) and compare
them across strains (Kruskal-Wallis test). See
`figures/strain_feature_comparison.png`.

| feature | Kruskal-Wallis p-value |
|---|---|
| colony diameter | 0.091 |
| circularity | 0.870 |
| mean hue | 0.072 |
| mean saturation | 0.074 |
| **mean brightness (V)** | **0.0002** |

### Important caveat — this is very likely a batch/lighting artifact, not biology

The four strains' plates were each photographed in their own session (all 3
plates of one strain shot back-to-back). When brightness is broken down
**within** a single strain, by individual source photo, it varies almost as
much as it does **between** strains:

- HS2's own 3 plates: mean brightness 30.1 / 48.7 / 30.1
- ALE's own 4 plates: mean brightness 60.2 / 58.8 / 40.4 / 63.2

That range (±10-15 units within one strain, from lighting/exposure alone) is
comparable to the between-strain spread that produced the "significant"
p-value above. **A classifier trained on this data could easily learn to
recognize the photo session (lighting/exposure) instead of the strain**, and
would report deceptively high accuracy that would not generalize to a new
photo of the same strain taken on a different day.

## Conclusion so far

With only 3 physical plates per strain, all imaged in one uninterrupted
session per strain, **strain and imaging-batch are confounded** in this
dataset. No result from it — classical features or a CNN — can currently
distinguish "this is really the strain" from "this is really the lighting
that day." This needs to be fixed at the data-collection stage, not the
modeling stage.

## Recommended next steps

1. **Re-photograph with interleaved sessions**: shoot plates from different
   strains in the same sitting, under the same lighting/exposure, ideally
   with a color reference card in frame for white-balance correction.
2. **More plates per strain** (5-10+) so a train/test split can hold out
   entire plates per class — never split at the colony level, since colonies
   from the same plate are not independent samples.
3. Only once (1) and (2) are addressed does it make sense to train an image
   classifier (transfer learning on a small pretrained CNN, given the likely
   dataset size) and evaluate it on held-out plates, with a Grad-CAM/attention
   check to confirm it's attending to colony morphology and not to
   labels/background artifacts.

## Repo contents

- `scripts/` — the full processing pipeline (segmentation, feature
  extraction, plotting, QC contact sheets).
- `data/colony_manifest.csv` — every accepted colony crop with its source
  photo, bounding box, and shape metrics.
- `data/colony_features.csv` — per-colony size/color features used in the
  analysis above.
- `data/cropped_colonies/` — the 53 single-colony crops, by strain.
- `figures/` — the contact sheet and the strain comparison figure.
- `qc_samples/` — example QC overlays (one per strain) showing accepted vs.
  rejected candidates.
