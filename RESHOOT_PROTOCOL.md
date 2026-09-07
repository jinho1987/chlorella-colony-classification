# Re-shoot Protocol: Breaking the Strain/Imaging-Session Confound

## Why this exists

Every analysis in this repo (feature comparison, LOPO classifier, pairwise
tests, augmentation) traced back to the same root cause: in the current
data, **strain is perfectly confounded with imaging session** — all 3 plates
of one strain were photographed back-to-back, then the next strain, etc.
Nothing image-based can be trusted from that data no matter how it's
modeled. This protocol exists to make the *next* photo set actually usable.

The single design change that matters most is **#1 below (interleaving)**.
Everything else improves data quality, but interleaving is what makes the
data analyzable at all.

## 1. Interleave strains — never shoot in strain-batches

This is non-negotiable; it's the only thing that breaks the confound.

- Every imaging **session** (one sitting) must include plates from **all 4
  strains**, shot in **randomized order** within that session — not
  "all ALE, then all HMF."
- Spread plate replicates across **multiple sessions on different
  days**, so "which day" and "which strain" are no longer the same
  variable. E.g. with 12 plates/strain (see §5), don't shoot all 12 of one
  strain's plates in one day even if the order is randomized within that
  day — split each strain's replicates across at least 3–4 separate days,
  interleaved with the other strains on those same days.
- Randomize the order with an actual randomizer, not by hand — e.g. assign
  each plate a number, shuffle in a spreadsheet, shoot in that order. A
  worked example schedule is in §6.

## 2. Fix the imaging rig

Cell-phone photos are fine — the problem wasn't the phone, it was that
lighting/exposure drifted between the strain-batches. Fix that instead of
switching equipment:

- **Same phone, same physical position every time.** Mount it on a small
  tripod/phone clamp at a fixed height and angle looking straight down at
  the plate. Mark the tripod's floor position with tape so it's
  reproducible across sessions.
- **Lock exposure and white balance.** Most camera apps let you tap-and-hold
  to lock AE/AWB, or use manual/pro mode. Auto-exposure drifting between
  shots is very likely what produced the brightness confound we found.
  Set it once at the start of a session and don't let it re-auto-adjust
  between plates.
- **Constant, diffuse artificial light** — not daylight. A small photo
  light box/tent (~$20–30) with built-in LED lighting is the cheapest fix:
  it gives the same light every session regardless of time of day or
  weather, and diffuses it evenly so there's no hot spot or shadow that
  shifts with plate placement.
- **Same distance/zoom every shot** — don't use digital zoom; keep the
  phone-to-plate distance fixed (the tripod mount handles this).
- Turn off any "auto-enhance" / HDR / computational photography mode — it
  actively re-processes brightness/contrast per shot, which is exactly the
  kind of shot-to-shot variability that caused the original confound.

## 3. Put reference standards in every frame

- **A color/gray reference card** (an 18% gray card, or a cheap printed
  color-checker strip) placed in the same corner of every photo. This lets
  every photo be color-normalized after the fact against a known target —
  a safety net even if lighting isn't perfectly identical session to
  session.
- **A scale reference** (a small ruler or a printed scale bar) in frame, so
  colony size can be measured in real mm, not just relative pixels — pixel
  size depends on exact camera distance, which will drift slightly even
  with a tripod.

## 4. Blind the photographer to strain identity

Where practical, label plates with a coded ID (e.g. a random 4-digit
sticker) during the shoot, with the strain key kept in a separate log filled
in afterward. This prevents unconscious bias in framing, lighting
touch-ups, or which colonies get centered in frame — the same logic as
blinding in any other measurement.

## 5. More plates per strain, and consider spread plates

- Current data: 2–3 physical plates per strain. That's too few for a
  believable held-out-plate evaluation (leaving one out of two leaves a
  single training plate).
- **Target: at least 10–12 plates per strain**, so a leave-one-plate-out or
  k-fold-by-plate evaluation has enough held-out samples to be stable, and
  so plate-to-plate noise can be estimated with reasonable precision.
- Streak plates only yield a handful of well-isolated single colonies each
  (most of the plate is streak lines). A **dilution spread plate**
  (standard serial-dilution + spread technique) yields dozens to hundreds
  of well-separated single colonies per plate — this alone would multiply
  the usable single-colony image count severalfold from the same number of
  physical plates, independent of everything else in this protocol.

## 6. Standardize biology, not just imaging

Colony color/size genuinely does change with age and growth conditions —
controlling for these matters as much as controlling the camera:

- **Same media batch/lot** for agar across all strains and sessions (a new
  batch of media can shift background color and growth rate).
- **Same incubation conditions** (temperature, light, duration) — grow all
  4 strains' plates for a given round together in the same incubator run,
  not sequentially.
- **Fixed colony age at photo time** (e.g. exactly 5 days post-inoculation
  for every plate) — log the inoculation and imaging timestamps for every
  plate so this can be verified, not just assumed.

## 7. Log metadata per photo

A simple spreadsheet row per photo, filled in at shoot time:

`photo_id, blinded_plate_id, session_date, session_id, shoot_order_in_session, inoculation_date, incubation_temp, media_lot, operator, camera_settings_locked (Y/N)`

The strain-identity key (`blinded_plate_id → strain`) stays in a separate
file, merged in only at analysis time.

## Worked example: a 12-plates/strain, 4-session schedule

48 plates total (12 × 4 strains), spread across 4 sessions, 12 plates/session,
3 plates of each strain per session, shot in randomized order within the
session:

| Session (day) | Plates shot (randomized order, example) |
|---|---|
| 1 | HMF, HS2, ALE, HS19, HS19, HMF, ALE, HS2, HS19, ALE, HMF, HS2 |
| 2 | ALE, HS19, HMF, HS2, HMF, HS19, ALE, HS2, HS19, HMF, HS2, ALE |
| 3 | HS2, ALE, HS19, HMF, ALE, HS2, HMF, HS19, HMF, ALE, HS19, HS2 |
| 4 | HS19, HMF, HS2, ALE, HS2, HS19, ALE, HMF, ALE, HMF, HS19, HS2 |

(Re-randomize for the actual shoot — this table is illustrative of the
pattern, not a fixed schedule to reuse verbatim.)

## Pre-registered analysis plan for the new data

Decide this *before* looking at results, so the evaluation can't be
massaged after the fact:

1. Split by **plate**, never by colony, and never by shuffling colonies
   randomly across plates.
2. Reserve **2–3 plates per strain** as a held-out test set, chosen before
   any model is fit (e.g. randomly, logged in advance).
3. Evaluate with leave-one-plate-out (or k-fold-by-plate) cross-validation
   on the remaining plates, exactly as done in `scripts/train_classifier.py`
   — reuse that script unchanged as the evaluation harness.
4. Run the same permutation test (shuffle strain labels at the plate level,
   not the colony level) to confirm any accuracy is above the null
   distribution, not just above 1/4.
5. Only if that passes: check the model isn't attending to the reference
   card, plate edge, or any residual label/background artifact (Grad-CAM or
   an equivalent saliency check for a CNN; for the classical-feature
   approach, re-run the `explain_strains.py` confound-ratio check on the new
   data and confirm ratio > 1 for whatever feature the model is relying on).

## Minimal checklist for shoot day

- [ ] Tripod at marked position, phone locked (AE/AWB locked, HDR off)
- [ ] Light box on, same as last session
- [ ] Color card + scale ruler in frame
- [ ] Plates labeled with blinded ID only
- [ ] Shoot order pre-randomized and printed/on hand
- [ ] Metadata spreadsheet open and filled in per photo
- [ ] At least one plate from every strain in today's session
