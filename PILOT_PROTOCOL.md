# Pilot Protocol (Phase 1) — validate the fix before scaling up

Before committing to the full [RESHOOT_PROTOCOL.md](RESHOOT_PROTOCOL.md)
(48 plates), run this small pilot to cheaply answer two questions:

1. **Does interleaving + a controlled rig actually shrink the imaging-session
   confound?** (the thing that broke every analysis in this repo)
2. **Is the new shoot procedure logistically workable** — timing, blinded
   IDs, color card, metadata logging?

## Scale: 12 plates total (same order of magnitude as the original data)

- 4 strains × 3 plates each = **12 plates**, front+back = 24 photos — this
  deliberately matches the original dataset's plate count, so the pilot is a
  fair like-for-like test of "did fixing the *protocol* fix the problem,"
  not a bigger undertaking.
- Spread across **3 imaging sessions** (different days) — the minimum
  needed to get any real estimate of between-session variance at all. Each
  session shoots 1 plate per strain, in randomized order.

## One deliberate constraint: keep the plate technique identical

Use the same **streak-plate** technique as the original shoot, not spread
plates, for the pilot specifically. The pilot is testing one thing —
whether the imaging protocol (interleaving + rig) fixes the confound. Changing
the plate technique at the same time would confound *that* comparison.
Spread plates (for more colonies per plate) belong in the full-scale round,
as a separate, deliberate improvement — see RESHOOT_PROTOCOL.md §5.

## Rig checklist (same as the full protocol)

- [ ] Tripod at a marked, fixed position
- [ ] Exposure and white balance locked (not auto) for the whole session
- [ ] Diffuse LED light box, not daylight
- [ ] HDR / auto-enhance off
- [ ] Color reference card + scale ruler in every frame, same corner
- [ ] Plates labeled with blinded ID only (strain key filled in afterward)
- [ ] Metadata log open and filled in per photo (template below)

## Shoot order — actual randomized schedule for this pilot

Generated with `random.seed(20260907)` (documented here so it's
reproducible / auditable, not hand-picked):

| Session (day) | Shoot order |
|---|---|
| 1 | HS19 → HS2 → ALE → HMF |
| 2 | HS2 → ALE → HS19 → HMF |
| 3 | HS19 → ALE → HMF → HS2 |

Each entry = 1 plate of that strain, front+back photo, before moving to the
next. Use a fresh blinded ID sticker per plate; log the strain in the
separate key sheet, not on the plate itself.

## File naming convention

Once photos are organized post-shoot (strain re-attached from the blinded
key), use plain ASCII filenames so the analysis scripts can parse session
number automatically:

```
Colony picture pilot/<strain>/<strain>_s<session>_<front|back>.jpg
e.g. Colony picture pilot/ALE/ALE_s1_front.jpg
     Colony picture pilot/ALE/ALE_s1_back.jpg
     Colony picture pilot/ALE/ALE_s2_front.jpg  ...
```

This mirrors the original `<strain>/<strain>(앞/뒤)-<plate>.jpg` layout, just
with `s<session>` in place of the plate index, since session (not physical
plate identity) is the grouping variable that matters for the pilot's
confound check.

## Metadata log

Template at `data/pilot_metadata_template.csv` — one row per photo, columns:

`photo_id, blinded_plate_id, strain_key(fill in later, separately), session_id, session_date, shoot_order_in_session, side(front/back), inoculation_date, incubation_temp, media_lot, operator, ae_awb_locked(Y/N)`

## Analysis plan for the pilot (decide this now, before shooting)

1. Run the existing segmentation pipeline (`scripts/segment_colonies.py`) on
   the 24 pilot photos exactly as before — same thresholds, same manual QC
   pass on the overlays.
2. Run `scripts/analyze_pilot_session_confound.py` (added in this commit,
   ready to run the moment pilot photos are in) — it's
   `explain_strains.py`'s confound-ratio check, but grouped by **imaging
   session** instead of by plate, since session is now the thing crossed
   with strain rather than confounded with it.
3. **Directly compare the new confound ratios to the original ones**,
   feature by feature (the original table is in the main README). The
   comparison that matters:

   | Result | Interpretation |
   |---|---|
   | Ratios rise toward/above 1 for brightness & color features | Interleaving + rig worked — proceed to full shoot |
   | Ratios stay low, similar to before | Residual lighting drift even with the rig — revisit the rig (light box, AE/AWB lock) before scaling up |
4. Also directly check the rig mechanically: compute brightness variance
   *within* a session (should be small — that's what "locked exposure"
   buys you) vs. *between* sessions (reflects whatever residual drift is
   left, hopefully much smaller than the original ±10-15 unit swings).

**Do not attempt to train a classifier on the pilot data.** 12 plates is
still far too few for that; the pilot's only job is to validate the fix,
not to deliver a working model. If the pilot passes, the classifier attempt
happens after the full 48-plate shoot, per RESHOOT_PROTOCOL.md's
pre-registered analysis plan.

## Go / No-Go

- **Go** to the full 48-plate shoot if confound ratios clearly improve and
  the shoot logistics were workable.
- **Iterate** — fix the rig and re-pilot — if ratios are still low. Cheaper
  to find that out from a 12-plate pilot than after 48 plates.
