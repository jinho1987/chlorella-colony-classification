"""
Runs the exact same segmentation pipeline as segment_colonies.py, pointed at
the pilot photo set instead of the original one. No detection logic is
duplicated -- this just repoints segment_colonies's paths and calls its
main().

Expects photos organized as:
  colony_pics/Colony picture pilot/<strain>/<strain>_s<session>_<front|back>.jpg
(see PILOT_PROTOCOL.md "File naming convention")
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import segment_colonies as sc

sc.SRC_DIR = os.path.join(sc.ROOT, "Colony picture pilot")
sc.CROP_DIR = os.path.join(sc.ROOT, "cropped_colonies_pilot")
sc.QC_DIR = os.path.join(sc.ROOT, "qc_overlays_pilot")
sc.MANIFEST_PATH = os.path.join(sc.ROOT, "colony_manifest_pilot.csv")

if __name__ == "__main__":
    if not os.path.isdir(sc.SRC_DIR):
        print(f"Pilot photos not found at: {sc.SRC_DIR}")
        print("Organize pilot photos per PILOT_PROTOCOL.md's naming convention first, then re-run this.")
        sys.exit(1)
    sc.main()
