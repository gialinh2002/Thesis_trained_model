"""Step 1: download FitRec -> build user profiles -> build/load exercise catalog.

  python scripts/01_prepare_data.py                 # download + process
  python scripts/01_prepare_data.py --synthetic     # skip FitRec, use synthetic users
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gymrl.config import load_config
from gymrl.data.catalog import load_catalog
from gymrl.data.fitrec import build_profiles, download, synthetic_profiles

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--synthetic", action="store_true")
args = ap.parse_args()
cfg = load_config(args.config)
d = cfg["data"]

if args.synthetic:
    prof = synthetic_profiles(seed=cfg["seed"])
else:
    raw = download(d["fitrec_url"], d["raw_path"])
    prof = build_profiles(raw, d["max_workouts"], d["min_workouts_per_user"], cfg["seed"])

Path(d["profiles_out"]).parent.mkdir(parents=True, exist_ok=True)
prof.to_parquet(d["profiles_out"], index=False)
cat = load_catalog(d["catalog_path"], d["catalog_size"])
print(f"profiles: {len(prof)} users -> {d['profiles_out']}")
print(prof.describe(include='all').T.head(20))
print(f"catalog: {len(cat)} exercises -> {d['catalog_path']}")
