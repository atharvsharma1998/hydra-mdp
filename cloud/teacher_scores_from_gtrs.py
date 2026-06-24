"""Use GTRS's released teacher scores (navtrain_8192.pkl) instead of recomputing.

GTRS ships one big dict {token: {sub_score: (8192,) array, ...}} scored against
traj_final/8192.npy. Our trainer reads one pkl per token from a cache dir. This
tool (a) VERIFIES the released scores were computed against the same vocab as
ours, and (b) CONVERTS the big pkl into the per-token cache the trainer expects.

Verify (compare against a few self-generated scores):
    python cloud/teacher_scores_from_gtrs.py verify \
        --gtrs-pkl navtrain_8192.pkl --self-cache <dir of self-generated <token>.pkl>

Convert (split into per-token cache the trainer loads):
    python cloud/teacher_scores_from_gtrs.py convert \
        --gtrs-pkl navtrain_8192.pkl --output-dir $NAVSIM_WS/teacher_scores_cache
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

# Sub-scores our PlanningHead distillation actually consumes.
NEEDED_KEYS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "driving_direction_compliance",
    "lane_keeping",
    "traffic_light_compliance",
]


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def verify(args):
    big = _load(args.gtrs_pkl)
    self_dir = Path(args.self_cache)
    common = [p.stem for p in self_dir.glob("*.pkl") if p.stem in big]
    print(f"big pkl tokens: {len(big)} | self-generated: {len(list(self_dir.glob('*.pkl')))} | common: {len(common)}")
    if not common:
        print("NO COMMON TOKENS — generate a few navtrain tokens first "
              "(precompute_teacher_scores.py --sensor-blobs-path ... --limit 10).")
        return
    worst = 1.0
    for tok in common[: args.num or 10]:
        ours, theirs = _load(self_dir / f"{tok}.pkl"), big[tok]
        print(f"\ntoken {tok}")
        for k in NEEDED_KEYS:
            a = np.asarray(ours[k]).ravel().astype(np.float64)
            b = np.asarray(theirs[k]).ravel().astype(np.float64)
            if a.shape != b.shape:
                print(f"  {k:32s} SHAPE MISMATCH {a.shape} vs {b.shape}")
                worst = -1
                continue
            corr = np.corrcoef(a, b)[0, 1] if a.std() > 1e-9 and b.std() > 1e-9 else 1.0
            exact = float((np.abs(a - b) < 1e-3).mean())
            worst = min(worst, corr)
            print(f"  {k:32s} corr={corr:.4f} exact_frac={exact:.3f} maxdiff={np.abs(a-b).max():.3f}")
    print("\n" + ("=" * 60))
    if worst > 0.99:
        print(f"VERDICT: MATCH (min corr {worst:.4f}) -> safe to use the GTRS scores. Run `convert`.")
    elif worst < 0:
        print("VERDICT: SHAPE MISMATCH -> vocab differs. Do NOT use; recompute instead.")
    else:
        print(f"VERDICT: LOW CORRELATION (min {worst:.4f}) -> likely different vocab/scorer. "
              "Inspect before using; recompute is the safe fallback.")


def convert(args):
    big = _load(args.gtrs_pkl)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dt = np.float16 if args.float16 else np.float32
    items = list(big.items())
    if args.limit:
        items = items[: args.limit]
    written, skipped = 0, 0
    for i, (tok, sc) in enumerate(items):
        f = out / f"{tok}.pkl"
        if f.exists() and not args.overwrite:
            skipped += 1
            continue
        missing = [k for k in NEEDED_KEYS if k not in sc]
        if missing:
            raise KeyError(f"token {tok} missing keys {missing}")
        d = {k: np.asarray(sc[k], dtype=dt) for k in sc.keys()}
        with open(f, "wb") as fh:
            pickle.dump(d, fh)
        written += 1
        if i % 5000 == 0:
            print(f"  {i}/{len(items)} (written={written}, skipped={skipped})")
    print(f"done -> {out} | written={written} skipped={skipped} (dtype={dt.__name__})")


def main():
    ap = argparse.ArgumentParser(description="Verify/convert GTRS released teacher scores")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify")
    v.add_argument("--gtrs-pkl", required=True)
    v.add_argument("--self-cache", required=True, help="dir of self-generated <token>.pkl")
    v.add_argument("--num", type=int, default=10)
    v.set_defaults(func=verify)

    c = sub.add_parser("convert")
    c.add_argument("--gtrs-pkl", required=True)
    c.add_argument("--output-dir", required=True)
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--float16", action="store_true", help="store half precision (~halves disk)")
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=convert)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
