from __future__ import annotations

import argparse
import sys

from autovalidator.runner import run_validation


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    p = argparse.ArgumentParser(prog="autovalidator")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="Validate a dataset (CSV/Parquet)")
    v.add_argument("--dataset", required=True, help="Path to CSV/Parquet")
    v.add_argument("--out", required=True, help="Output directory for artifacts")
    v.add_argument("--sample-n", type=int, default=500, help="Sample size for profiling/validation")
    v.add_argument("--seed", type=int, default=42, help="Random seed")
    v.add_argument("--policy", default=None, help="Optional path to policy YAML")

    args = p.parse_args(argv)

    if args.cmd == "validate":
        res = run_validation(
            dataset_ref=args.dataset,
            out_dir=args.out,
            sample_n=args.sample_n,
            seed=args.seed,
            policy_path=args.policy,
        )
        print(f"run_id={res.run_id}")
        print(f"decision={res.decision}")
        print(f"artifacts_dir={res.run_dir}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
