"""Command-line training.

Simulated fleet (default):
    python -m pvm.train

Your own data — any CSV, any columns:
    python -m pvm.train --data my_fleet.csv --targets failed_next_month --rul km_to_failure
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .model import MaintenanceModel
from .schema import applicability, component_names, target_name
from .simulate import generate_fleet

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "vehicle_model.joblib"


def train_default(n: int = 40000, seed: int = 42, path: Path = DEFAULT_MODEL_PATH, progress=None) -> MaintenanceModel:
    df = generate_fleet(n, seed=seed)
    model = MaintenanceModel(
        classification_targets=[target_name(c) for c in component_names()],
        regression_target="rul_km",
        applicability=applicability(),
        group_col="powertrain",
        drop_cols=["vehicle_id"],
    )
    model.fit(df, progress=progress)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    (path.parent / "metrics.json").write_text(json.dumps(model.report_, indent=2))
    return model


def load_or_train(path: Path = DEFAULT_MODEL_PATH, progress=None) -> MaintenanceModel:
    if path.exists():
        try:
            return MaintenanceModel.load(path)
        except Exception:  # e.g. pickled with a different scikit-learn version
            pass
    return train_default(path=path, progress=progress)


def print_report(report: dict) -> None:
    print(f"\nRows: {report['n_rows']:,}   Features used: {report['n_features']}")
    print(f"{'target':28s} {'prev':>6s} {'ROC-AUC':>8s} {'PR-AUC':>7s} {'recall':>7s} {'precision':>9s}")
    for t, m in report["targets"].items():
        if "skipped" in m:
            print(f"{t:28s} skipped: {m['skipped']}")
            continue
        print(f"{t:28s} {m['prevalence']:6.1%} {m['roc_auc']:8.3f} {m['pr_auc']:7.3f} {m['recall']:7.1%} {m['precision']:9.1%}")
    if "regression" in report:
        r = report["regression"]
        print(f"\nRemaining life ({r['target']}): MAE {r['mae']:,.0f}   R² {r['r2']:.3f}   "
              f"80% interval coverage {r['interval_coverage']:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="CSV to train on (default: simulate a fleet)")
    ap.add_argument("--targets", help="comma-separated 0/1 label columns (required with --data)")
    ap.add_argument("--rul", help="optional numeric remaining-life column")
    ap.add_argument("--group", help="optional column to group baselines by")
    ap.add_argument("--drop", default="", help="comma-separated columns to ignore")
    ap.add_argument("--n", type=int, default=40000, help="simulated vehicles (default 40000)")
    ap.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    args = ap.parse_args()

    if args.data:
        if not args.targets:
            ap.error("--targets is required with --data")
        df = pd.read_csv(args.data)
        model = MaintenanceModel(
            classification_targets=[t.strip() for t in args.targets.split(",")],
            regression_target=args.rul, group_col=args.group,
            drop_cols=[c.strip() for c in args.drop.split(",") if c.strip()],
        )
        report = model.fit(df)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        model.save(args.out)
    else:
        model = train_default(n=args.n, path=Path(args.out),
                              progress=lambda f, msg: print(f"[{f:4.0%}] {msg}"))
        report = model.report_
    print_report(report)
    print(f"\nSaved model to {args.out}")


if __name__ == "__main__":
    main()
