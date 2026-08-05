"""Command-line interface for data, training, comparison, and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import load_dataset, split_dataset, validate_dataset
from .download import download_dataset
from .evidence import verify_cuda_evidence
from .gpu_benchmark import run_gpu_benchmark
from .learning import run_interactive_check
from .modeling import load_bundle, predict_rows, train_evaluate_save
from .torch_model import train_torch_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="SensorGuard predictive-maintenance ML pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="download and verify the UCI dataset")
    download_parser.add_argument("--destination", type=Path, default=Path("data/raw"))
    download_parser.add_argument("--force", action="store_true")

    audit_parser = subparsers.add_parser("audit", help="validate and summarize the dataset")
    audit_parser.add_argument("--data", type=Path, default=Path("data/raw/ai4i2020.csv"))

    train_parser = subparsers.add_parser("train", help="train, select, and evaluate classical models")
    train_parser.add_argument("--data", type=Path, default=Path("data/raw/ai4i2020.csv"))
    train_parser.add_argument("--out", type=Path, default=Path("outputs/baseline"))
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--with-torch", action="store_true")
    train_parser.add_argument("--torch-epochs", type=int, default=40)

    predict_parser = subparsers.add_parser("predict", help="run batch predictions from a CSV file")
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--input", type=Path, required=True)
    predict_parser.add_argument("--out", type=Path, required=True)

    learn_parser = subparsers.add_parser("learn", help="complete the interactive ML-readiness check")
    learn_parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/learning-check/answers.json"),
    )

    gpu_parser = subparsers.add_parser(
        "gpu-benchmark", help="compare frozen CPU and CUDA XGBoost on train/validation"
    )
    gpu_parser.add_argument("--data", type=Path, default=Path("data/raw/ai4i2020.csv"))
    gpu_parser.add_argument(
        "--out", type=Path, default=Path("outputs/cuda-benchmark/report.json")
    )
    gpu_parser.add_argument("--random-state", type=int, default=42)
    gpu_parser.add_argument("--repeats", type=int, default=15)

    evidence_parser = subparsers.add_parser(
        "verify-evidence", help="audit the published CUDA benchmark evidence"
    )
    evidence_parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/evidence/cuda-colab-t4-report.json"),
    )

    args = parser.parse_args()
    if args.command == "download":
        path = download_dataset(args.destination, force=args.force)
        print(f"Dataset ready: {path}")
        return 0
    if args.command == "audit":
        frame = load_dataset(args.data)
        print(json.dumps(validate_dataset(frame), indent=2))
        return 0
    if args.command == "train":
        frame = load_dataset(args.data)
        audit = validate_dataset(frame)
        splits = split_dataset(frame, random_state=args.random_state)
        report = train_evaluate_save(splits, args.out, random_state=args.random_state)
        report["dataset_audit"] = audit
        if args.with_torch:
            report["torch_comparison"] = train_torch_comparison(
                splits,
                args.out,
                random_state=args.random_state,
                epochs=args.torch_epochs,
            )
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "experiment_summary.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        test = report["test_metrics"]
        print(
            f"Selected {report['selected_model']} at threshold {report['selected_threshold']:.2f}; "
            f"test precision={test['precision']:.3f} recall={test['recall']:.3f} "
            f"F1={test['f1']:.3f} AP={test['average_precision']:.3f}"
        )
        return 0
    if args.command == "predict":
        bundle = load_bundle(args.model)
        rows = pd.read_csv(args.input)
        results = predict_rows(bundle, rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.out, index=False)
        print(f"Wrote {len(results)} predictions to {args.out}")
        return 0
    if args.command == "learn":
        run_interactive_check(args.out)
        return 0
    if args.command == "gpu-benchmark":
        frame = load_dataset(args.data)
        splits = split_dataset(frame, random_state=args.random_state)
        report = run_gpu_benchmark(
            splits,
            args.out,
            random_state=args.random_state,
            repeats=args.repeats,
        )
        timing = report["timing_seconds"]
        print(
            f"Verified CUDA run; median CPU fit={timing['cpu_fit_median']:.4f}s, "
            f"CUDA fit={timing['cuda_fit_median']:.4f}s, "
            f"speedup={timing['cpu_over_cuda_speedup']:.3f}x. "
            f"Wrote {args.out}"
        )
        return 0
    if args.command == "verify-evidence":
        # A failed audit is a result, not a crash: report it plainly and exit
        # non-zero so CI and a human read the same sentence.
        try:
            result = verify_cuda_evidence(args.report)
        except (ValueError, KeyError) as error:
            print(f"Evidence rejected: {error}")
            return 1
        print(json.dumps(result, indent=2))
        return 0
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
