"""Run the local EEG models on a fixed train/validation/test split.

By default, load the three splits from data/stroke_hc/all beside this script.

Examples (from the EEG directory)::

    python experiment_runner.py --dry-run
    python experiment_runner.py --models EEGNet --device cuda:0
    python experiment_runner.py --exclude-channels O1 Oz O2 --device cuda:0
"""

import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote

from configs.data_configs import get_dataset_class


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "stroke_hc"
DEFAULT_MODEL_LIST = ("CodimNet", "ADformer", "EEGNet", "DBConformer")
REQUIRED_SPLITS = ("train", "val", "test")


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def validate_fixed_split(data_path, dataset):
    """Resolve the same data directory as dataloader_preprocessed."""
    base_dir = Path(data_path) / dataset
    if not base_dir.is_dir() and (Path(data_path) / "all").is_dir():
        base_dir = Path(data_path) / "all"
    missing = [
        str(base_dir / f"{split}_raw_eeg_data.pt")
        for split in REQUIRED_SPLITS
        if not (base_dir / f"{split}_raw_eeg_data.pt").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing fixed-split data files:\n" + "\n".join(missing)
        )
    return base_dir


def resolve_device(requested):
    if requested != "auto":
        return requested
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run EEG binary classification with a fixed train/val/test split."
    )
    parser.add_argument(
        "--data-path", "--data_path", type=Path, default=DEFAULT_DATA_PATH,
        help="Data root containing <dataset>/ or all/ with the three .pt files (default: data/stroke_hc).",
    )
    parser.add_argument("--dataset", default="all", help="Dataset config name.")
    parser.add_argument(
        "--exclude-channels", "--exclude_channels", nargs="+", default=[],
        help="Channel names to exclude from every split (case-insensitive).",
    )
    parser.add_argument(
        "--models", nargs="+", choices=DEFAULT_MODEL_LIST,
        default=list(DEFAULT_MODEL_LIST), help="Models to train sequentially.",
    )
    parser.add_argument("--seed-ids", "--seed_ids", nargs="+", type=int, default=[42])
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:<index>.")
    parser.add_argument(
        "--weights-root", "--weights_root", type=Path,
        default=PROJECT_ROOT / "weights",
    )
    parser.add_argument("--run-description", "--run_description", default="fixed_val")
    parser.add_argument(
        "--experiment-description", "--experiment_description", default=None,
        help="Output subdirectory; defaults to <data-path-name>_fixed_val.",
    )
    parser.add_argument(
        "--num-epochs", "--num_epochs", type=positive_int, default=None,
        help="Override num_epochs from configs/hparams.py for this run.",
    )
    parser.add_argument(
        "--batch-size", "--batch_size", type=positive_int, default=None,
        help="Override batch_size from configs/hparams.py for this run.",
    )
    parser.add_argument(
        "--rerun-completed", "--rerun_completed", action="store_true",
        help="Train again even when the test metrics file exists.",
    )
    parser.add_argument("--dry-run", "--dry_run", action="store_true",
                        help="Validate paths and print planned runs without training.")
    args = parser.parse_args(argv)
    excluded = [name.strip().casefold() for name in args.exclude_channels]
    if any(not name for name in excluded):
        parser.error("Excluded channel names must not be empty")
    if len(set(excluded)) != len(excluded):
        parser.error("Duplicate excluded channel names were requested")
    args.exclude_channels = excluded
    args.data_path = str(args.data_path.expanduser().resolve())
    args.weights_root = str(args.weights_root.expanduser().resolve())
    if args.experiment_description is None:
        args.experiment_description = f"{Path(args.data_path).name}_fixed_val"
    if args.exclude_channels:
        channel_tag = "_".join(
            quote(name, safe="").replace("_", "%5F") for name in args.exclude_channels
        )
        args.experiment_description += f"_drop_{channel_tag}"
    return args


def run_models_on_fixed_split(common_args):
    from trainer import trainer

    for seed_id in common_args.seed_ids:
        for model_name in common_args.models:
            run_args = argparse.Namespace(**vars(common_args))
            run_args.seed_id = seed_id
            run_args.models = model_name
            run_args.run_description = (
                f"{common_args.run_description}_seed_{seed_id}_{model_name}"
            )
            save_dir = (
                Path(run_args.weights_root)
                / run_args.experiment_description
                / run_args.run_description
            )
            if (save_dir / f"test_{model_name}.xlsx").is_file() and not common_args.rerun_completed:
                print(f"Skip completed: seed={seed_id}, model={model_name}", flush=True)
                continue

            runner = trainer(run_args)
            for key in ("num_epochs", "batch_size"):
                value = getattr(common_args, key)
                if value is not None:
                    runner.hparams[key] = value
            manifest = {
                "protocol": "fixed_train_val_test",
                "selection_set": "val",
                "final_evaluation_set": "test",
                "dataset": run_args.dataset,
                "data_path": run_args.data_path,
                "seed": seed_id,
                "model": model_name,
                "device": run_args.device,
                "excluded_channels": list(run_args.exclude_channels),
                "hparams": dict(runner.hparams),
            }
            manifest_path = save_dir / "run_manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"Start: seed={seed_id}, model={model_name}", flush=True)
            runner.train()
            ch_names = getattr(runner.dataset_configs, "ch_names", None)
            manifest.update({
                "ch_names": list(ch_names) if ch_names is not None else None,
                "input_channels": runner.dataset_configs.input_channels,
            })
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"Finished: seed={seed_id}, model={model_name}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    try:
        if len(set(args.models)) != len(args.models) or len(set(args.seed_ids)) != len(args.seed_ids):
            raise ValueError("Duplicate model names or seeds were requested")
        configs = get_dataset_class(args.dataset)()
        if configs.num_classes != 2:
            raise ValueError("This runner requires a binary dataset configuration")
        base_dir = validate_fixed_split(args.data_path, args.dataset)
        args.device = resolve_device(args.device)
        print("Fixed split: train -> validation checkpoint selection -> test evaluation")
        print(f"Dataset: {args.dataset}; classes: {configs.class_names}")
        print(f"Data: {base_dir}")
        print(f"Excluded channels: {args.exclude_channels or 'none (keep all)'}")
        print(f"Output: {Path(args.weights_root) / args.experiment_description}")
        print(f"Device: {args.device}")
        for seed_id in args.seed_ids:
            for model_name in args.models:
                print(f"Planned: seed={seed_id}, model={model_name}")
        if args.dry_run:
            return 0
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/eeg_matplotlib_cache")
        run_models_on_fixed_split(args)
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
