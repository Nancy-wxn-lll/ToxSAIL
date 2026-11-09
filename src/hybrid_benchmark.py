from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.baselines import BASE_MODEL_NAMES, train_predict_baseline
from src.data.datasets import DataBundle, load_toxicity_data
from src.features.descriptors import RDKitDescriptorExtractor
from src.features.fingerprints import MorganFingerprintExtractor
from src.features.fragments import BRICSFragmentAlertExtractor
from src.features.fusion import fuse_feature_blocks
from src.features.toxic_alerts import ToxicAlertExtractor
from src.metrics import compute_prediction_metrics
from src.utils import save_json, set_seed


TOXICITY_DATASETS = [
    "hERG",
    "AMES",
    "DILI",
    "LD50",
    "Tox21",
    "ClinTox",
    "Skin-Reaction",
    "Carcinogens",
]

PRIMARY_MODEL_NAME = "toxsail"
LEGACY_MODEL_ALIASES = {"tox_mamba_hybrid": PRIMARY_MODEL_NAME, "toxalert_hybrid": PRIMARY_MODEL_NAME}
ENSEMBLE_MODEL_NAMES = {"ensemble_mean", PRIMARY_MODEL_NAME}
DERIVED_MODEL_NAMES = ENSEMBLE_MODEL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ToxSAIL benchmark with ECFP, descriptors, toxic alerts, BRICS fragment GPS features, and tree/boosting ensembles."
    )
    parser.add_argument("--datasets", nargs="+", default=TOXICITY_DATASETS)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgb", "lgbm", "extra_trees", PRIMARY_MODEL_NAME],
        help="Any of: xgb, lgbm, extra_trees, random_forest, ensemble_mean, toxsail.",
    )
    parser.add_argument("--split", default="scaffold", choices=["scaffold", "random", "standard"])
    parser.add_argument("--frac-train", type=float, default=0.8)
    parser.add_argument("--frac-valid", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--data-dir", default="data", help="Directory containing cached public .tab datasets.")
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--alert-set", default="expanded", choices=["basic", "expanded"])
    parser.add_argument("--fragment-vocab-size", type=int, default=256)
    parser.add_argument("--min-fragment-count", type=int, default=2)
    parser.add_argument(
        "--feature-set",
        default="ecfp_desc_alerts_gps",
        choices=[
            "ecfp",
            "desc",
            "alerts",
            "gps",
            "fragments",
            "ecfp_desc",
            "ecfp_alerts",
            "desc_alerts",
            "alerts_gps",
            "ecfp_gps",
            "ecfp_alerts_gps",
            "desc_alerts_gps",
            "ecfp_desc_alerts",
            "ecfp_desc_alerts_gps",
        ],
    )
    parser.add_argument(
        "--fusion",
        default="concat",
        choices=["concat", "alert_gated", "alert_interactions", "toxic_context"],
        help="Feature fusion strategy for ECFP/descriptors/toxic alerts.",
    )
    parser.add_argument("--target-normalize", default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.75)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--results-dir", default="results/hybrid_benchmark")
    parser.add_argument("--tables-dir", default="tables")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _normalize_model_names(model_names: List[str]) -> List[str]:
    return [LEGACY_MODEL_ALIASES.get(model_name, model_name) for model_name in model_names]


def _labels_and_masks(df: pd.DataFrame, label_cols: List[str]) -> tuple[np.ndarray, np.ndarray]:
    values = df[label_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    masks = np.isfinite(values).astype(np.float32)
    labels = np.nan_to_num(values, nan=0.0).astype(np.float32)
    return labels, masks


def _should_normalize_targets(args: argparse.Namespace, task_type: str) -> bool:
    if task_type != "regression":
        return False
    if args.target_normalize == "true":
        return True
    if args.target_normalize == "false":
        return False
    return True


def _normalize_targets(
    labels: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    enabled: bool,
) -> tuple[Dict[str, np.ndarray], np.ndarray | None, np.ndarray | None]:
    if not enabled:
        return {key: value.copy() for key, value in labels.items()}, None, None
    train = labels["train"].copy()
    train[masks["train"] < 0.5] = np.nan
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0).astype(np.float32)
    normalized = {}
    for split, values in labels.items():
        current = (values - mean.reshape(1, -1)) / std.reshape(1, -1)
        current[masks[split] < 0.5] = 0.0
        normalized[split] = current.astype(np.float32)
    return normalized, mean, std


def _inverse_normalize(preds: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    if mean is None or std is None:
        return preds
    return preds * std.reshape(1, -1) + mean.reshape(1, -1)


def _build_features(
    bundle: DataBundle,
    args: argparse.Namespace,
) -> tuple[Dict[str, np.ndarray], dict]:
    train_smiles = bundle.train_df[bundle.smiles_col].astype(str).tolist()
    smiles_by_split = {
        "train": train_smiles,
        "valid": bundle.valid_df[bundle.smiles_col].astype(str).tolist(),
        "test": bundle.test_df[bundle.smiles_col].astype(str).tolist(),
    }
    fp = MorganFingerprintExtractor(n_bits=args.n_bits, radius=args.radius)
    desc = RDKitDescriptorExtractor().fit(train_smiles)
    alerts = ToxicAlertExtractor(alert_set=args.alert_set)

    feature_tokens = set(args.feature_set.split("_"))
    use_ecfp = "ecfp" in feature_tokens
    use_desc = "desc" in feature_tokens
    use_alerts = "alerts" in feature_tokens
    use_gps = bool({"gps", "fragments", "brics"} & feature_tokens)
    fragments = None
    if use_gps:
        fragments = BRICSFragmentAlertExtractor(
            max_fragments=getattr(args, "fragment_vocab_size", 256),
            min_fragment_count=getattr(args, "min_fragment_count", 2),
            alert_set=args.alert_set,
        ).fit(train_smiles)

    blocks_by_split = {}
    for split, smiles in smiles_by_split.items():
        blocks = {}
        if use_ecfp:
            blocks["ecfp"] = fp.transform(smiles)
        if use_desc:
            blocks["descriptors"] = desc.transform(smiles)
        if use_alerts:
            blocks["alerts"] = alerts.transform(smiles)
        if use_gps:
            blocks["gps"] = fragments.transform(smiles)
        blocks_by_split[split] = blocks
    features, fusion_meta = fuse_feature_blocks(
        blocks_by_split,
        fusion=args.fusion,
        feature_set=args.feature_set,
    )
    meta = {
        "fingerprint_backend": fp.describe_backend(),
        "descriptor_backend": desc.describe_backend() if use_desc else "disabled",
        "alert_backend": alerts.describe_backend() if use_alerts else "disabled",
        "fragment_backend": fragments.describe_backend() if use_gps else "disabled",
        **fusion_meta,
    }
    return features, meta


def _prob_to_logits(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return np.log(probs / (1.0 - probs)).astype(np.float32)


def _metric_outputs(
    preds: np.ndarray,
    task_type: str,
    target_mean: np.ndarray | None,
    target_std: np.ndarray | None,
) -> np.ndarray:
    if task_type == "classification":
        return _prob_to_logits(preds)
    return _inverse_normalize(preds, target_mean, target_std)


def _evaluate_predictions(
    dataset: str,
    model_name: str,
    split: str,
    seed: int,
    preds: np.ndarray,
    labels_original: np.ndarray,
    masks: np.ndarray,
    bundle: DataBundle,
    task_type: str,
    target_mean: np.ndarray | None,
    target_std: np.ndarray | None,
    feature_meta: dict,
    source_model: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    outputs = _metric_outputs(preds, task_type, target_mean, target_std)
    macro, per_task = compute_prediction_metrics(
        outputs,
        labels_original,
        masks,
        bundle.label_cols,
        task_type=task_type,
    )
    row = {
        "seed": seed,
        "split": split,
        "dataset": dataset,
        "data_source": bundle.source,
        "model": model_name,
        "source_model": source_model or model_name,
        "task_type": task_type,
        "primary_metric": bundle.primary_metric,
        "metric_direction": bundle.metric_direction,
        "target_normalized": target_mean is not None,
        **feature_meta,
        **macro,
    }
    per_task = per_task.copy()
    per_task["seed"] = seed
    per_task["split"] = split
    per_task["dataset"] = dataset
    per_task["model"] = model_name
    per_task["source_model"] = source_model or model_name
    return row, per_task


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "macro_roc_auc",
        "macro_pr_auc",
        "macro_accuracy",
        "macro_f1",
        "macro_mae",
        "macro_rmse",
        "macro_spearman",
    ]
    rows = []
    group_cols = ["dataset", "model", "split", "task_type", "primary_metric", "metric_direction"]
    for keys, group in metrics.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = group["seed"].nunique()
        row["source_model"] = ",".join(sorted(set(group["source_model"].astype(str))))
        for col in numeric_cols:
            if col in group:
                mean = group[col].mean()
                std = group[col].std(ddof=0)
                row[f"{col}_mean"] = mean
                row[f"{col}_std"] = std
                row[f"{col}_mean_std"] = f"{mean:.4f} +/- {std:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def run_dataset_seed(
    dataset: str,
    seed: int,
    args: argparse.Namespace,
    requested_base_models: List[str],
    requested_derived_models: List[str],
) -> tuple[list[dict], list[pd.DataFrame], dict]:
    set_seed(seed)
    bundle = load_toxicity_data(
        dataset=dataset,
        split=args.split,
        seed=seed,
        data_dir=args.data_dir,
        allow_demo_fallback=False,
        frac_train=args.frac_train,
        frac_valid=args.frac_valid,
    )
    task_type = bundle.task_type
    labels_original, masks = {}, {}
    for split, df in [
        ("train", bundle.train_df),
        ("valid", bundle.valid_df),
        ("test", bundle.test_df),
    ]:
        labels_original[split], masks[split] = _labels_and_masks(df, bundle.label_cols)

    labels_for_fit, target_mean, target_std = _normalize_targets(
        labels_original,
        masks,
        enabled=_should_normalize_targets(args, task_type),
    )
    features, feature_meta = _build_features(bundle, args)

    predictions: dict[str, Dict[str, np.ndarray]] = {}
    metric_rows: list[dict] = []
    per_task_rows: list[pd.DataFrame] = []

    for model_name in requested_base_models:
        preds = train_predict_baseline(
            model_name,
            features,
            labels_for_fit,
            masks,
            task_type,
            seed,
            args,
        )
        predictions[model_name] = preds
        for split in ["train", "valid", "test"]:
            row, per_task = _evaluate_predictions(
                dataset,
                model_name,
                split,
                seed,
                preds[split],
                labels_original[split],
                masks[split],
                bundle,
                task_type,
                target_mean,
                target_std,
                feature_meta,
            )
            metric_rows.append(row)
            per_task_rows.append(per_task)

    requested_ensemble_names = [name for name in requested_derived_models if name in ENSEMBLE_MODEL_NAMES]
    if requested_ensemble_names and predictions:
        ensemble_preds = {}
        for split in ["train", "valid", "test"]:
            ensemble_preds[split] = np.mean(
                np.stack([preds[split] for preds in predictions.values()], axis=0),
                axis=0,
            ).astype(np.float32)
        source_model = ",".join(requested_base_models)
        for ensemble_name in requested_ensemble_names:
            predictions[ensemble_name] = ensemble_preds
            for split in ["train", "valid", "test"]:
                row, per_task = _evaluate_predictions(
                    dataset,
                    ensemble_name,
                    split,
                    seed,
                    ensemble_preds[split],
                    labels_original[split],
                    masks[split],
                    bundle,
                    task_type,
                    target_mean,
                    target_std,
                    feature_meta,
                    source_model=source_model,
                )
                metric_rows.append(row)
                per_task_rows.append(per_task)

    run_meta = {
        "dataset": dataset,
        "seed": seed,
        "task_type": task_type,
        "primary_metric": bundle.primary_metric,
        "metric_direction": bundle.metric_direction,
        "n_train": len(bundle.train_df),
        "n_valid": len(bundle.valid_df),
        "n_test": len(bundle.test_df),
        "tasks": bundle.label_cols,
        **feature_meta,
    }
    return metric_rows, per_task_rows, run_meta


def main() -> None:
    args = parse_args()
    args.models = _normalize_model_names(args.models)
    unknown = set(args.models) - BASE_MODEL_NAMES - DERIVED_MODEL_NAMES
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    requested_base_models = [model for model in args.models if model in BASE_MODEL_NAMES]
    requested_derived_models = [model for model in args.models if model in DERIVED_MODEL_NAMES]
    if not requested_base_models:
        raise ValueError("At least one base model is required.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.results_dir) / f"{stamp}_hybrid_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", vars(args))

    metric_rows: list[dict] = []
    per_task_frames: list[pd.DataFrame] = []
    run_rows: list[dict] = []
    for dataset in args.datasets:
        for seed in args.seeds:
            try:
                rows, per_task, meta = run_dataset_seed(
                    dataset,
                    seed,
                    args,
                    requested_base_models,
                    requested_derived_models,
                )
                metric_rows.extend(rows)
                per_task_frames.extend(per_task)
                run_rows.append({"dataset": dataset, "seed": seed, "status": "ok", **meta})
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                run_rows.append(
                    {"dataset": dataset, "seed": seed, "status": "failed", "error": repr(exc)}
                )

            metrics_df = pd.DataFrame(metric_rows)
            metrics_df.to_csv(output_dir / "metrics.csv", index=False)
            if per_task_frames:
                pd.concat(per_task_frames, ignore_index=True).to_csv(
                    output_dir / "per_task_metrics.csv", index=False
                )
            pd.DataFrame(run_rows).to_csv(output_dir / "runs.csv", index=False)
            if not metrics_df.empty:
                summary = _aggregate(metrics_df)
                summary.to_csv(output_dir / "metrics_summary.csv", index=False)
                summary.to_csv(tables_dir / "toxicity_hybrid_benchmark.csv", index=False)

    print(f"Saved hybrid benchmark to {output_dir}")


if __name__ == "__main__":
    main()
