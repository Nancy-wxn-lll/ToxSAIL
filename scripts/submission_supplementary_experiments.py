from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import train_predict_baseline
from src.data.admet_benchmark import get_dataset_spec
from src.data.datasets import DataBundle, load_toxicity_data
from src.data.splitting import scaffold_key, split_dataframe
from src.features.descriptors import RDKitDescriptorExtractor
from src.features.fingerprints import MorganFingerprintExtractor
from src.features.fragments import BRICSFragmentAlertExtractor, FRAGMENT_STAT_NAMES
from src.features.fusion import fuse_feature_blocks
from src.features.toxic_alerts import SMARTS_PATTERNS, ToxicAlertExtractor
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
CLASSIFICATION_DATASETS = [
    "hERG",
    "AMES",
    "DILI",
    "Tox21",
    "ClinTox",
    "Skin-Reaction",
    "Carcinogens",
]
BASE_MODELS = ["xgb", "lgbm", "extra_trees"]
ENSEMBLE_SPECS = {
    "xgb": ("xgb",),
    "lgbm": ("lgbm",),
    "extra_trees": ("extra_trees",),
    "xgb_lgbm_mean": ("xgb", "lgbm"),
    "xgb_extra_trees_mean": ("xgb", "extra_trees"),
    "lgbm_extra_trees_mean": ("lgbm", "extra_trees"),
    "toxsail": ("xgb", "lgbm", "extra_trees"),
}
LOCAL_DATA_FILES = {
    "herg": "data/herg.tab",
    "ames": "data/ames.tab",
    "dili": "data/dili.tab",
    "ld50": "data/ld50_zhu.tab",
    "tox21": "data/tox21.tab",
    "clintox": "data/clintox.tab",
    "skin-reaction": "data/skin_reaction.tab",
    "carcinogens": "data/carcinogens_lagunin.tab",
}
SMILES_COLUMN_CANDIDATES = ["Drug", "smiles", "SMILES", "X", "drug", "molecule", "Mol", "mol"]
ID_COLUMN_CANDIDATES = {"Drug_ID", "drug_id", "id", "ID", "name", "Name"}
ALERT_MEANINGS = {
    "nitro_group": "Nitro-containing electrophilic or redox-active motif, frequently associated with mutagenicity alerts.",
    "aromatic_amine": "Aromatic amine motif that can be bioactivated to reactive intermediates.",
    "epoxide": "Strained electrophilic three-membered ether with potential covalent reactivity.",
    "aldehyde": "Carbonyl electrophile that can form reversible or covalent adducts with biomolecules.",
    "michael_acceptor": "Conjugated electrophile capable of nucleophilic addition.",
    "azo_group": "Azo linkage associated with reductive metabolism and aromatic amine release.",
    "hydrazine": "Hydrazine/hydrazide-like nitrogen motif associated with reactive metabolite liability.",
    "sulfonyl_halide": "Highly electrophilic sulfonyl halide group.",
    "alkyl_halide": "Alkyl halide electrophile with potential alkylating activity.",
    "anilide": "Aromatic amide context related to aniline-like structural alerts.",
    "polyhalogenated_aromatic_ring": "Halogen-rich aromatic motif associated with persistence and lipophilicity.",
    "quinone": "Redox-active quinone motif that can participate in oxidative stress and Michael addition.",
    "phenol": "Phenolic aromatic motif; may indicate redox or conjugation-related liabilities depending on context.",
    "catechol": "Catechol motif with redox cycling and metal-binding potential.",
    "thiophene": "Thiophene motif with known bioactivation liability in some chemical contexts.",
    "furan": "Furan motif with bioactivation liability in some chemical contexts.",
    "acyl_halide": "Reactive acylating functional group.",
    "isocyanate": "Reactive isocyanate electrophile.",
    "isothiocyanate": "Reactive isothiocyanate electrophile.",
    "aziridine": "Strained electrophilic nitrogen heterocycle.",
    "alkyne": "Alkyne motif; context-dependent bioactivation or off-target liability marker.",
    "aromatic_n_oxide": "Aromatic N-oxide motif with redox/metabolic relevance.",
    "hydroxylamine": "Hydroxylamine motif related to nitroso/nitro redox chemistry.",
    "nitroso": "Nitroso motif associated with reactive nitrogen chemistry.",
    "peroxide": "Peroxide motif associated with oxidative reactivity.",
    "carboxylic_acid": "Ionizable acid motif; included as contextual toxicity/ADMET feature rather than direct alert.",
    "aniline_unsubstituted": "Unsubstituted aniline motif with possible metabolic activation liability.",
    "halogenated_alkene": "Halogenated alkene electrophile/reactive metabolite motif.",
    "alpha_beta_unsaturated_carbonyl": "Conjugated carbonyl Michael acceptor motif.",
}


def dataset_key(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and audit supplementary experiments requested for the ToxSAIL submission checklist."
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "dependencies",
            "gps_audit",
            "ensemble_ablation",
            "strict_conformal",
            "recalibration",
            "mmp",
            "figure6_audit",
        ],
    )
    parser.add_argument("--datasets", nargs="+", default=TOXICITY_DATASETS)
    parser.add_argument("--classification-datasets", nargs="+", default=CLASSIFICATION_DATASETS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--split", default="scaffold", choices=["scaffold", "random", "standard"])
    parser.add_argument("--frac-train", type=float, default=0.8)
    parser.add_argument("--frac-valid", type=float, default=0.1)
    parser.add_argument("--four-fracs", type=float, nargs=4, default=[0.7, 0.1, 0.1, 0.1])
    parser.add_argument("--feature-set", default="ecfp_desc_alerts_gps")
    parser.add_argument("--fusion", default="concat")
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--alert-set", default="expanded", choices=["basic", "expanded"])
    parser.add_argument("--fragment-vocab-size", type=int, default=256)
    parser.add_argument("--min-fragment-count", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.75)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--target-normalize", default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--conformal-coverages", type=float, nargs="+", default=[0.8, 0.9, 0.95])
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--mmp-primary-threshold", type=float, default=0.6)
    parser.add_argument("--mmp-fallback-threshold", type=float, default=0.2)
    parser.add_argument("--tables-dir", default="tables/submission_supplement")
    parser.add_argument("--results-dir", default="results/submission_supplement")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _enabled(args: argparse.Namespace, section: str) -> bool:
    return "all" in args.sections or section in args.sections


def _safe_module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def dependency_audit(args: argparse.Namespace, output_dir: Path, tables_dir: Path) -> pd.DataFrame:
    rows = []
    for module in ["rdkit", "tdc", "mamba_ssm", "xgboost", "lightgbm", "shap", "torch", "sklearn"]:
        rows.append({"module": module, "available": _safe_module_available(module)})
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "dependency_audit.csv", index=False)
    df.to_csv(tables_dir / "dependency_audit.csv", index=False)
    return df


def _read_full_dataset(dataset: str) -> tuple[pd.DataFrame, str, list[str], str]:
    key = dataset_key(dataset)
    path = LOCAL_DATA_FILES.get(key)
    if path and Path(path).exists():
        df = pd.read_csv(path, sep="\t")
        smiles_col = next((col for col in SMILES_COLUMN_CANDIDATES if col in df.columns), None)
        if smiles_col is None:
            raise ValueError(f"Could not find a SMILES column in {path}.")
        excluded = set(ID_COLUMN_CANDIDATES) | {smiles_col}
        label_cols = []
        for col in df.columns:
            if col in excluded:
                continue
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().any():
                df[col] = numeric
                label_cols.append(col)
        if not label_cols:
            raise ValueError(f"Could not infer labels from {path}.")
        df = df[[smiles_col] + label_cols].dropna(subset=[smiles_col]).copy()
        df[smiles_col] = df[smiles_col].astype(str)
        return df, smiles_col, label_cols, f"local:{path}"
    bundle = load_toxicity_data(dataset=dataset, split="random", seed=0, allow_demo_fallback=False)
    df = pd.concat([bundle.train_df, bundle.valid_df, bundle.test_df], ignore_index=True)
    return df, bundle.smiles_col, bundle.label_cols, bundle.source


def load_bundle(
    dataset: str,
    split: str,
    seed: int,
    frac_train: float,
    frac_valid: float,
) -> DataBundle:
    df, smiles_col, label_cols, source = _read_full_dataset(dataset)
    train_df, valid_df, test_df = split_dataframe(
        df,
        smiles_col,
        split=split,
        seed=seed,
        frac_train=frac_train,
        frac_valid=frac_valid,
    )
    spec = get_dataset_spec(dataset)
    return DataBundle(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        smiles_col=smiles_col,
        label_cols=label_cols,
        dataset_name=spec.display_name if spec is not None else dataset,
        source=source,
        task_type=spec.task_type if spec is not None else "classification",
        primary_metric=spec.primary_metric if spec is not None else "roc_auc",
        metric_direction=spec.direction if spec is not None else "max",
        category="Toxicity",
        tdc_name=spec.tdc_name if spec is not None else dataset,
    )


def split_dataframe_four_way(
    df: pd.DataFrame,
    smiles_col: str,
    split: str,
    seed: int,
    fracs: Iterable[float],
) -> dict[str, pd.DataFrame]:
    frac_values = np.asarray(list(fracs), dtype=float)
    frac_values = frac_values / frac_values.sum()
    names = ["train", "valid", "calibration", "test"]
    df = df.reset_index(drop=True)
    if split in {"random", "standard"}:
        rng = np.random.default_rng(seed)
        indices = np.arange(len(df))
        rng.shuffle(indices)
        cuts = np.cumsum(frac_values[:-1] * len(df)).round().astype(int)
        chunks = np.split(indices, cuts)
        return {name: df.iloc[sorted(chunk)].reset_index(drop=True) for name, chunk in zip(names, chunks)}

    groups: dict[str, list[int]] = {}
    for idx, smiles in enumerate(df[smiles_col].tolist()):
        groups.setdefault(scaffold_key(smiles), []).append(idx)
    rng = np.random.default_rng(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    target = frac_values * len(df)
    assigned = {name: [] for name in names}
    counts = {name: 0 for name in names}
    for _, indices in group_items:
        deficits = {
            name: (target[idx] - counts[name]) / max(target[idx], 1.0)
            for idx, name in enumerate(names)
        }
        selected = max(names, key=lambda name: deficits[name])
        assigned[selected].extend(indices)
        counts[selected] += len(indices)

    # Keep all four splits non-empty when the dataset permits it.
    for name in names:
        if assigned[name]:
            continue
        donor = max(names, key=lambda item: len(assigned[item]))
        if len(assigned[donor]) > 1:
            assigned[name].append(assigned[donor].pop())

    return {name: df.iloc[sorted(indices)].reset_index(drop=True) for name, indices in assigned.items()}


def _target_normalize_enabled(args: argparse.Namespace, task_type: str) -> bool:
    if task_type != "regression":
        return False
    if args.target_normalize == "true":
        return True
    if args.target_normalize == "false":
        return False
    return True


def _labels_and_masks(df: pd.DataFrame, label_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = df[label_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    masks = np.isfinite(values).astype(np.float32)
    labels = np.nan_to_num(values, nan=0.0).astype(np.float32)
    return labels, masks


def _normalize_targets(
    labels: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    enabled: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray | None, np.ndarray | None]:
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


def _prob_to_logits(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return np.log(probs / (1.0 - probs)).astype(np.float32)


def _labels_masks_for_frames(frames: dict[str, pd.DataFrame], label_cols: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    labels = {}
    masks = {}
    for name, frame in frames.items():
        labels[name], masks[name] = _labels_and_masks(frame, label_cols)
    return labels, masks


def build_features_for_frames(
    frames: dict[str, pd.DataFrame],
    smiles_col: str,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict]:
    smiles_by_split = {
        split: frame[smiles_col].astype(str).tolist()
        for split, frame in frames.items()
    }
    train_smiles = smiles_by_split["train"]
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
            max_fragments=args.fragment_vocab_size,
            min_fragment_count=args.min_fragment_count,
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
        if use_gps and fragments is not None:
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
        "fragment_backend": fragments.describe_backend() if use_gps and fragments is not None else "disabled",
        "fragment_vocab_size": len(getattr(fragments, "vocabulary_", [])) if fragments is not None else 0,
        **fusion_meta,
    }
    return features, meta


def _train_base_predictions(
    dataset: str,
    seed: int,
    frames: dict[str, pd.DataFrame],
    smiles_col: str,
    label_cols: list[str],
    task_type: str,
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray | None, np.ndarray | None, dict]:
    labels, masks = _labels_masks_for_frames(frames, label_cols)
    labels_for_fit, target_mean, target_std = _normalize_targets(
        labels,
        masks,
        enabled=_target_normalize_enabled(args, task_type),
    )
    features, feature_meta = build_features_for_frames(frames, smiles_col, args)
    predictions = {}
    for model_name in BASE_MODELS:
        predictions[model_name] = train_predict_baseline(
            model_name,
            features,
            labels_for_fit,
            masks,
            task_type,
            seed,
            args,
        )
    return predictions, labels, masks, target_mean, target_std, feature_meta


def _combo_prediction(base_predictions: dict[str, dict[str, np.ndarray]], members: tuple[str, ...], split: str) -> np.ndarray:
    return np.mean([base_predictions[member][split] for member in members], axis=0).astype(np.float32)


def _metric_outputs(preds: np.ndarray, task_type: str, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    if task_type == "classification":
        return _prob_to_logits(preds)
    return _inverse_normalize(preds, mean, std)


def _format_mean_std(mean: float, std: float) -> str:
    return "" if not np.isfinite(mean) else f"{mean:.4f} +/- {std:.4f}"


def _aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "macro_roc_auc",
        "macro_pr_auc",
        "macro_accuracy",
        "macro_f1",
        "macro_mae",
        "macro_rmse",
        "macro_spearman",
    ]
    rows = []
    group_cols = ["dataset", "task_type", "model", "split", "primary_metric", "metric_direction"]
    for keys, group in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique())
        for col in metric_cols:
            if col not in group:
                continue
            values = pd.to_numeric(group[col], errors="coerce")
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_mean_std"] = _format_mean_std(mean, std)
            if col == row["primary_metric"] or f"macro_{row['primary_metric']}" == col:
                if row["metric_direction"] == "min":
                    worst = values.max()
                    best = values.min()
                else:
                    worst = values.min()
                    best = values.max()
                row["primary_metric_worst_seed_value"] = float(worst)
                row["primary_metric_best_seed_value"] = float(best)
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_tests(df: pd.DataFrame, reference_model: str, comparator_models: list[str]) -> pd.DataFrame:
    from scipy.stats import ttest_rel, wilcoxon

    rows = []
    metrics = ["macro_roc_auc", "macro_pr_auc", "macro_mae", "macro_rmse", "macro_spearman"]
    test_df = df[df["split"] == "test"].copy()
    for dataset, dset in test_df.groupby("dataset"):
        for comparator in comparator_models:
            ref = dset[dset["model"] == reference_model].set_index("seed")
            cmp = dset[dset["model"] == comparator].set_index("seed")
            shared = sorted(set(ref.index) & set(cmp.index))
            if len(shared) < 2:
                continue
            task_type = str(ref.iloc[0]["task_type"])
            for metric in metrics:
                if metric not in ref or metric not in cmp:
                    continue
                a = pd.to_numeric(ref.loc[shared, metric], errors="coerce").to_numpy(dtype=float)
                b = pd.to_numeric(cmp.loc[shared, metric], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(a) & np.isfinite(b)
                if valid.sum() < 2:
                    continue
                delta = a[valid] - b[valid]
                try:
                    t_p = float(ttest_rel(a[valid], b[valid]).pvalue)
                except Exception:
                    t_p = np.nan
                try:
                    w_p = float(wilcoxon(delta).pvalue) if np.any(np.abs(delta) > 1e-12) else np.nan
                except Exception:
                    w_p = np.nan
                rows.append(
                    {
                        "dataset": dataset,
                        "task_type": task_type,
                        "reference_model": reference_model,
                        "comparator_model": comparator,
                        "metric": metric,
                        "n_pairs": int(valid.sum()),
                        "delta_reference_minus_comparator_mean": float(np.mean(delta)),
                        "delta_reference_minus_comparator_std": float(np.std(delta, ddof=0)),
                        "paired_ttest_p": t_p,
                        "wilcoxon_p": w_p,
                    }
                )
    return pd.DataFrame(rows)


def run_ensemble_ablation(args: argparse.Namespace, output_dir: Path, tables_dir: Path) -> None:
    from src.metrics import compute_prediction_metrics

    rows = []
    per_task_frames = []
    run_rows = []
    for dataset in args.datasets:
        spec = get_dataset_spec(dataset)
        if spec is None:
            continue
        for seed in args.seeds:
            try:
                set_seed(seed)
                bundle = load_bundle(dataset, args.split, seed, args.frac_train, args.frac_valid)
                frames = {"train": bundle.train_df, "valid": bundle.valid_df, "test": bundle.test_df}
                base_predictions, labels, masks, target_mean, target_std, feature_meta = _train_base_predictions(
                    dataset,
                    seed,
                    frames,
                    bundle.smiles_col,
                    bundle.label_cols,
                    bundle.task_type,
                    args,
                )
                for model_name, members in ENSEMBLE_SPECS.items():
                    for split in ["valid", "test"]:
                        preds = _combo_prediction(base_predictions, members, split)
                        outputs = _metric_outputs(preds, bundle.task_type, target_mean, target_std)
                        macro, per_task = compute_prediction_metrics(
                            outputs,
                            labels[split],
                            masks[split],
                            bundle.label_cols,
                            task_type=bundle.task_type,
                        )
                        row = {
                            "dataset": dataset,
                            "seed": seed,
                            "split": split,
                            "model": model_name,
                            "members": "+".join(members),
                            "task_type": bundle.task_type,
                            "primary_metric": bundle.primary_metric,
                            "metric_direction": bundle.metric_direction,
                            "data_source": bundle.source,
                            "n_train": len(bundle.train_df),
                            "n_valid": len(bundle.valid_df),
                            "n_test": len(bundle.test_df),
                            **feature_meta,
                            **macro,
                        }
                        rows.append(row)
                        per_task = per_task.copy()
                        per_task["dataset"] = dataset
                        per_task["seed"] = seed
                        per_task["split"] = split
                        per_task["model"] = model_name
                        per_task["members"] = "+".join(members)
                        per_task_frames.append(per_task)
                run_rows.append({"dataset": dataset, "seed": seed, "status": "ok"})
                print(f"OK ensemble_ablation {dataset} seed={seed}", flush=True)
            except Exception as exc:
                run_rows.append({"dataset": dataset, "seed": seed, "status": "failed", "error": repr(exc)})
                print(f"FAILED ensemble_ablation {dataset} seed={seed}: {exc!r}", flush=True)
                if not args.continue_on_error:
                    raise
            pd.DataFrame(run_rows).to_csv(output_dir / "ensemble_model_ablation_runs.csv", index=False)
            if rows:
                pd.DataFrame(rows).to_csv(output_dir / "ensemble_model_ablation_per_seed.csv", index=False)

    metrics = pd.DataFrame(rows)
    per_task = pd.concat(per_task_frames, ignore_index=True) if per_task_frames else pd.DataFrame()
    summary = _aggregate_metrics(metrics) if not metrics.empty else pd.DataFrame()
    paired = _paired_tests(
        metrics,
        reference_model="toxsail",
        comparator_models=[name for name in ENSEMBLE_SPECS if name != "toxsail"],
    )
    for name, df in [
        ("ensemble_model_ablation_per_seed.csv", metrics),
        ("ensemble_model_ablation_per_task.csv", per_task),
        ("ensemble_model_ablation_summary.csv", summary),
        ("ensemble_model_ablation_paired_tests.csv", paired),
    ]:
        df.to_csv(output_dir / name, index=False)
        df.to_csv(tables_dir / name, index=False)


def _finite_quantile(scores: np.ndarray, coverage: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return np.nan
    rank = int(math.ceil((len(scores) + 1) * coverage))
    rank = min(max(rank, 1), len(scores))
    return float(np.sort(scores)[rank - 1])


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
    if len(y_true) == 0:
        return np.nan
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    value = 0.0
    for idx in range(n_bins):
        left, right = edges[idx], edges[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if mask.any():
            value += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(value)


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> dict:
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    out = {
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "brier": float(np.mean((y_prob - y_true) ** 2)) if len(y_true) else np.nan,
        "ece": _ece(y_true, y_prob, n_bins),
        "nll": np.nan,
    }
    if len(y_true):
        try:
            out["nll"] = float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6), labels=[0, 1]))
        except Exception:
            pass
    if len(np.unique(y_true)) >= 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            pass
    return out


def _similarity_bins(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.asarray([], dtype=object)
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    bins = np.full(len(values), "medium", dtype=object)
    bins[values < q1] = "low"
    bins[values >= q2] = "high"
    return bins


def _nearest_tanimoto(train_bits: np.ndarray, test_bits: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    train = train_bits.astype(np.float32)
    test = test_bits.astype(np.float32)
    train_sum = train.sum(axis=1)
    out = np.zeros(test.shape[0], dtype=np.float32)
    for start in range(0, test.shape[0], chunk_size):
        cur = test[start : start + chunk_size]
        inter = cur @ train.T
        denom = cur.sum(axis=1, keepdims=True) + train_sum.reshape(1, -1) - inter
        sim = np.divide(inter, np.maximum(denom, 1e-8))
        out[start : start + len(cur)] = sim.max(axis=1)
    return out


def _strict_conformal_classification(
    dataset: str,
    seed: int,
    task_names: list[str],
    labels: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    probs: dict[str, np.ndarray],
    train_smiles: list[str],
    test_smiles: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    rows = []
    conditional_rows = []
    fp = MorganFingerprintExtractor(n_bits=args.n_bits, radius=args.radius)
    similarity = _nearest_tanimoto(fp.transform(train_smiles), fp.transform(test_smiles))
    ad_bins_all = _similarity_bins(similarity)
    pooled_by_cov = {coverage: [] for coverage in args.conformal_coverages}

    for task_idx, task in enumerate(task_names):
        cal_valid = masks["calibration"][:, task_idx] > 0.5
        test_valid = masks["test"][:, task_idx] > 0.5
        y_cal = labels["calibration"][cal_valid, task_idx].astype(int)
        p_cal = probs["calibration"][cal_valid, task_idx].astype(float)
        y_test = labels["test"][test_valid, task_idx].astype(int)
        p_test = probs["test"][test_valid, task_idx].astype(float)
        ad_bins = ad_bins_all[test_valid]
        if len(y_test) == 0:
            continue
        scores = np.where(y_cal == 1, 1.0 - p_cal, p_cal)
        for coverage in args.conformal_coverages:
            qhat = _finite_quantile(scores, coverage)
            contains_zero = p_test <= qhat
            contains_one = (1.0 - p_test) <= qhat
            set_size = contains_zero.astype(int) + contains_one.astype(int)
            covered = np.where(y_test == 1, contains_one, contains_zero)
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "task": task,
                    "target_coverage": coverage,
                    "protocol": "train_valid_calibration_test_7_1_1_1",
                    "n_calibration": int(len(scores)),
                    "n_test": int(len(y_test)),
                    "qhat": qhat,
                    "coverage": float(np.mean(covered)),
                    "average_set_size": float(np.mean(set_size)),
                    "singleton_rate": float(np.mean(set_size == 1)),
                    "empty_set_rate": float(np.mean(set_size == 0)),
                    "both_classes_rate": float(np.mean(set_size == 2)),
                }
            )
            pooled_by_cov[coverage].append(
                pd.DataFrame(
                    {
                        "covered": covered.astype(float),
                        "set_size": set_size.astype(float),
                        "true_label": y_test,
                        "ad_bin": ad_bins,
                    }
                )
            )
            for label_value in [0, 1]:
                idx = y_test == label_value
                if idx.any():
                    conditional_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "task": task,
                            "target_coverage": coverage,
                            "condition_type": "true_label",
                            "condition": str(label_value),
                            "n": int(idx.sum()),
                            "coverage": float(np.mean(covered[idx])),
                            "average_set_size": float(np.mean(set_size[idx])),
                        }
                    )
            for ad_bin in ["low", "medium", "high"]:
                idx = ad_bins == ad_bin
                if idx.any():
                    conditional_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "task": task,
                            "target_coverage": coverage,
                            "condition_type": "ad_bin",
                            "condition": ad_bin,
                            "n": int(idx.sum()),
                            "coverage": float(np.mean(covered[idx])),
                            "average_set_size": float(np.mean(set_size[idx])),
                        }
                    )
    for coverage, frames in pooled_by_cov.items():
        if not frames:
            continue
        pooled = pd.concat(frames, ignore_index=True)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "task": "macro_pooled",
                "target_coverage": coverage,
                "protocol": "train_valid_calibration_test_7_1_1_1",
                "n_calibration": np.nan,
                "n_test": int(len(pooled)),
                "qhat": np.nan,
                "coverage": float(pooled["covered"].mean()),
                "average_set_size": float(pooled["set_size"].mean()),
                "singleton_rate": float((pooled["set_size"] == 1).mean()),
                "empty_set_rate": float((pooled["set_size"] == 0).mean()),
                "both_classes_rate": float((pooled["set_size"] == 2).mean()),
            }
        )
        for ad_bin, group in pooled.groupby("ad_bin"):
            conditional_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "task": "macro_pooled",
                    "target_coverage": coverage,
                    "condition_type": "ad_bin",
                    "condition": ad_bin,
                    "n": int(len(group)),
                    "coverage": float(group["covered"].mean()),
                    "average_set_size": float(group["set_size"].mean()),
                }
            )
    return rows, conditional_rows


def _strict_conformal_regression(
    dataset: str,
    seed: int,
    task_names: list[str],
    labels: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    preds_original: dict[str, np.ndarray],
    train_smiles: list[str],
    test_smiles: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    rows = []
    conditional_rows = []
    fp = MorganFingerprintExtractor(n_bits=args.n_bits, radius=args.radius)
    similarity = _nearest_tanimoto(fp.transform(train_smiles), fp.transform(test_smiles))
    ad_bins_all = _similarity_bins(similarity)
    for task_idx, task in enumerate(task_names):
        cal_valid = masks["calibration"][:, task_idx] > 0.5
        test_valid = masks["test"][:, task_idx] > 0.5
        y_cal = labels["calibration"][cal_valid, task_idx].astype(float)
        p_cal = preds_original["calibration"][cal_valid, task_idx].astype(float)
        y_test = labels["test"][test_valid, task_idx].astype(float)
        p_test = preds_original["test"][test_valid, task_idx].astype(float)
        ad_bins = ad_bins_all[test_valid]
        if len(y_test) == 0:
            continue
        residuals = np.abs(y_cal - p_cal)
        abs_error = np.abs(y_test - p_test)
        for coverage in args.conformal_coverages:
            qhat = _finite_quantile(residuals, coverage)
            lower = p_test - qhat
            upper = p_test + qhat
            covered = (y_test >= lower) & (y_test <= upper)
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "task": task,
                    "target_coverage": coverage,
                    "protocol": "train_valid_calibration_test_7_1_1_1",
                    "n_calibration": int(len(residuals)),
                    "n_test": int(len(y_test)),
                    "qhat": qhat,
                    "coverage": float(np.mean(covered)),
                    "average_interval_width": float(2.0 * qhat),
                    "mean_absolute_error": float(np.mean(abs_error)),
                }
            )
            for ad_bin in ["low", "medium", "high"]:
                idx = ad_bins == ad_bin
                if idx.any():
                    conditional_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "task": task,
                            "target_coverage": coverage,
                            "condition_type": "ad_bin",
                            "condition": ad_bin,
                            "n": int(idx.sum()),
                            "coverage": float(np.mean(covered[idx])),
                            "average_interval_width": float(2.0 * qhat),
                            "mean_absolute_error": float(np.mean(abs_error[idx])),
                        }
                    )
    return rows, conditional_rows


def _aggregate_simple(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group else np.nan
        for col in value_cols:
            if col not in group:
                continue
            values = pd.to_numeric(group[col], errors="coerce")
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_mean_std"] = _format_mean_std(mean, std)
        rows.append(row)
    return pd.DataFrame(rows)


def run_strict_conformal(args: argparse.Namespace, output_dir: Path, tables_dir: Path) -> None:
    from src.metrics import compute_prediction_metrics

    cls_rows = []
    reg_rows = []
    cond_rows = []
    point_rows = []
    run_rows = []
    for dataset in args.datasets:
        spec = get_dataset_spec(dataset)
        if spec is None:
            continue
        for seed in args.seeds:
            try:
                set_seed(seed)
                df, smiles_col, label_cols, source = _read_full_dataset(dataset)
                frames = split_dataframe_four_way(df, smiles_col, args.split, seed, args.four_fracs)
                base_predictions, labels, masks, target_mean, target_std, feature_meta = _train_base_predictions(
                    dataset,
                    seed,
                    frames,
                    smiles_col,
                    label_cols,
                    spec.task_type,
                    args,
                )
                ensemble = {
                    split: _combo_prediction(base_predictions, ENSEMBLE_SPECS["toxsail"], split)
                    for split in frames
                }
                if spec.task_type == "classification":
                    rows, conditional = _strict_conformal_classification(
                        dataset,
                        seed,
                        label_cols,
                        labels,
                        masks,
                        ensemble,
                        frames["train"][smiles_col].astype(str).tolist(),
                        frames["test"][smiles_col].astype(str).tolist(),
                        args,
                    )
                    cls_rows.extend(rows)
                    cond_rows.extend(conditional)
                    outputs = _prob_to_logits(ensemble["test"])
                else:
                    preds_original = {
                        split: _inverse_normalize(values, target_mean, target_std)
                        for split, values in ensemble.items()
                    }
                    rows, conditional = _strict_conformal_regression(
                        dataset,
                        seed,
                        label_cols,
                        labels,
                        masks,
                        preds_original,
                        frames["train"][smiles_col].astype(str).tolist(),
                        frames["test"][smiles_col].astype(str).tolist(),
                        args,
                    )
                    reg_rows.extend(rows)
                    cond_rows.extend(conditional)
                    outputs = preds_original["test"]
                macro, _ = compute_prediction_metrics(outputs, labels["test"], masks["test"], label_cols, spec.task_type)
                point_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "task_type": spec.task_type,
                        "protocol": "train_valid_calibration_test_7_1_1_1",
                        "point_predictor_note": "independently retrained for strict conformal protocol",
                        "data_source": source,
                        "n_train": len(frames["train"]),
                        "n_valid": len(frames["valid"]),
                        "n_calibration": len(frames["calibration"]),
                        "n_test": len(frames["test"]),
                        **feature_meta,
                        **macro,
                    }
                )
                run_rows.append({"dataset": dataset, "seed": seed, "status": "ok"})
                print(f"OK strict_conformal {dataset} seed={seed}", flush=True)
            except Exception as exc:
                run_rows.append({"dataset": dataset, "seed": seed, "status": "failed", "error": repr(exc)})
                print(f"FAILED strict_conformal {dataset} seed={seed}: {exc!r}", flush=True)
                if not args.continue_on_error:
                    raise
            pd.DataFrame(run_rows).to_csv(output_dir / "strict_conformal_runs.csv", index=False)

    outputs = {
        "strict_conformal_classification.csv": pd.DataFrame(cls_rows),
        "strict_conformal_regression.csv": pd.DataFrame(reg_rows),
        "strict_conformal_conditional_coverage.csv": pd.DataFrame(cond_rows),
        "strict_conformal_point_predictor_metrics.csv": pd.DataFrame(point_rows),
    }
    if not outputs["strict_conformal_classification.csv"].empty:
        outputs["strict_conformal_classification_summary.csv"] = _aggregate_simple(
            outputs["strict_conformal_classification.csv"],
            ["dataset", "task", "target_coverage"],
            ["coverage", "average_set_size", "singleton_rate", "empty_set_rate", "both_classes_rate"],
        )
    if not outputs["strict_conformal_regression.csv"].empty:
        outputs["strict_conformal_regression_summary.csv"] = _aggregate_simple(
            outputs["strict_conformal_regression.csv"],
            ["dataset", "task", "target_coverage"],
            ["coverage", "average_interval_width", "mean_absolute_error"],
        )
    if not outputs["strict_conformal_conditional_coverage.csv"].empty:
        outputs["strict_conformal_conditional_coverage_summary.csv"] = _aggregate_simple(
            outputs["strict_conformal_conditional_coverage.csv"],
            ["dataset", "task", "target_coverage", "condition_type", "condition"],
            ["coverage", "average_set_size", "average_interval_width", "mean_absolute_error"],
        )
    for name, df in outputs.items():
        df.to_csv(output_dir / name, index=False)
        df.to_csv(tables_dir / name, index=False)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def _fit_recalibrators(y_cal: np.ndarray, p_cal: np.ndarray) -> dict[str, object]:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    calibrators: dict[str, object] = {"raw": None}
    if len(np.unique(y_cal)) < 2:
        return calibrators
    x_logit = _logit(p_cal).reshape(-1, 1)
    platt = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt.fit(x_logit, y_cal)
    calibrators["platt"] = platt
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, y_cal)
    calibrators["isotonic"] = iso
    x_beta = np.column_stack([np.log(np.clip(p_cal, 1e-6, 1.0)), np.log(np.clip(1.0 - p_cal, 1e-6, 1.0))])
    beta = LogisticRegression(solver="lbfgs", max_iter=1000)
    beta.fit(x_beta, y_cal)
    calibrators["beta"] = beta
    return calibrators


def _apply_recalibrator(method: str, calibrator: object, p_test: np.ndarray) -> np.ndarray:
    if method == "raw" or calibrator is None:
        return p_test
    if method == "platt":
        return calibrator.predict_proba(_logit(p_test).reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return calibrator.predict(p_test)
    if method == "beta":
        x_beta = np.column_stack([np.log(np.clip(p_test, 1e-6, 1.0)), np.log(np.clip(1.0 - p_test, 1e-6, 1.0))])
        return calibrator.predict_proba(x_beta)[:, 1]
    raise ValueError(method)


def run_recalibration(args: argparse.Namespace, output_dir: Path, tables_dir: Path) -> None:
    rows = []
    run_rows = []
    for dataset in args.classification_datasets:
        for seed in args.seeds:
            try:
                spec = get_dataset_spec(dataset)
                if spec is None or spec.task_type != "classification":
                    continue
                df, smiles_col, label_cols, source = _read_full_dataset(dataset)
                frames = split_dataframe_four_way(df, smiles_col, args.split, seed, args.four_fracs)
                base_predictions, labels, masks, _, _, _ = _train_base_predictions(
                    dataset,
                    seed,
                    frames,
                    smiles_col,
                    label_cols,
                    "classification",
                    args,
                )
                ensemble = {
                    split: _combo_prediction(base_predictions, ENSEMBLE_SPECS["toxsail"], split)
                    for split in frames
                }
                macro_frames = []
                for task_idx, task in enumerate(label_cols):
                    cal_valid = masks["calibration"][:, task_idx] > 0.5
                    test_valid = masks["test"][:, task_idx] > 0.5
                    y_cal = labels["calibration"][cal_valid, task_idx].astype(int)
                    p_cal = ensemble["calibration"][cal_valid, task_idx].astype(float)
                    y_test = labels["test"][test_valid, task_idx].astype(int)
                    p_test = ensemble["test"][test_valid, task_idx].astype(float)
                    if len(y_test) == 0:
                        continue
                    calibrators = _fit_recalibrators(y_cal, p_cal)
                    for method, calibrator in calibrators.items():
                        calibrated = np.clip(_apply_recalibrator(method, calibrator, p_test), 1e-6, 1 - 1e-6)
                        metrics = _binary_metrics(y_test, calibrated, args.n_bins)
                        row = {
                            "dataset": dataset,
                            "seed": seed,
                            "task": task,
                            "method": method,
                            "n_calibration": int(len(y_cal)),
                            "n_test": int(len(y_test)),
                            "calibration_has_two_classes": bool(len(np.unique(y_cal)) >= 2),
                            **metrics,
                        }
                        rows.append(row)
                        macro_frames.append(pd.DataFrame([{**row, "y_true": np.nan, "y_prob": np.nan}]))
                # Macro rows are metric averages across tasks, preserving missing-label task masking.
                current = pd.DataFrame([row for row in rows if row["dataset"] == dataset and row["seed"] == seed and row["task"] != "macro_average"])
                for method, group in current.groupby("method"):
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "task": "macro_average",
                            "method": method,
                            "n_calibration": float(group["n_calibration"].mean()),
                            "n_test": float(group["n_test"].mean()),
                            "calibration_has_two_classes": bool(group["calibration_has_two_classes"].all()),
                            "roc_auc": float(group["roc_auc"].mean()),
                            "pr_auc": float(group["pr_auc"].mean()),
                            "brier": float(group["brier"].mean()),
                            "ece": float(group["ece"].mean()),
                            "nll": float(group["nll"].mean()),
                        }
                    )
                run_rows.append({"dataset": dataset, "seed": seed, "status": "ok", "data_source": source})
                print(f"OK recalibration {dataset} seed={seed}", flush=True)
            except Exception as exc:
                run_rows.append({"dataset": dataset, "seed": seed, "status": "failed", "error": repr(exc)})
                print(f"FAILED recalibration {dataset} seed={seed}: {exc!r}", flush=True)
                if not args.continue_on_error:
                    raise
            pd.DataFrame(run_rows).to_csv(output_dir / "recalibration_runs.csv", index=False)
    df = pd.DataFrame(rows)
    summary = _aggregate_simple(df, ["dataset", "task", "method"], ["roc_auc", "pr_auc", "brier", "ece", "nll"]) if not df.empty else pd.DataFrame()
    for name, table in [("recalibration_comparison.csv", df), ("recalibration_comparison_summary.csv", summary)]:
        table.to_csv(output_dir / name, index=False)
        table.to_csv(tables_dir / name, index=False)


def _fragment_set(extractor: BRICSFragmentAlertExtractor, smiles_list: list[str]) -> Counter:
    counter = Counter()
    for smiles in smiles_list:
        counter.update(extractor.fragments(smiles))
    return counter


def run_gps_audit(args: argparse.Namespace, output_dir: Path, tables_dir: Path) -> None:
    audit_rows = []
    vocab_dir = tables_dir / "gps_vocabularies"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        for seed in args.seeds:
            try:
                bundle = load_bundle(dataset, args.split, seed, args.frac_train, args.frac_valid)
                train_smiles = bundle.train_df[bundle.smiles_col].astype(str).tolist()
                valid_smiles = bundle.valid_df[bundle.smiles_col].astype(str).tolist()
                test_smiles = bundle.test_df[bundle.smiles_col].astype(str).tolist()
                extractor = BRICSFragmentAlertExtractor(
                    max_fragments=args.fragment_vocab_size,
                    min_fragment_count=args.min_fragment_count,
                    alert_set=args.alert_set,
                ).fit(train_smiles)
                train_counts = _fragment_set(extractor, train_smiles)
                valid_counts = _fragment_set(extractor, valid_smiles)
                test_counts = _fragment_set(extractor, test_smiles)
                vocab = [frag for frag in extractor.vocabulary_ if not frag.startswith("__unused_fragment_")]
                vocab_set = set(vocab)
                train_set = set(train_counts)
                test_set = set(test_counts)
                valid_set = set(valid_counts)
                leakage = sorted(vocab_set - train_set)
                test_only = sorted(test_set - train_set)
                valid_only = sorted(valid_set - train_set)
                vocab_table = pd.DataFrame(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "rank": np.arange(1, len(extractor.vocabulary_) + 1),
                        "fragment": extractor.vocabulary_,
                        "train_count": [int(train_counts.get(fragment, 0)) for fragment in extractor.vocabulary_],
                        "in_vocabulary": [not fragment.startswith("__unused_fragment_") for fragment in extractor.vocabulary_],
                    }
                )
                vocab_table.to_csv(vocab_dir / f"gps_vocab_{dataset_key(dataset)}_seed{seed}.csv", index=False)
                audit_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "split": args.split,
                        "n_train": len(train_smiles),
                        "n_valid": len(valid_smiles),
                        "n_test": len(test_smiles),
                        "fragment_vocab_size_requested": args.fragment_vocab_size,
                        "min_fragment_count": args.min_fragment_count,
                        "actual_vocab_size": len(vocab),
                        "gps_feature_dim": extractor.dim,
                        "n_alerts": extractor.alert_extractor.dim,
                        "fragment_stat_dim": len(FRAGMENT_STAT_NAMES),
                        "train_unique_fragments": len(train_set),
                        "valid_unique_fragments": len(valid_set),
                        "test_unique_fragments": len(test_set),
                        "valid_only_unique_fragments": len(valid_only),
                        "test_only_unique_fragments": len(test_only),
                        "vocab_fragments_not_seen_in_train": len(leakage),
                        "leakage_check_passed": len(leakage) == 0,
                        "valid_only_fragment_examples": ";".join(valid_only[:10]),
                        "test_only_fragment_examples": ";".join(test_only[:10]),
                    }
                )
                print(f"OK gps_audit {dataset} seed={seed}", flush=True)
            except Exception as exc:
                audit_rows.append({"dataset": dataset, "seed": seed, "leakage_check_passed": False, "error": repr(exc)})
                print(f"FAILED gps_audit {dataset} seed={seed}: {exc!r}", flush=True)
                if not args.continue_on_error:
                    raise
    audit = pd.DataFrame(audit_rows)
    alert_extractor = ToxicAlertExtractor(alert_set=args.alert_set)
    alert_catalog = pd.DataFrame(
        [
            {
                "alert_index": idx,
                "alert_name": name,
                "smarts": SMARTS_PATTERNS.get(name, ""),
                "source": "curated SMARTS toxic-alert set implemented in src/features/toxic_alerts.py",
                "toxicological_meaning": ALERT_MEANINGS.get(name, ""),
            }
            for idx, name in enumerate(alert_extractor.alert_names)
        ]
    )
    unit = pd.DataFrame(
        [
            {
                "test_name": "gps_vocabulary_is_fit_on_training_split_only",
                "n_dataset_seed_checks": len(audit),
                "passed": bool(audit.get("leakage_check_passed", pd.Series(dtype=bool)).fillna(False).all()),
                "failed_checks": int((~audit.get("leakage_check_passed", pd.Series(dtype=bool)).fillna(False)).sum()),
            }
        ]
    )
    for name, table in [
        ("gps_leakage_dimension_audit.csv", audit),
        ("toxic_alert_supplementary_catalog.csv", alert_catalog),
        ("gps_leakage_unit_test.csv", unit),
    ]:
        table.to_csv(output_dir / name, index=False)
        table.to_csv(tables_dir / name, index=False)


def _greedy_nonredundant_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    used = set()
    sorted_df = df.sort_values(["similarity", "prediction_delta_with_minus_without"], ascending=[False, False])
    for _, row in sorted_df.iterrows():
        with_id = ("with", row["with_split"], int(row["with_row_id"]))
        without_id = ("without", row["without_split"], int(row["without_row_id"]))
        # A molecule cannot appear in any selected pair, regardless of whether it is the alert-positive or alert-negative member.
        mol_a = (row["with_split"], int(row["with_row_id"]))
        mol_b = (row["without_split"], int(row["without_row_id"]))
        if with_id in used or without_id in used or mol_a in used or mol_b in used:
            continue
        rows.append(row)
        used.update([with_id, without_id, mol_a, mol_b])
    return pd.DataFrame(rows)


def _ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    boot = []
    rng = np.random.default_rng(13)
    for _ in range(2000):
        boot.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def run_mmp_summary(args: argparse.Namespace, output_dir: Path, tables_dir: Path) -> None:
    from scipy.stats import ttest_rel, wilcoxon

    src = Path("tables/matched_toxicophore_pairs.csv")
    if not src.exists():
        print("SKIP mmp: tables/matched_toxicophore_pairs.csv not found", flush=True)
        return
    pairs = pd.read_csv(src)
    selected_frames = []
    summary_rows = []
    for dataset, group in pairs.groupby("dataset"):
        layers = {
            "strict_tanimoto_ge_0.6_nonredundant": group[group["similarity"] >= args.mmp_primary_threshold],
            "exploratory_tanimoto_0.2_to_0.6_nonredundant": group[
                (group["similarity"] >= args.mmp_fallback_threshold) & (group["similarity"] < args.mmp_primary_threshold)
            ],
        }
        for layer, candidates in layers.items():
            selected = _greedy_nonredundant_pairs(candidates)
            if selected.empty:
                values = np.asarray([], dtype=float)
            else:
                selected = selected.copy()
                selected["mmp_layer"] = layer
                selected_frames.append(selected)
                values = selected["prediction_delta_with_minus_without"].to_numpy(dtype=float)
            ci_low, ci_high = _ci95(values)
            try:
                t_p = float(ttest_rel(selected["prediction_with"], selected["prediction_without"]).pvalue) if len(values) >= 2 else np.nan
            except Exception:
                t_p = np.nan
            try:
                w_p = float(wilcoxon(values).pvalue) if len(values) >= 2 and np.any(np.abs(values) > 1e-12) else np.nan
            except Exception:
                w_p = np.nan
            summary_rows.append(
                {
                    "dataset": dataset,
                    "mmp_layer": layer,
                    "n_pairs": int(len(values)),
                    "mean_prediction_delta": float(np.mean(values)) if len(values) else np.nan,
                    "median_prediction_delta": float(np.median(values)) if len(values) else np.nan,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "paired_ttest_p": t_p,
                    "wilcoxon_p": w_p,
                    "small_n_caution": bool(len(values) < 10),
                }
            )
    selected_pairs = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    for name, table in [
        ("mmp_strict_exploratory_pairs_nonredundant.csv", selected_pairs),
        ("mmp_strict_exploratory_summary.csv", summary),
    ]:
        table.to_csv(output_dir / name, index=False)
        table.to_csv(tables_dir / name, index=False)


def run_figure6_audit(output_dir: Path, tables_dir: Path) -> None:
    paths = {
        "selective": Path("tables/selective_prediction.csv"),
        "ad": Path("tables/applicability_domain.csv"),
        "figure6f": Path("tables/figure6_real_ad_molecule_examples.csv"),
    }
    if paths["selective"].exists():
        selective = pd.read_csv(paths["selective"])
        aurc_rows = []
        for keys, group in selective.groupby(["dataset", "seed", "task"], dropna=False):
            group = group.sort_values("coverage_fraction")
            coverage = group["coverage_fraction"].to_numpy(dtype=float)
            risk = group["error_rate"].to_numpy(dtype=float)
            valid = np.isfinite(coverage) & np.isfinite(risk)
            if valid.sum() >= 2:
                aurc = float(np.trapz(risk[valid], coverage[valid]) / (coverage[valid].max() - coverage[valid].min()))
            else:
                aurc = np.nan
            aurc_rows.append(
                {
                    "dataset": keys[0],
                    "seed": keys[1],
                    "task": keys[2],
                    "coverage_min": float(np.nanmin(coverage)) if len(coverage) else np.nan,
                    "coverage_max": float(np.nanmax(coverage)) if len(coverage) else np.nan,
                    "aurc_error_rate": aurc,
                }
            )
        aurc_df = pd.DataFrame(aurc_rows)
        aurc_summary = _aggregate_simple(aurc_df, ["dataset", "task"], ["aurc_error_rate"])
        for name, table in [("figure6_selective_aurc.csv", aurc_df), ("figure6_selective_aurc_summary.csv", aurc_summary)]:
            table.to_csv(output_dir / name, index=False)
            table.to_csv(tables_dir / name, index=False)
        tox21 = selective[selective["dataset"] == "Tox21"].copy()
        if not tox21.empty:
            tox21.to_csv(tables_dir / "figure6_tox21_assay_level_selective_prediction.csv", index=False)
            tox21[tox21["task"] == "macro_pooled"].to_csv(
                tables_dir / "figure6_tox21_dataset_level_selective_prediction.csv", index=False
            )
    if paths["ad"].exists():
        ad = pd.read_csv(paths["ad"])
        ad.to_csv(output_dir / "figure6_ad_bin_per_seed_task.csv", index=False)
        ad.to_csv(tables_dir / "figure6_ad_bin_per_seed_task.csv", index=False)
        tox21 = ad[ad["dataset"] == "Tox21"].copy()
        if not tox21.empty:
            tox21.to_csv(tables_dir / "figure6_tox21_assay_level_applicability_domain.csv", index=False)
            tox21[tox21["task"] == "macro_pooled"].to_csv(
                tables_dir / "figure6_tox21_dataset_level_applicability_domain.csv", index=False
            )
    if paths["figure6f"].exists():
        f6 = pd.read_csv(paths["figure6f"])
        expected = {"high": 0.620, "medium": 0.392, "low": 0.179}
        rows = []
        for _, row in f6.iterrows():
            expected_value = expected.get(str(row["ad_bin"]).strip().lower())
            observed = float(row["nearest_neighbor_tanimoto"])
            rows.append(
                {
                    "ad_bin": row["ad_bin"],
                    "dataset": row["dataset"],
                    "smiles": row["smiles"],
                    "nearest_neighbor_tanimoto": observed,
                    "expected_reported_value": expected_value,
                    "absolute_difference": abs(observed - expected_value) if expected_value is not None else np.nan,
                    "matches_rounded_value": bool(expected_value is not None and round(observed, 3) == round(expected_value, 3)),
                }
            )
        bind = pd.DataFrame(rows)
        bind.to_csv(output_dir / "figure6f_data_binding_audit.csv", index=False)
        bind.to_csv(tables_dir / "figure6f_data_binding_audit.csv", index=False)


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.results_dir) / f"{stamp}_submission_supplement"
    tables_dir = Path(args.tables_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", vars(args))
    save_json(tables_dir / "submission_supplement_config.json", vars(args))

    if _enabled(args, "dependencies"):
        dependency_audit(args, output_dir, tables_dir)
    if _enabled(args, "gps_audit"):
        run_gps_audit(args, output_dir, tables_dir)
    if _enabled(args, "ensemble_ablation"):
        run_ensemble_ablation(args, output_dir, tables_dir)
    if _enabled(args, "strict_conformal"):
        run_strict_conformal(args, output_dir, tables_dir)
    if _enabled(args, "recalibration"):
        run_recalibration(args, output_dir, tables_dir)
    if _enabled(args, "mmp"):
        run_mmp_summary(args, output_dir, tables_dir)
    if _enabled(args, "figure6_audit"):
        run_figure6_audit(output_dir, tables_dir)

    produced = sorted(path.relative_to(tables_dir).as_posix() for path in tables_dir.rglob("*.csv"))
    with (output_dir / "produced_tables.json").open("w", encoding="utf-8") as f:
        json.dump(produced, f, indent=2, ensure_ascii=False)
    print(f"Saved submission supplementary experiments to {output_dir}")
    print(f"Saved submission supplementary tables to {tables_dir}")


if __name__ == "__main__":
    main()
