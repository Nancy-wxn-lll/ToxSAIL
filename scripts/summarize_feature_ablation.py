from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


PRIMARY_MODEL_NAMES = ["toxsail", "toxalert_hybrid"]

FEATURE_LABELS = {
    "ecfp": "ECFP only",
    "desc": "RDKit descriptors only",
    "alerts": "Toxic alerts only",
    "gps": "BRICS GPS only",
    "fragments": "BRICS GPS only",
    "ecfp_desc": "ECFP + RDKit",
    "ecfp_alerts": "ECFP + alerts",
    "ecfp_desc_alerts": "ToxAlert-Hybrid",
    "alerts_gps": "Alerts + BRICS GPS",
    "ecfp_gps": "ECFP + BRICS GPS",
    "ecfp_alerts_gps": "ECFP + alerts + BRICS GPS",
    "desc_alerts_gps": "RDKit + alerts + BRICS GPS",
    "ecfp_desc_alerts_gps": "Full ToxSAIL",
}


def _latest_metrics(results_dir: Path, feature_set: str) -> Path | None:
    root = results_dir / f"feature_ablation_{feature_set}"
    candidates = sorted(root.glob("*_hybrid_benchmark/metrics.csv"))
    return candidates[-1] if candidates else None


def _fmt(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return ""
    return f"{mean:.4f} +/- {std:.4f}"


def _load_feature_metrics(results_dir: Path, feature_sets: list[str]) -> pd.DataFrame:
    frames = []
    for feature_set in feature_sets:
        path = _latest_metrics(results_dir, feature_set)
        if path is None:
            print(f"[warn] missing metrics for {feature_set}")
            continue
        df = pd.read_csv(path)
        df["feature_set"] = feature_set
        df["feature_label"] = FEATURE_LABELS.get(feature_set, feature_set)
        df["source_path"] = str(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No feature ablation metrics were found.")
    return pd.concat(frames, ignore_index=True)


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    test = df[(df["split"] == "test") & (df["model"].isin(PRIMARY_MODEL_NAMES))].copy()
    rows = []
    for (dataset, feature_set, feature_label), group in test.groupby(
        ["dataset", "feature_set", "feature_label"], dropna=False
    ):
        row = {
            "dataset": dataset,
            "feature_set": feature_set,
            "feature_label": feature_label,
            "n_seeds": group["seed"].nunique(),
        }
        for metric in ["macro_roc_auc", "macro_pr_auc", "macro_accuracy", "macro_f1"]:
            if metric in group:
                mean = group[metric].mean()
                std = group[metric].std(ddof=0)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
                row[f"{metric}_mean_std"] = _fmt(mean, std)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "feature_set"])


def _pivot(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    value_col = f"{metric}_mean_std"
    if value_col not in summary:
        return pd.DataFrame()
    pivot = summary.pivot(index="dataset", columns="feature_label", values=value_col)
    ordered = [label for label in FEATURE_LABELS.values() if label in pivot.columns]
    return pivot[ordered].reset_index()


def _prauc_table(summary: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    subset = summary[
        (summary["dataset"].isin(datasets)) & (summary["feature_set"] == "ecfp_desc_alerts_gps")
    ].copy()
    cols = [
        "dataset",
        "feature_label",
        "n_seeds",
        "macro_roc_auc_mean_std",
        "macro_pr_auc_mean_std",
    ]
    return subset[[col for col in cols if col in subset.columns]]


def _paired_tests(df: pd.DataFrame, baselines: list[str]) -> pd.DataFrame:
    test = df[(df["split"] == "test") & (df["model"].isin(PRIMARY_MODEL_NAMES))].copy()
    rows = []
    for dataset, dataset_df in test.groupby("dataset"):
        full = dataset_df[dataset_df["feature_set"] == "ecfp_desc_alerts_gps"].set_index("seed")
        for baseline in baselines:
            base = dataset_df[dataset_df["feature_set"] == baseline].set_index("seed")
            shared = sorted(set(full.index) & set(base.index))
            if len(shared) < 2:
                continue
            for metric in ["macro_roc_auc", "macro_pr_auc"]:
                if metric not in full or metric not in base:
                    continue
                x = full.loc[shared, metric].astype(float).to_numpy()
                y = base.loc[shared, metric].astype(float).to_numpy()
                valid = np.isfinite(x) & np.isfinite(y)
                x = x[valid]
                y = y[valid]
                if len(x) < 2:
                    continue
                diff = x - y
                try:
                    t_p = float(ttest_rel(x, y, nan_policy="omit").pvalue)
                except Exception:
                    t_p = np.nan
                try:
                    w_p = float(wilcoxon(x, y, zero_method="wilcox").pvalue)
                except Exception:
                    w_p = np.nan
                rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "comparison": f"Full model vs {FEATURE_LABELS.get(baseline, baseline)}",
                        "baseline_feature_set": baseline,
                        "n_pairs": int(len(x)),
                        "full_mean": float(np.mean(x)),
                        "baseline_mean": float(np.mean(y)),
                        "mean_delta": float(np.mean(diff)),
                        "delta_std": float(np.std(diff, ddof=0)),
                        "paired_ttest_p": t_p,
                        "wilcoxon_p": w_p,
                    }
                )
    return pd.DataFrame(rows).sort_values(["dataset", "metric", "baseline_feature_set"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tables-dir", default="tables")
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["ecfp", "desc", "alerts", "gps", "ecfp_desc", "ecfp_alerts", "alerts_gps", "ecfp_desc_alerts", "ecfp_desc_alerts_gps"],
    )
    args = parser.parse_args()

    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    df = _load_feature_metrics(Path(args.results_dir), args.feature_sets)
    summary = _summarize(df)
    summary.to_csv(tables_dir / "feature_ablation_summary.csv", index=False)
    _pivot(summary, "macro_roc_auc").to_csv(
        tables_dir / "feature_ablation_pivot_roc_auc.csv", index=False
    )
    _pivot(summary, "macro_pr_auc").to_csv(
        tables_dir / "feature_ablation_pivot_pr_auc.csv", index=False
    )
    _prauc_table(summary, ["Tox21", "ClinTox", "DILI"]).to_csv(
        tables_dir / "prauc_imbalanced_tasks.csv", index=False
    )
    _paired_tests(df, ["ecfp", "ecfp_desc", "ecfp_alerts"]).to_csv(
        tables_dir / "significance_tests_feature_ablation.csv", index=False
    )
    print(f"Saved feature ablation summaries to {tables_dir}")


if __name__ == "__main__":
    main()
