"""Create compact manuscript-oriented summaries for submission supplement experiments."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def fmt(value: float, digits: int = 4) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def mean_std(row: dict[str, str], prefix: str) -> str:
    cached = row.get(f"{prefix}_mean_std", "")
    if cached:
        return cached
    mean = as_float(row.get(f"{prefix}_mean"))
    std = as_float(row.get(f"{prefix}_std"))
    if math.isnan(mean):
        return ""
    if math.isnan(std):
        return fmt(mean)
    return f"{mean:.4f} +/- {std:.4f}"


def primary_metric_name(task_type: str) -> str:
    return "macro_mae" if task_type == "regression" else "macro_roc_auc"


def primary_direction(task_type: str) -> str:
    return "min" if task_type == "regression" else "max"


def is_better(candidate: float, incumbent: float, direction: str) -> bool:
    if math.isnan(candidate):
        return False
    if math.isnan(incumbent):
        return True
    return candidate < incumbent if direction == "min" else candidate > incumbent


def improvement(candidate: float, reference: float, direction: str) -> float:
    if math.isnan(candidate) or math.isnan(reference):
        return math.nan
    return reference - candidate if direction == "min" else candidate - reference


def summarize_ensemble(tables_dir: Path) -> Path:
    rows = read_csv(tables_dir / "ensemble_model_ablation_summary.csv")
    paired = read_csv(tables_dir / "ensemble_model_ablation_paired_tests.csv")
    paired_map = {
        (
            r.get("dataset", ""),
            r.get("comparator_model", ""),
            r.get("metric", ""),
        ): r
        for r in paired
        if r.get("reference_model") == "toxsail"
    }

    by_dataset: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)

    out_rows: list[dict[str, object]] = []
    for dataset, dataset_rows in sorted(by_dataset.items()):
        task_type = dataset_rows[0].get("task_type", "")
        metric = primary_metric_name(task_type)
        direction = primary_direction(task_type)
        valid_rows = [r for r in dataset_rows if r.get("split") == "valid"]
        test_rows = [r for r in dataset_rows if r.get("split") == "test"]
        test_by_model = {r.get("model", ""): r for r in test_rows}

        selected_valid = None
        selected_score = math.nan
        for row in valid_rows:
            score = as_float(row.get(f"{metric}_mean"))
            if selected_valid is None or is_better(score, selected_score, direction):
                selected_valid = row
                selected_score = score

        best_test = None
        best_test_score = math.nan
        for row in test_rows:
            score = as_float(row.get(f"{metric}_mean"))
            if best_test is None or is_better(score, best_test_score, direction):
                best_test = row
                best_test_score = score

        toxsail = test_by_model.get("toxsail")
        selected_test = test_by_model.get(selected_valid.get("model", "")) if selected_valid else None
        toxsail_score = as_float(toxsail.get(f"{metric}_mean")) if toxsail else math.nan
        selected_test_score = as_float(selected_test.get(f"{metric}_mean")) if selected_test else math.nan
        best_test_score = as_float(best_test.get(f"{metric}_mean")) if best_test else math.nan
        selected_model = selected_valid.get("model", "") if selected_valid else ""
        paired_row = paired_map.get((dataset, selected_model, metric), {})

        out = {
            "dataset": dataset,
            "task_type": task_type,
            "selection_rule": "best validation primary metric",
            "primary_metric": metric,
            "direction": direction,
            "validation_selected_model": selected_model,
            "validation_selected_valid_primary": mean_std(selected_valid, metric) if selected_valid else "",
            "validation_selected_test_primary": mean_std(selected_test, metric) if selected_test else "",
            "toxsail_test_primary": mean_std(toxsail, metric) if toxsail else "",
            "selected_improvement_over_toxsail": fmt(improvement(selected_test_score, toxsail_score, direction)),
            "paired_ttest_p_vs_toxsail": paired_row.get("paired_ttest_p", ""),
            "wilcoxon_p_vs_toxsail": paired_row.get("wilcoxon_p", ""),
            "descriptive_best_test_model": best_test.get("model", "") if best_test else "",
            "descriptive_best_test_primary": mean_std(best_test, metric) if best_test else "",
            "descriptive_best_test_improvement_over_toxsail": fmt(
                improvement(best_test_score, toxsail_score, direction)
            ),
            "toxsail_roc_auc": mean_std(toxsail, "macro_roc_auc") if toxsail else "",
            "toxsail_pr_auc": mean_std(toxsail, "macro_pr_auc") if toxsail else "",
            "toxsail_mae": mean_std(toxsail, "macro_mae") if toxsail else "",
            "toxsail_rmse": mean_std(toxsail, "macro_rmse") if toxsail else "",
            "toxsail_spearman": mean_std(toxsail, "macro_spearman") if toxsail else "",
        }
        out_rows.append(out)

    path = tables_dir / "ensemble_best_vs_toxsail.csv"
    write_csv(path, out_rows, out_rows[0].keys() if out_rows else [])
    return path


def summarize_conformal(tables_dir: Path) -> Path:
    cls_rows = read_csv(tables_dir / "strict_conformal_classification_summary.csv")
    reg_rows = read_csv(tables_dir / "strict_conformal_regression_summary.csv")
    cond_rows = read_csv(tables_dir / "strict_conformal_conditional_coverage_summary.csv")
    cond_lookup = {
        (r.get("dataset", ""), r.get("task", ""), r.get("target_coverage", ""), r.get("condition_type", ""), r.get("condition", "")): r
        for r in cond_rows
    }
    out_rows: list[dict[str, object]] = []

    for row in cls_rows:
        if row.get("task") != "macro_pooled" or row.get("target_coverage") != "0.9":
            continue
        dataset = row["dataset"]
        base = {
            "dataset": dataset,
            "task_type": "classification",
            "target_coverage": "0.9",
            "coverage": row.get("coverage_mean_std", ""),
            "average_set_size": row.get("average_set_size_mean_std", ""),
            "singleton_rate": row.get("singleton_rate_mean_std", ""),
            "interval_width": "",
            "mae": "",
        }
        for bin_name in ("high", "medium", "low"):
            cond = cond_lookup.get((dataset, "macro_pooled", "0.9", "ad_bin", bin_name), {})
            base[f"{bin_name}_ad_coverage"] = cond.get("coverage_mean_std", "")
            base[f"{bin_name}_ad_set_size"] = cond.get("average_set_size_mean_std", "")
        out_rows.append(base)

    for row in reg_rows:
        if row.get("target_coverage") != "0.9":
            continue
        dataset = row["dataset"]
        base = {
            "dataset": dataset,
            "task_type": "regression",
            "target_coverage": "0.9",
            "coverage": row.get("coverage_mean_std", ""),
            "average_set_size": "",
            "singleton_rate": "",
            "interval_width": row.get("average_interval_width_mean_std", ""),
            "mae": row.get("mean_absolute_error_mean_std", ""),
        }
        for bin_name in ("high", "medium", "low"):
            cond = cond_lookup.get((dataset, "Y", "0.9", "ad_bin", bin_name), {})
            base[f"{bin_name}_ad_coverage"] = cond.get("coverage_mean_std", "")
            base[f"{bin_name}_ad_set_size"] = cond.get("average_set_size_mean_std", "")
        out_rows.append(base)

    fieldnames = [
        "dataset",
        "task_type",
        "target_coverage",
        "coverage",
        "average_set_size",
        "singleton_rate",
        "interval_width",
        "mae",
        "high_ad_coverage",
        "high_ad_set_size",
        "medium_ad_coverage",
        "medium_ad_set_size",
        "low_ad_coverage",
        "low_ad_set_size",
    ]
    path = tables_dir / "strict_conformal_main_summary.csv"
    write_csv(path, out_rows, fieldnames)
    return path


def summarize_recalibration(tables_dir: Path) -> Path:
    rows = [
        r
        for r in read_csv(tables_dir / "recalibration_comparison_summary.csv")
        if r.get("task") == "macro_average"
    ]
    by_dataset: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)

    out_rows: list[dict[str, object]] = []
    for dataset, dataset_rows in sorted(by_dataset.items()):
        best_ece = min(dataset_rows, key=lambda r: as_float(r.get("ece_mean")))
        best_brier = min(dataset_rows, key=lambda r: as_float(r.get("brier_mean")))
        raw = next((r for r in dataset_rows if r.get("method") == "raw"), {})
        out_rows.append(
            {
                "dataset": dataset,
                "raw_brier": raw.get("brier_mean_std", ""),
                "raw_ece": raw.get("ece_mean_std", ""),
                "best_ece_method": best_ece.get("method", ""),
                "best_ece": best_ece.get("ece_mean_std", ""),
                "best_brier_method": best_brier.get("method", ""),
                "best_brier": best_brier.get("brier_mean_std", ""),
                "raw_roc_auc": raw.get("roc_auc_mean_std", ""),
                "raw_pr_auc": raw.get("pr_auc_mean_std", ""),
            }
        )

    fieldnames = [
        "dataset",
        "raw_brier",
        "raw_ece",
        "best_ece_method",
        "best_ece",
        "best_brier_method",
        "best_brier",
        "raw_roc_auc",
        "raw_pr_auc",
    ]
    path = tables_dir / "recalibration_best_method_by_dataset.csv"
    write_csv(path, out_rows, fieldnames)
    return path


def summarize_gps_audit(tables_dir: Path) -> Path:
    rows = read_csv(tables_dir / "gps_leakage_dimension_audit.csv")
    out_rows: list[dict[str, object]] = []
    by_dataset: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    for dataset, dataset_rows in sorted(by_dataset.items()):
        dims = sorted({r.get("gps_feature_dim", "") for r in dataset_rows})
        vocabs = [as_float(r.get("actual_vocab_size")) for r in dataset_rows]
        passed = all(r.get("leakage_check_passed") == "True" for r in dataset_rows)
        out_rows.append(
            {
                "dataset": dataset,
                "n_seeds": len(dataset_rows),
                "gps_feature_dim_values": ";".join(dims),
                "actual_vocab_size_min": fmt(min(vocabs)),
                "actual_vocab_size_max": fmt(max(vocabs)),
                "test_only_fragment_count_min": fmt(min(as_float(r.get("test_only_unique_fragments")) for r in dataset_rows)),
                "test_only_fragment_count_max": fmt(max(as_float(r.get("test_only_unique_fragments")) for r in dataset_rows)),
                "leakage_check_passed_all_seeds": passed,
            }
        )
    fieldnames = [
        "dataset",
        "n_seeds",
        "gps_feature_dim_values",
        "actual_vocab_size_min",
        "actual_vocab_size_max",
        "test_only_fragment_count_min",
        "test_only_fragment_count_max",
        "leakage_check_passed_all_seeds",
    ]
    path = tables_dir / "gps_leakage_dimension_audit_summary.csv"
    write_csv(path, out_rows, fieldnames)
    return path


def write_completion_status(tables_dir: Path) -> Path:
    neural_table = Path("baselines/tables/smiles_mamba_neural_baselines.csv")
    neural_status = "partial_prior_run" if neural_table.exists() else "implemented_not_run"
    checks = [
        (
            "dependency_audit",
            "Dependency audit for PyTDC, RDKit, mamba-ssm and ML libraries",
            tables_dir / "dependency_audit.csv",
        ),
        (
            "gps_train_only_audit",
            "Train-only GPS vocabulary, dimensions, test-only fragments and leakage test",
            tables_dir / "gps_leakage_dimension_audit.csv",
        ),
        (
            "ensemble_ablation",
            "XGBoost, LightGBM, ExtraTrees, two-way ensembles and three-model late ensemble",
            tables_dir / "ensemble_model_ablation_summary.csv",
        ),
        (
            "strict_conformal",
            "Four-way scaffold split conformal calibration for classification and LD50 regression",
            tables_dir / "strict_conformal_classification_summary.csv",
        ),
        (
            "mmp_layered_analysis",
            "Strict and exploratory non-redundant toxicophore MMP risk shift analysis",
            tables_dir / "mmp_strict_exploratory_summary.csv",
        ),
        (
            "figure6_audit",
            "Selective prediction, applicability domain bins, and Figure 6F molecule binding audit",
            tables_dir / "figure6f_data_binding_audit.csv",
        ),
        (
            "recalibration",
            "Raw, Platt, isotonic and beta calibration comparison",
            tables_dir / "recalibration_comparison_summary.csv",
        ),
        (
            "neural_baseline_code",
            "Morgan MLP, SMILES CNN, GCN, NeuralFP and SMILES-Mamba code is available; full 8-dataset x 5-seed rerun was not completed in this supplement run",
            Path("baselines/neural_baselines.py"),
        ),
        (
            "neural_baseline_prior_partial_results",
            "Earlier neural baseline table covers only the available prior subset, not the complete current checklist",
            neural_table,
        ),
        (
            "neural_baseline_smoke_tests",
            "Smoke tests verified real mamba-ssm sequence training on Tox21 and normalized LD50 regression training",
            Path("baselines/results/neural_baselines_smoke"),
        ),
    ]
    rows = []
    for item, description, path in checks:
        status = "done" if path.exists() else "missing"
        if item == "neural_baseline_code":
            status = "implemented_not_full_run"
        if item == "neural_baseline_prior_partial_results":
            status = neural_status
        rows.append(
            {
                "item": item,
                "status": status,
                "description": description,
                "path": str(path),
            }
        )
    out = tables_dir / "supplementary_experiment_completion_status.csv"
    write_csv(out, rows, ["item", "status", "description", "path"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", default="tables/submission_supplement")
    args = parser.parse_args()
    tables_dir = Path(args.tables_dir)

    produced = [
        summarize_ensemble(tables_dir),
        summarize_conformal(tables_dir),
        summarize_recalibration(tables_dir),
        summarize_gps_audit(tables_dir),
        write_completion_status(tables_dir),
    ]
    print("Generated summary tables:")
    for path in produced:
        print(f"  {path}")


if __name__ == "__main__":
    main()
