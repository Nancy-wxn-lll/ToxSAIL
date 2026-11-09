from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    try:
        from scipy.stats import spearmanr

        value = spearmanr(y_true, y_pred).correlation
        return float(value) if np.isfinite(value) else np.nan
    except Exception:
        true_rank = pd.Series(y_true).rank(method="average").to_numpy(dtype=float)
        pred_rank = pd.Series(y_pred).rank(method="average").to_numpy(dtype=float)
        if np.std(true_rank) == 0 or np.std(pred_rank) == 0:
            return np.nan
        return float(np.corrcoef(true_rank, pred_rank)[0, 1])


def _safe_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    out = {"roc_auc": np.nan, "pr_auc": np.nan, "accuracy": np.nan, "f1": np.nan}
    if len(y_true) == 0:
        return out
    y_pred = (y_prob >= 0.5).astype(int)
    try:
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
        out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception:
        pass
    if len(np.unique(y_true)) < 2:
        return out
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        pass
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        pass
    return out


def compute_multilabel_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    task_names: List[str],
) -> Tuple[Dict[str, float], pd.DataFrame]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    rows = []
    for task_idx, task in enumerate(task_names):
        valid = masks[:, task_idx] > 0.5
        y_true = labels[valid, task_idx].astype(int)
        y_prob = probs[valid, task_idx]
        metrics = _safe_binary_metrics(y_true, y_prob)
        rows.append(
            {
                "task": task,
                "n": int(valid.sum()),
                "positive": int(y_true.sum()) if len(y_true) else 0,
                **metrics,
            }
        )
    per_task = pd.DataFrame(rows)
    macro = {}
    for metric in ["roc_auc", "pr_auc", "accuracy", "f1"]:
        macro[f"macro_{metric}"] = float(np.nanmean(per_task[metric].to_numpy(dtype=float)))
    return macro, per_task


def compute_regression_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    task_names: List[str],
) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows = []
    for task_idx, task in enumerate(task_names):
        valid = masks[:, task_idx] > 0.5
        y_true = labels[valid, task_idx].astype(float)
        y_pred = preds[valid, task_idx].astype(float)
        if len(y_true):
            error = y_pred - y_true
            mae = float(np.mean(np.abs(error)))
            rmse = float(np.sqrt(np.mean(error**2)))
            spearman = _safe_spearman(y_true, y_pred)
        else:
            mae = rmse = spearman = np.nan
        rows.append(
            {
                "task": task,
                "n": int(valid.sum()),
                "mae": mae,
                "rmse": rmse,
                "spearman": spearman,
            }
        )
    per_task = pd.DataFrame(rows)
    macro = {}
    for metric in ["mae", "rmse", "spearman"]:
        macro[f"macro_{metric}"] = float(np.nanmean(per_task[metric].to_numpy(dtype=float)))
    return macro, per_task


def compute_prediction_metrics(
    outputs: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    task_names: List[str],
    task_type: str,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    if task_type == "regression":
        return compute_regression_metrics(outputs, labels, masks, task_names)
    return compute_multilabel_metrics(outputs, labels, masks, task_names)
