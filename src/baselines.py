from __future__ import annotations

import argparse
import importlib.util
from typing import Dict

import numpy as np


BASE_MODEL_NAMES = {"xgb", "lgbm", "extra_trees", "random_forest"}


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def build_estimator(
    model_name: str,
    task_type: str,
    seed: int,
    args: argparse.Namespace,
    y_train: np.ndarray,
):
    if model_name == "xgb":
        if not _module_available("xgboost"):
            raise RuntimeError("xgboost is not installed.")
        if task_type == "classification":
            from xgboost import XGBClassifier

            pos = max(float((y_train > 0.5).sum()), 1.0)
            neg = max(float((y_train <= 0.5).sum()), 1.0)
            return XGBClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                reg_lambda=2.0,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=neg / pos,
                tree_method="hist",
                n_jobs=args.n_jobs,
                random_state=seed,
                verbosity=0,
            )
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=2.0,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=args.n_jobs,
            random_state=seed,
            verbosity=0,
        )

    if model_name == "lgbm":
        if not _module_available("lightgbm"):
            raise RuntimeError("lightgbm is not installed.")
        if task_type == "classification":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
                num_leaves=max(8, min(64, 2 ** max(args.max_depth, 1))),
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                reg_lambda=2.0,
                class_weight="balanced",
                n_jobs=args.n_jobs,
                random_state=seed,
                verbose=-1,
            )
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            num_leaves=max(8, min(64, 2 ** max(args.max_depth, 1))),
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=2.0,
            n_jobs=args.n_jobs,
            random_state=seed,
            verbose=-1,
        )

    if model_name == "extra_trees":
        if task_type == "classification":
            from sklearn.ensemble import ExtraTreesClassifier

            return ExtraTreesClassifier(
                n_estimators=args.n_estimators,
                max_features="sqrt",
                min_samples_leaf=args.min_samples_leaf,
                class_weight="balanced",
                n_jobs=args.n_jobs,
                random_state=seed,
            )
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            max_features="sqrt",
            min_samples_leaf=args.min_samples_leaf,
            n_jobs=args.n_jobs,
            random_state=seed,
        )

    if model_name == "random_forest":
        if task_type == "classification":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=args.n_estimators,
                max_features="sqrt",
                min_samples_leaf=args.min_samples_leaf,
                class_weight="balanced",
                n_jobs=args.n_jobs,
                random_state=seed,
            )
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_features="sqrt",
            min_samples_leaf=args.min_samples_leaf,
            n_jobs=args.n_jobs,
            random_state=seed,
        )

    raise ValueError(f"Unknown base model: {model_name}")


def _constant_predictions(
    task_type: str,
    y_train: np.ndarray,
    n_samples_by_split: Dict[str, int],
) -> Dict[str, np.ndarray]:
    if task_type == "classification":
        value = float(np.clip(np.mean(y_train) if len(y_train) else 0.5, 1e-4, 1.0 - 1e-4))
    else:
        value = float(np.mean(y_train) if len(y_train) else 0.0)
    return {
        split: np.full((n_samples,), value, dtype=np.float32)
        for split, n_samples in n_samples_by_split.items()
    }


def predict_estimator(estimator, x: np.ndarray, task_type: str) -> np.ndarray:
    if task_type == "classification":
        proba = estimator.predict_proba(x)
        if isinstance(proba, list):
            proba = proba[0]
        if proba.ndim == 1:
            return np.clip(proba, 1e-4, 1.0 - 1e-4).astype(np.float32)
        if proba.shape[1] == 1:
            return np.full((proba.shape[0],), float(proba[:, 0].mean()), dtype=np.float32)
        return np.clip(proba[:, 1], 1e-4, 1.0 - 1e-4).astype(np.float32)
    return estimator.predict(x).astype(np.float32)


def train_predict_baseline(
    model_name: str,
    features: Dict[str, np.ndarray],
    labels_for_fit: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    task_type: str,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    n_tasks = labels_for_fit["train"].shape[1]
    n_samples_by_split = {split: value.shape[0] for split, value in features.items()}
    preds = {
        split: np.zeros((n_samples, n_tasks), dtype=np.float32)
        for split, n_samples in n_samples_by_split.items()
    }
    for task_idx in range(n_tasks):
        valid = masks["train"][:, task_idx] > 0.5
        y_train = labels_for_fit["train"][valid, task_idx]
        if len(y_train) == 0:
            task_preds = _constant_predictions(task_type, y_train, n_samples_by_split)
        elif task_type == "classification" and len(np.unique(y_train.astype(int))) < 2:
            task_preds = _constant_predictions(task_type, y_train, n_samples_by_split)
        else:
            estimator = build_estimator(model_name, task_type, seed + task_idx, args, y_train)
            estimator.fit(features["train"][valid], y_train)
            task_preds = {
                split: predict_estimator(estimator, x, task_type)
                for split, x in features.items()
            }
        for split, values in task_preds.items():
            preds[split][:, task_idx] = values
    return preds
