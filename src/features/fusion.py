from __future__ import annotations

from typing import Dict

import numpy as np


FeatureBlocks = Dict[str, Dict[str, np.ndarray]]


def _alert_summary(alerts: np.ndarray) -> np.ndarray:
    count = alerts.sum(axis=1, keepdims=True).astype(np.float32)
    density = count / max(float(alerts.shape[1]), 1.0)
    has_alert = (count > 0).astype(np.float32)
    return np.concatenate([count, density, has_alert], axis=1).astype(np.float32)


def _desc_alert_interactions(desc: np.ndarray, alerts: np.ndarray) -> np.ndarray:
    return (desc[:, :, None] * alerts[:, None, :]).reshape(desc.shape[0], -1).astype(np.float32)


def fuse_feature_blocks(
    blocks_by_split: FeatureBlocks,
    fusion: str,
    feature_set: str,
) -> tuple[Dict[str, np.ndarray], dict]:
    if fusion not in {"concat", "alert_gated", "alert_interactions", "toxic_context"}:
        raise ValueError(f"Unsupported fusion strategy: {fusion}")

    features: Dict[str, np.ndarray] = {}
    dims: dict[str, int | str] = {"fusion": fusion}
    for split, blocks in blocks_by_split.items():
        fp = blocks.get("ecfp")
        desc = blocks.get("descriptors")
        alerts = blocks.get("alerts")
        gps = blocks.get("gps")
        parts = []
        if fp is not None:
            parts.append(fp)
        if desc is not None:
            parts.append(desc)
        if alerts is not None:
            parts.append(alerts)
        if gps is not None:
            parts.append(gps)
        if not parts:
            raise ValueError("No feature blocks were provided for fusion.")

        if fusion != "concat":
            if alerts is None:
                raise ValueError(f"Fusion strategy {fusion} requires toxic-alert features.")
            summary = _alert_summary(alerts)
            alert_count = summary[:, :1]
            alert_density = summary[:, 1:2]
            has_alert = summary[:, 2:3]
            parts.append(summary)

            if fusion in {"alert_gated", "toxic_context"}:
                if fp is not None:
                    parts.append((fp * has_alert).astype(np.float32))
                    parts.append((fp * alert_density).astype(np.float32))
                if desc is not None:
                    parts.append((desc * has_alert).astype(np.float32))
                    parts.append((desc * alert_count).astype(np.float32))

            if fusion in {"alert_interactions", "toxic_context"} and desc is not None:
                parts.append(_desc_alert_interactions(desc, alerts))

        features[split] = np.concatenate(parts, axis=1).astype(np.float32)

    dims["feature_dim"] = int(features["train"].shape[1])
    dims["base_feature_set"] = feature_set
    return features, dims
