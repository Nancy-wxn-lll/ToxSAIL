from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DatasetSpec:
    display_name: str
    tdc_name: str
    tdc_group: str
    task_type: str
    primary_metric: str
    direction: str
    paper_mean: Optional[float]
    paper_std: Optional[float]
    has_paper_result: bool = True


TOXICITY_DATASETS: dict[str, DatasetSpec] = {
    "herg": DatasetSpec("hERG", "hERG", "Tox", "classification", "roc_auc", "max", 0.708, 0.045),
    "ames": DatasetSpec("AMES", "AMES", "Tox", "classification", "roc_auc", "max", 0.801, 0.030),
    "dili": DatasetSpec("DILI", "DILI", "Tox", "classification", "roc_auc", "max", 0.928, 0.022),
    "ld50": DatasetSpec("LD50", "LD50_Zhu", "Tox", "regression", "mae", "min", 0.678, 0.012),
    "tox21": DatasetSpec("Tox21", "Tox21", "Tox", "classification", "roc_auc", "max", None, None, False),
    "clintox": DatasetSpec("ClinTox", "ClinTox", "Tox", "classification", "roc_auc", "max", None, None, False),
    "skin-reaction": DatasetSpec("Skin-Reaction", "Skin_Reaction", "Tox", "classification", "roc_auc", "max", None, None, False),
    "carcinogens": DatasetSpec("Carcinogens", "Carcinogens_Lagunin", "Tox", "classification", "roc_auc", "max", None, None, False),
}

DATASET_ALIASES = {
    "ld50_zhu": "ld50",
    "skin_reaction": "skin-reaction",
    "carcinogens_lagunin": "carcinogens",
}


def normalize_dataset_key(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "-").replace("_", "-")
    compact = key.replace("-", "_")
    return DATASET_ALIASES.get(compact, DATASET_ALIASES.get(key, key))


def get_dataset_spec(name: str) -> DatasetSpec | None:
    return TOXICITY_DATASETS.get(normalize_dataset_key(name))
