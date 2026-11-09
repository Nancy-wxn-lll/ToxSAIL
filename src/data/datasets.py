from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data.admet_benchmark import get_dataset_spec, normalize_dataset_key
from src.data.splitting import split_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATA_FILES = {
    "herg": "herg.tab",
    "ames": "ames.tab",
    "dili": "dili.tab",
    "ld50": "ld50_zhu.tab",
    "tox21": "tox21.tab",
    "clintox": "clintox.tab",
    "skin-reaction": "skin_reaction.tab",
    "carcinogens": "carcinogens_lagunin.tab",
}

SMILES_COLUMN_CANDIDATES = ["smiles", "SMILES", "Drug", "X", "drug", "molecule", "Mol", "mol"]
ID_COLUMN_CANDIDATES = {"Drug_ID", "drug_id", "id", "ID", "name", "Name"}


@dataclass
class DataBundle:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame
    smiles_col: str
    label_cols: List[str]
    dataset_name: str
    source: str
    task_type: str = "classification"
    primary_metric: str = "roc_auc"
    metric_direction: str = "max"
    category: str | None = None
    paper_mean: float | None = None
    paper_std: float | None = None
    tdc_name: str | None = None
    warning: str | None = None
    target_mean: list[float] | None = None
    target_std: list[float] | None = None
    target_normalized: bool = False


def _disable_rdkit_warnings() -> None:
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.warning")
        RDLogger.DisableLog("rdApp.error")
    except Exception:
        pass


def canonicalize_smiles(smiles: str) -> str:
    try:
        _disable_rdkit_warnings()
        from rdkit import Chem

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return str(smiles)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return str(smiles)


def _find_smiles_col(df: pd.DataFrame) -> str:
    for col in SMILES_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not find a SMILES column. Expected one of "
        + ", ".join(SMILES_COLUMN_CANDIDATES)
    )


def _find_label_cols(df: pd.DataFrame, smiles_col: str) -> List[str]:
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
        raise ValueError("Could not infer numeric label columns.")
    return label_cols


def _normalize_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, str, List[str]]:
    df = df.copy()
    smiles_col = _find_smiles_col(df)
    df[smiles_col] = df[smiles_col].map(canonicalize_smiles)
    label_cols = _find_label_cols(df, smiles_col)
    return df[[smiles_col] + label_cols].dropna(subset=[smiles_col]), smiles_col, label_cols


def _demo_rows() -> List[Dict[str, float | str]]:
    rows = [
        ("CCO", 0, 0, 0),
        ("CC(=O)O", 0, 0, 0),
        ("c1ccccc1", 0, 0, 0),
        ("c1ccc([N+](=O)[O-])cc1", 1, 0, 0),
        ("Nc1ccccc1", 1, 0, 0),
        ("O=CC1=CC=CC=C1", 0, 1, 0),
        ("C=CC(=O)O", 0, 1, 0),
        ("C1OC1", 0, 1, 0),
        ("ClCCCl", 0, 0, 1),
        ("O=S(=O)(Cl)c1ccccc1", 0, 1, 1),
        ("c1ccc(N=Nc2ccccc2)cc1", 1, 0, 1),
        ("NNc1ccccc1", 1, 1, 0),
        ("CCNC(=O)c1ccccc1", 0, 0, 1),
        ("Clc1ccc(Cl)cc1", 0, 0, 1),
        ("CCN(CC)CC", 0, 0, 0),
        ("CCOC(=O)c1ccccc1", 0, 0, 0),
        ("O=[N+]([O-])c1ccc(N)cc1", 1, 0, 0),
        ("C=CC(=O)c1ccccc1", 1, 1, 0),
        ("BrCC(=O)O", 0, 1, 1),
        ("FC(F)(F)c1ccc(Cl)cc1", 0, 0, 1),
        ("CC(C)O", 0, 0, 0),
        ("CC(C)=O", 0, 0, 0),
        ("NCCO", 0, 0, 0),
        ("O=CN", 0, 1, 0),
        ("CCN=NCC", 1, 0, 0),
        ("CCNNCC", 1, 1, 0),
        ("C=CC#N", 0, 1, 0),
        ("Ic1ccc(Br)cc1", 0, 0, 1),
        ("CCCl", 0, 0, 1),
        ("CCBr", 0, 0, 1),
        ("CC(C)C", 0, 0, 0),
        ("COc1ccccc1", 0, 0, 0),
        ("CC(=O)Nc1ccccc1", 0, 0, 1),
        ("C1CO1", 0, 1, 0),
        ("CCS(=O)(=O)Cl", 0, 1, 1),
        ("c1ccc(N=Nc2ccc(Cl)cc2)cc1", 1, 0, 1),
    ]
    output = []
    for idx, (smiles, y_mut, y_clin, y_side) in enumerate(rows):
        row = {
            "smiles": smiles,
            "mutagenicity": float(y_mut),
            "clinical_toxicity": float(y_clin),
            "side_effect": float(y_side),
        }
        if idx in {6, 19, 31}:
            row["clinical_toxicity"] = np.nan
        if idx in {10, 24, 30}:
            row["side_effect"] = np.nan
        output.append(row)
    return output


def load_demo_dataset() -> tuple[pd.DataFrame, str, List[str]]:
    df = pd.DataFrame(_demo_rows())
    return _normalize_dataframe(df)


def load_tdc_dataset(name: str) -> tuple[pd.DataFrame, str, List[str]]:
    if name.lower() == "tox21":
        _disable_rdkit_warnings()
        from tdc.single_pred import Tox
        from tdc.utils import retrieve_label_name_list

        label_names = retrieve_label_name_list("Tox21")
        merged = None
        for label_name in label_names:
            data = Tox(name="Tox21", label_name=label_name)
            task_df = data.get_data(format="df")[["Drug_ID", "Drug", "Y"]].copy()
            task_df = task_df.rename(columns={"Y": label_name})
            if merged is None:
                merged = task_df
            else:
                merged = merged.merge(task_df, on=["Drug_ID", "Drug"], how="outer")
        if merged is None:
            raise ValueError("TDC returned no Tox21 label tables.")
        return _normalize_dataframe(merged)

    _disable_rdkit_warnings()
    spec = get_dataset_spec(name)
    if spec is not None and spec.tdc_group.lower() == "adme":
        from tdc.single_pred import ADME

        data = ADME(name=spec.tdc_name)
    else:
        from tdc.single_pred import Tox

        data = Tox(name=spec.tdc_name if spec is not None else name)
    df = data.get_data(format="df")
    return _normalize_dataframe(df)


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".tab", ".tsv"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path, sep=None, engine="python")


def _local_data_path(dataset: str, data_dir: str | Path | None = None) -> Path | None:
    key = normalize_dataset_key(dataset)
    filename = LOCAL_DATA_FILES.get(key)
    if filename is None:
        return None
    root = Path(data_dir) if data_dir is not None else PROJECT_ROOT / "data"
    path = root / filename
    return path if path.exists() else None


def load_csv_dataset(path: str) -> tuple[pd.DataFrame, str, List[str]]:
    df = _read_table(path)
    return _normalize_dataframe(df)


def load_toxicity_data(
    dataset: str,
    split: str,
    seed: int,
    data_path: str | None = None,
    data_dir: str | Path | None = None,
    allow_demo_fallback: bool = True,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
) -> DataBundle:
    warning = None
    source = "tdc"
    dataset_name = dataset
    spec = get_dataset_spec(dataset)

    if data_path:
        df, smiles_col, label_cols = load_csv_dataset(data_path)
        source = f"csv:{data_path}"
    elif dataset.lower() == "demo":
        df, smiles_col, label_cols = load_demo_dataset()
        source = "demo"
        dataset_name = "demo"
    else:
        local_path = _local_data_path(dataset, data_dir=data_dir)
        if local_path is not None:
            df, smiles_col, label_cols = load_csv_dataset(str(local_path))
            source = f"local:{local_path}"
        else:
            try:
                df, smiles_col, label_cols = load_tdc_dataset(dataset)
            except Exception as exc:
                if not allow_demo_fallback:
                    raise
                warning = (
                    f"Could not load {dataset} from local cache or TDC "
                    f"({exc.__class__.__name__}: {exc}); using built-in demo dataset instead."
                )
                df, smiles_col, label_cols = load_demo_dataset()
                source = "demo_fallback"
                dataset_name = f"{dataset}_demo_fallback"

    train_df, valid_df, test_df = split_dataframe(
        df,
        smiles_col,
        split=split,
        seed=seed,
        frac_train=frac_train,
        frac_valid=frac_valid,
    )
    return DataBundle(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        smiles_col=smiles_col,
        label_cols=label_cols,
        dataset_name=dataset_name,
        source=source,
        task_type=spec.task_type if spec is not None else "classification",
        primary_metric=spec.primary_metric if spec is not None else "roc_auc",
        metric_direction=spec.direction if spec is not None else "max",
        category="Toxicity" if spec is not None else None,
        paper_mean=spec.paper_mean if spec is not None else None,
        paper_std=spec.paper_std if spec is not None else None,
        tdc_name=spec.tdc_name if spec is not None else dataset,
        warning=warning,
    )
