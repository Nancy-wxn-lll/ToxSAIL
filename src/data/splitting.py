from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def _stable_hash(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _fallback_scaffold(smiles: str) -> str:
    text = str(smiles)
    text = re.sub(r"\[[^\]]+\]", "A", text)
    text = re.sub(r"Cl|Br|F|I", "X", text)
    text = re.sub(r"[NOSPFIBCHnoscpb]", "A", text)
    text = re.sub(r"\d+", "R", text)
    text = re.sub(r"[^A-Za-zR=#()]", "", text)
    return text[:32] or "empty"


@lru_cache(maxsize=200_000)
def scaffold_key(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return f"invalid_{_stable_hash(str(smiles))}"
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaffold or f"acyclic_{_stable_hash(str(smiles)) % 1009}"
    except Exception:
        return _fallback_scaffold(str(smiles))


def _assign_groups(
    groups: Dict[str, List[int]], frac_train: float, frac_valid: float, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    n_total = sum(len(indices) for _, indices in group_items)
    train_cutoff = int(round(frac_train * n_total))
    valid_cutoff = int(round((frac_train + frac_valid) * n_total))

    train_idx: List[int] = []
    valid_idx: List[int] = []
    test_idx: List[int] = []
    for _, indices in group_items:
        if len(train_idx) + len(indices) <= train_cutoff or not train_idx:
            train_idx.extend(indices)
        elif len(train_idx) + len(valid_idx) + len(indices) <= valid_cutoff or not valid_idx:
            valid_idx.extend(indices)
        else:
            test_idx.extend(indices)

    if not valid_idx and test_idx:
        valid_idx.append(test_idx.pop())
    if not test_idx and valid_idx:
        test_idx.append(valid_idx.pop())
    return train_idx, valid_idx, test_idx


def split_dataframe(
    df: pd.DataFrame,
    smiles_col: str,
    split: str = "scaffold",
    seed: int = 42,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split not in {"scaffold", "random", "standard"}:
        raise ValueError(f"Unsupported split: {split}")

    df = df.reset_index(drop=True)
    if len(df) < 3:
        raise ValueError("Need at least 3 molecules to create train/valid/test splits.")

    if split in {"random", "standard"}:
        indices = list(range(len(df)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        n_train = max(1, int(round(frac_train * len(indices))))
        n_valid = max(1, int(round(frac_valid * len(indices))))
        if n_train + n_valid >= len(indices):
            n_train = len(indices) - 2
            n_valid = 1
        train_idx = indices[:n_train]
        valid_idx = indices[n_train : n_train + n_valid]
        test_idx = indices[n_train + n_valid :]
    else:
        groups: Dict[str, List[int]] = defaultdict(list)
        for idx, smiles in enumerate(df[smiles_col].tolist()):
            groups[scaffold_key(smiles)].append(idx)
        train_idx, valid_idx, test_idx = _assign_groups(groups, frac_train, frac_valid, seed)

    return (
        df.iloc[sorted(train_idx)].reset_index(drop=True),
        df.iloc[sorted(valid_idx)].reset_index(drop=True),
        df.iloc[sorted(test_idx)].reset_index(drop=True),
    )
