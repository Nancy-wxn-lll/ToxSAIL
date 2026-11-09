from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.splitting import split_dataframe
from src.features.fragments import BRICSFragmentAlertExtractor


def test_gps_vocabulary_is_fit_on_train_split_only() -> None:
    df = pd.read_csv(ROOT / "data" / "ames.tab", sep="\t")
    smiles_col = "Drug"
    df = df[[smiles_col, "Y"]].dropna(subset=[smiles_col]).head(400).copy()
    df[smiles_col] = df[smiles_col].astype(str)
    train_df, _, test_df = split_dataframe(
        df,
        smiles_col,
        split="random",
        seed=42,
        frac_train=0.8,
        frac_valid=0.1,
    )
    extractor = BRICSFragmentAlertExtractor(
        max_fragments=64,
        min_fragment_count=2,
        alert_set="expanded",
    ).fit(train_df[smiles_col].astype(str).tolist())

    train_fragments = set()
    for smiles in train_df[smiles_col].astype(str):
        train_fragments.update(extractor.fragments(smiles))
    test_fragments = set()
    for smiles in test_df[smiles_col].astype(str):
        test_fragments.update(extractor.fragments(smiles))

    vocabulary = {fragment for fragment in extractor.vocabulary_ if not fragment.startswith("__unused_fragment_")}
    test_only_fragments = test_fragments - train_fragments

    assert vocabulary <= train_fragments
    assert vocabulary.isdisjoint(test_only_fragments)
