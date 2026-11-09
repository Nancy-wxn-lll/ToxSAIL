from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List

import numpy as np

from src.features.toxic_alerts import ToxicAlertExtractor


FRAGMENT_STAT_NAMES = [
    "brics_fragment_count",
    "brics_unique_fragment_count",
    "brics_vocab_hit_fraction",
    "brics_any_fragment_alert",
    "brics_fragment_alert_hits",
    "brics_fragment_alert_density",
]


def _clean_brics_dummy_labels(smiles: str) -> str:
    smiles = re.sub(r"\[(\d+)\*\]", "[*]", smiles)
    smiles = re.sub(r"\[\*:(\d+)\]", "[*]", smiles)
    return smiles


@dataclass
class BRICSFragmentAlertExtractor:
    """BRICS fragment and fragment-alert features for ToxSAIL.

    Feature layout:
    1. train-vocabulary BRICS fragment normalized counts
    2. per-alert fraction of BRICS fragments containing each alert
    3. per-alert fragment-alert contribution score
    4. per-alert molecule/fragment alert disagreement
    5. compact fragment statistics
    """

    max_fragments: int = 256
    min_fragment_count: int = 2
    alert_set: str = "expanded"
    use_rdkit: bool = True
    vocabulary_: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.alert_extractor = ToxicAlertExtractor(
            use_rdkit=self.use_rdkit,
            alert_set=self.alert_set,
        )
        self.rdkit_available = False
        if self.use_rdkit:
            try:
                from rdkit import Chem
                from rdkit.Chem import BRICS

                self.Chem = Chem
                self.BRICS = BRICS
                self.rdkit_available = True
            except Exception:
                self.Chem = None
                self.BRICS = None
                self.rdkit_available = False

    @property
    def alert_names(self) -> list[str]:
        return self.alert_extractor.alert_names

    @property
    def dim(self) -> int:
        return self.max_fragments + 3 * self.alert_extractor.dim + len(FRAGMENT_STAT_NAMES)

    @property
    def feature_names(self) -> list[str]:
        return (
            [f"brics_fragment::{fragment}" for fragment in self.vocabulary_]
            + [f"fragment_alert_fraction::{name}" for name in self.alert_names]
            + [f"fragment_alert_contribution::{name}" for name in self.alert_names]
            + [f"fragment_alert_disagreement::{name}" for name in self.alert_names]
            + FRAGMENT_STAT_NAMES
        )

    def _fallback_fragments(self, smiles: str) -> list[str]:
        smiles = "" if smiles is None else str(smiles)
        return [smiles] if smiles else []

    def fragments(self, smiles: str) -> list[str]:
        smiles = "" if smiles is None else str(smiles)
        if not self.rdkit_available:
            return self._fallback_fragments(smiles)
        mol = self.Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        try:
            broken = self.BRICS.BreakBRICSBonds(mol)
            frags = self.Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True)
            out = []
            for frag in frags:
                frag_smiles = self.Chem.MolToSmiles(frag, canonical=True, isomericSmiles=True)
                frag_smiles = _clean_brics_dummy_labels(frag_smiles)
                if frag_smiles:
                    out.append(frag_smiles)
            if not out:
                out = [self.Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)]
            return out
        except Exception:
            canonical = self.Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            return [canonical] if canonical else []

    def fit(self, smiles_list: List[str]) -> "BRICSFragmentAlertExtractor":
        counter: Counter[str] = Counter()
        for smiles in smiles_list:
            counter.update(self.fragments(smiles))
        fragments = [
            fragment
            for fragment, count in counter.most_common()
            if count >= self.min_fragment_count
        ]
        self.vocabulary_ = fragments[: self.max_fragments]
        if len(self.vocabulary_) < self.max_fragments:
            self.vocabulary_.extend(
                [f"__unused_fragment_{idx}" for idx in range(len(self.vocabulary_), self.max_fragments)]
            )
        self._vocab_index = {fragment: idx for idx, fragment in enumerate(self.vocabulary_)}
        return self

    def _fragment_alert_matrix(self, fragments: list[str]) -> np.ndarray:
        if not fragments:
            return np.zeros((0, self.alert_extractor.dim), dtype=np.float32)
        return self.alert_extractor.transform(fragments).astype(np.float32)

    def transform_one(self, smiles: str) -> np.ndarray:
        if not hasattr(self, "_vocab_index"):
            self.fit([])
        fragments = self.fragments(smiles)
        n_fragments = max(len(fragments), 1)

        fragment_counts = np.zeros(self.max_fragments, dtype=np.float32)
        vocab_hits = 0
        for fragment in fragments:
            idx = self._vocab_index.get(fragment)
            if idx is not None:
                fragment_counts[idx] += 1.0
                vocab_hits += 1
        fragment_counts = fragment_counts / float(n_fragments)

        mol_alerts = self.alert_extractor.transform_one(smiles)
        fragment_alerts = self._fragment_alert_matrix(fragments)
        if len(fragment_alerts):
            fragment_alert_count = fragment_alerts.sum(axis=0).astype(np.float32)
            fragment_alert_fraction = fragment_alert_count / float(n_fragments)
        else:
            fragment_alert_count = np.zeros(self.alert_extractor.dim, dtype=np.float32)
            fragment_alert_fraction = np.zeros(self.alert_extractor.dim, dtype=np.float32)

        fragment_alert_fraction = np.clip(fragment_alert_fraction, 0.0, 1.0).astype(np.float32)
        contribution = (mol_alerts * fragment_alert_fraction).astype(np.float32)
        disagreement = np.abs(mol_alerts - fragment_alert_fraction).astype(np.float32)
        alert_density = (
            float(fragment_alert_count.sum())
            / max(float(n_fragments * self.alert_extractor.dim), 1.0)
        )
        stats = np.asarray(
            [
                float(len(fragments)),
                float(len(set(fragments))),
                float(vocab_hits) / float(n_fragments),
                float(fragment_alert_count.sum() > 0),
                float(fragment_alert_count.sum()),
                alert_density,
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                fragment_counts,
                fragment_alert_fraction,
                contribution,
                disagreement,
                stats,
            ]
        ).astype(np.float32)

    def transform(self, smiles_list: List[str]) -> np.ndarray:
        return np.stack([self.transform_one(smiles) for smiles in smiles_list]).astype(np.float32)

    def describe_backend(self) -> str:
        if self.rdkit_available:
            return f"rdkit_brics_fragment_alerts_{self.alert_set}_vocab{self.max_fragments}"
        return f"fallback_fragment_alerts_{self.alert_set}_vocab{self.max_fragments}"
