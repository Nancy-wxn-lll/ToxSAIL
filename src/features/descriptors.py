from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "HeavyAtomCount",
    "FractionCSP3",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumHeteroatoms",
    "NHOHCount",
    "NOCount",
    "LabuteASA",
    "BalabanJ",
    "BertzCT",
    "MolMR",
    "FormalCharge",
]


@dataclass
class RDKitDescriptorExtractor:
    """Small standardized RDKit descriptor vector for toxicity models."""

    use_rdkit: bool = True
    names: List[str] = field(default_factory=lambda: list(DESCRIPTOR_NAMES))

    def __post_init__(self) -> None:
        self.rdkit_available = False
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        if self.use_rdkit:
            try:
                from rdkit import Chem
                from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

                self.Chem = Chem
                self._funcs = {
                    "MolWt": Descriptors.MolWt,
                    "MolLogP": Crippen.MolLogP,
                    "TPSA": rdMolDescriptors.CalcTPSA,
                    "NumHDonors": Lipinski.NumHDonors,
                    "NumHAcceptors": Lipinski.NumHAcceptors,
                    "NumRotatableBonds": Lipinski.NumRotatableBonds,
                    "RingCount": Lipinski.RingCount,
                    "HeavyAtomCount": Descriptors.HeavyAtomCount,
                    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
                    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
                    "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
                    "NumHeteroatoms": Lipinski.NumHeteroatoms,
                    "NHOHCount": Lipinski.NHOHCount,
                    "NOCount": Lipinski.NOCount,
                    "LabuteASA": rdMolDescriptors.CalcLabuteASA,
                    "BalabanJ": Descriptors.BalabanJ,
                    "BertzCT": Descriptors.BertzCT,
                    "MolMR": Crippen.MolMR,
                    "FormalCharge": lambda mol: sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
                }
                self.rdkit_available = True
            except Exception:
                self.Chem = None
                self._funcs = {}
                self.rdkit_available = False

    @property
    def dim(self) -> int:
        return len(self.names)

    def _raw_one(self, smiles: str) -> np.ndarray:
        if not self.rdkit_available:
            return np.zeros(self.dim, dtype=np.float32)
        mol = self.Chem.MolFromSmiles("" if smiles is None else str(smiles))
        if mol is None:
            return np.zeros(self.dim, dtype=np.float32)
        values = []
        for name in self.names:
            try:
                value = float(self._funcs[name](mol))
            except Exception:
                value = 0.0
            if not np.isfinite(value):
                value = 0.0
            values.append(value)
        return np.asarray(values, dtype=np.float32)

    def transform_raw(self, smiles_list: List[str]) -> np.ndarray:
        return np.stack([self._raw_one(smiles) for smiles in smiles_list]).astype(np.float32)

    def fit(self, smiles_list: List[str]) -> "RDKitDescriptorExtractor":
        raw = self.transform_raw(smiles_list)
        mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        self.mean_ = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
        self.std_ = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0).astype(np.float32)
        return self

    def transform(self, smiles_list: List[str]) -> np.ndarray:
        raw = self.transform_raw(smiles_list)
        if self.mean_ is None or self.std_ is None:
            return raw
        return ((raw - self.mean_) / self.std_).astype(np.float32)

    def describe_backend(self) -> str:
        return "rdkit_descriptors" if self.rdkit_available else "zero_descriptors_fallback"
