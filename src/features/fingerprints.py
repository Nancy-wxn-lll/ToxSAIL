from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import List

import numpy as np


def _disable_rdkit_warnings() -> None:
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.warning")
        RDLogger.DisableLog("rdApp.error")
    except Exception:
        pass


@dataclass
class MorganFingerprintExtractor:
    """RDKit Morgan/ECFP fingerprint features with a deterministic fallback."""

    n_bits: int = 2048
    radius: int = 2
    use_chirality: bool = True
    use_rdkit: bool = True

    def __post_init__(self) -> None:
        self.rdkit_available = False
        self.Chem = None
        self.AllChem = None
        self.DataStructs = None
        if self.use_rdkit:
            try:
                _disable_rdkit_warnings()
                from rdkit import Chem, DataStructs
                from rdkit.Chem import AllChem

                self.Chem = Chem
                self.AllChem = AllChem
                self.DataStructs = DataStructs
                self.rdkit_available = True
            except Exception:
                self.rdkit_available = False

    @property
    def dim(self) -> int:
        return int(self.n_bits)

    def _fallback_one(self, smiles: str) -> np.ndarray:
        vec = np.zeros(self.n_bits, dtype=np.float32)
        text = "" if smiles is None else str(smiles)
        for size in (2, 3, 4):
            if len(text) < size:
                continue
            for idx in range(len(text) - size + 1):
                token = text[idx : idx + size]
                digest = hashlib.blake2b(f"{size}:{token}".encode("utf-8"), digest_size=8)
                bit = int.from_bytes(digest.digest(), "little") % self.n_bits
                vec[bit] = 1.0
        return vec

    def transform_one(self, smiles: str) -> np.ndarray:
        if not self.rdkit_available:
            return self._fallback_one(smiles)
        mol = self.Chem.MolFromSmiles("" if smiles is None else str(smiles))
        if mol is None:
            return np.zeros(self.n_bits, dtype=np.float32)
        fp = self.AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=int(self.radius),
            nBits=int(self.n_bits),
            useChirality=bool(self.use_chirality),
        )
        arr = np.zeros((self.n_bits,), dtype=np.int8)
        self.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.astype(np.float32)

    def transform(self, smiles_list: List[str]) -> np.ndarray:
        return np.stack([self.transform_one(smiles) for smiles in smiles_list]).astype(np.float32)

    def describe_backend(self) -> str:
        if self.rdkit_available:
            return f"rdkit_morgan_radius{self.radius}_{self.n_bits}"
        return f"hashed_smiles_ngrams_{self.n_bits}"
