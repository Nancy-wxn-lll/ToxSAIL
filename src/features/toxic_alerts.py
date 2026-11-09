from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np


BASE_ALERT_NAMES = [
    "nitro_group",
    "aromatic_amine",
    "epoxide",
    "aldehyde",
    "michael_acceptor",
    "azo_group",
    "hydrazine",
    "sulfonyl_halide",
    "alkyl_halide",
    "anilide",
    "polyhalogenated_aromatic_ring",
]

EXPANDED_ALERT_NAMES = BASE_ALERT_NAMES + [
    "quinone",
    "phenol",
    "catechol",
    "thiophene",
    "furan",
    "acyl_halide",
    "isocyanate",
    "isothiocyanate",
    "aziridine",
    "alkyne",
    "aromatic_n_oxide",
    "hydroxylamine",
    "nitroso",
    "peroxide",
    "carboxylic_acid",
    "aniline_unsubstituted",
    "halogenated_alkene",
    "alpha_beta_unsaturated_carbonyl",
]

ALERT_NAMES = BASE_ALERT_NAMES


SMARTS_PATTERNS = {
    "nitro_group": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "aromatic_amine": "[a][NX3;H2,H1;!$(NC=O)]",
    "epoxide": "[OX2r3]1[CX4r3][CX4r3]1",
    "aldehyde": "[CX3H1](=O)[#6,#1]",
    "michael_acceptor": "[CX3]=[CX3]-[CX3](=O)",
    "azo_group": "[NX2]=[NX2]",
    "hydrazine": "[NX3][NX3]",
    "sulfonyl_halide": "S(=O)(=O)[F,Cl,Br,I]",
    "alkyl_halide": "[CX4][F,Cl,Br,I]",
    "anilide": "[c][NX3][CX3](=O)",
    "polyhalogenated_aromatic_ring": "a([F,Cl,Br,I])a([F,Cl,Br,I])",
    "quinone": "O=C1C=CC(=O)C=C1",
    "phenol": "[OX2H][c]",
    "catechol": "[OX2H][c][c][OX2H]",
    "thiophene": "c1ccsc1",
    "furan": "c1ccoc1",
    "acyl_halide": "[CX3](=O)[F,Cl,Br,I]",
    "isocyanate": "N=C=O",
    "isothiocyanate": "N=C=S",
    "aziridine": "[NX3r3]1[CX4r3][CX4r3]1",
    "alkyne": "[CX2]#[CX2]",
    "aromatic_n_oxide": "[n+]([O-])",
    "hydroxylamine": "[NX3][OX2H]",
    "nitroso": "[NX2]=O",
    "peroxide": "[OX2][OX2]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "aniline_unsubstituted": "[c][NX3H2]",
    "halogenated_alkene": "[CX3]=[CX3][F,Cl,Br,I]",
    "alpha_beta_unsaturated_carbonyl": "[CX3]=[CX3][CX3](=O)",
}


def _contains_any(text: str, needles: List[str]) -> bool:
    return any(needle in text for needle in needles)


def _fallback_matchers() -> Dict[str, Callable[[str], bool]]:
    return {
        "nitro_group": lambda s: _contains_any(
            s, ["[N+](=O)[O-]", "N(=O)=O", "N+(=O)[O-]", "NO2"]
        ),
        "aromatic_amine": lambda s: bool(
            re.search(r"c1.*N.*c", s) or re.search(r"cN|Nc", s)
        ),
        "epoxide": lambda s: _contains_any(s, ["C1OC1", "O1CC1", "C1CO1"]),
        "aldehyde": lambda s: _contains_any(s, ["C=O", "O=CC", "O=CH"]),
        "michael_acceptor": lambda s: _contains_any(
            s, ["C=CC(=O)", "C=CC=O", "C=CC(=O)O", "C=CC#N"]
        ),
        "azo_group": lambda s: "N=N" in s,
        "hydrazine": lambda s: bool(re.search(r"N[NH]", s) or "NN" in s),
        "sulfonyl_halide": lambda s: _contains_any(
            s, ["S(=O)(=O)Cl", "S(=O)(=O)F", "S(=O)(=O)Br", "S(=O)(=O)I"]
        ),
        "alkyl_halide": lambda s: bool(re.search(r"C(Cl|Br|I|F)", s)),
        "anilide": lambda s: _contains_any(s, ["cNC(=O)", "NC(=O)c", "cN(C(=O))"]),
        "polyhalogenated_aromatic_ring": lambda s: (
            s.count("c") >= 6
            and (s.count("Cl") + s.count("Br") + s.count("F") + s.count("I")) >= 2
        ),
        "quinone": lambda s: _contains_any(s, ["O=C1C=CC(=O)", "C1=CC(=O)C=CC1=O"]),
        "phenol": lambda s: _contains_any(s, ["Oc", "cO"]),
        "catechol": lambda s: _contains_any(s, ["Oc1ccccc1O", "O-c1ccccc1-O"]),
        "thiophene": lambda s: "s1" in s and s.count("c") >= 4,
        "furan": lambda s: "o1" in s and s.count("c") >= 4,
        "acyl_halide": lambda s: bool(re.search(r"C\(=O\)(Cl|Br|F|I)", s)),
        "isocyanate": lambda s: "N=C=O" in s,
        "isothiocyanate": lambda s: "N=C=S" in s,
        "aziridine": lambda s: _contains_any(s, ["N1CC1", "C1NC1"]),
        "alkyne": lambda s: "#" in s,
        "aromatic_n_oxide": lambda s: "[n+]" in s and "[O-]" in s,
        "hydroxylamine": lambda s: _contains_any(s, ["NO", "NHO"]),
        "nitroso": lambda s: "N=O" in s,
        "peroxide": lambda s: "OO" in s,
        "carboxylic_acid": lambda s: _contains_any(s, ["C(=O)O", "C(=O)[O-]"]),
        "aniline_unsubstituted": lambda s: bool(re.search(r"cN|Nc", s)),
        "halogenated_alkene": lambda s: bool(re.search(r"C=C(Cl|Br|F|I)", s)),
        "alpha_beta_unsaturated_carbonyl": lambda s: _contains_any(
            s, ["C=CC(=O)", "C=CC=O"]
        ),
    }


@dataclass
class ToxicAlertExtractor:
    """Extract structural toxic alert vectors from SMILES.

    RDKit SMARTS matching is used when RDKit is installed. In restricted
    environments the extractor falls back to transparent string/regex
    approximations so demos and tests remain runnable.
    """

    use_rdkit: bool = True
    alert_set: str = "basic"

    def __post_init__(self) -> None:
        if self.alert_set not in {"basic", "expanded"}:
            raise ValueError("alert_set must be 'basic' or 'expanded'")
        self.alert_names = list(
            EXPANDED_ALERT_NAMES if self.alert_set == "expanded" else BASE_ALERT_NAMES
        )
        self.rdkit_available = False
        self._compiled_smarts = {}
        self._fallback = _fallback_matchers()
        if self.use_rdkit:
            try:
                from rdkit import Chem

                self.Chem = Chem
                self.rdkit_available = True
                for name in self.alert_names:
                    mol = Chem.MolFromSmarts(SMARTS_PATTERNS[name])
                    self._compiled_smarts[name] = mol
            except Exception:
                self.rdkit_available = False

    @property
    def dim(self) -> int:
        return len(self.alert_names)

    def transform_one(self, smiles: str) -> np.ndarray:
        smiles = "" if smiles is None else str(smiles)
        if self.rdkit_available:
            mol = self.Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(self.dim, dtype=np.float32)
            values = []
            for name in self.alert_names:
                pattern = self._compiled_smarts.get(name)
                values.append(float(pattern is not None and mol.HasSubstructMatch(pattern)))
            return np.asarray(values, dtype=np.float32)

        values = [float(self._fallback[name](smiles)) for name in self.alert_names]
        return np.asarray(values, dtype=np.float32)

    def transform(self, smiles_list: List[str]) -> np.ndarray:
        return np.stack([self.transform_one(smiles) for smiles in smiles_list]).astype(np.float32)

    def describe_backend(self) -> str:
        suffix = f"_{self.alert_set}"
        return ("rdkit_smarts" if self.rdkit_available else "fallback_string_matchers") + suffix
