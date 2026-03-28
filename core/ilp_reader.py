"""

Leitura de metadados do arquivo .ilp gerado pelo Ilastik 1.4.x (ATESTAR A VERSÃO DO ILASTIK É MUITO IMPORTANTE)

O script extrai:

- Nomes e indices das classes (label_names)
- Features selecionadas no treinamento
- Numero de canais esperados na entrada
- Versao do Ilastik e tipo de workflow
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ILPMetadata:
    """
    Metadados extraidos de um arquivo .ilp do Ilastik.

    Atributos
    ---------
    ilp_path         : caminho absoluto para o arquivo
    label_names      : lista de nomes de classes na ordem interna do Ilastik
    pore_class_index : indice da classe 'poro' em label_names
    n_classes        : numero total de classes
    n_input_channels : numero de canais de entrada esperado
    selected_features: lista legivel das features selecionadas
    ilastik_version  : versao do Ilastik que gerou o arquivo
    workflow_type    : tipo de workflow (ex.: 'PixelClassificationWorkflow')
    """
    ilp_path: Path
    label_names: List[str] = field(default_factory=list)
    pore_class_index: int = 0
    n_classes: int = 0
    n_input_channels: int = 1
    selected_features: List[str] = field(default_factory=list)
    ilastik_version: str = "unknown"
    workflow_type: str = "unknown"

    def describe(self) -> str:
        lines = [
            f"Arquivo .ilp   : {self.ilp_path}",
            f"Workflow       : {self.workflow_type}",
            f"Versao Ilastik : {self.ilastik_version}",
            f"Canais entrada : {self.n_input_channels}",
            f"Classes ({self.n_classes}):",
        ]
        for i, name in enumerate(self.label_names):
            marker = " <- PORO" if i == self.pore_class_index else ""
            lines.append(f"  [{i}] {name}{marker}")
        if self.selected_features:
            lines.append(f"Features ({len(self.selected_features)}):")
            for fname in self.selected_features[:10]:
                lines.append(f"  - {fname}")
            if len(self.selected_features) > 10:
                lines.append(f"  ... (+{len(self.selected_features) - 10} outras)")
        return "\n".join(lines)


class ILPReader:
    """
    Le metadados de um arquivo .ilp do Ilastik 1.4.x.

    Uso
    ---
    >>> reader = ILPReader("model.ilp")
    >>> meta = reader.read()
    >>> print(meta.describe())
    """

    _PORE_KEYWORDS = ["pore", "poro", "void", "vazio", "porosity", "porosidade"]

    def __init__(self, ilp_path: str | Path) -> None:
        self.ilp_path = Path(ilp_path).resolve()
        if not self.ilp_path.exists():
            raise FileNotFoundError(f"Arquivo .ilp nao encontrado: {self.ilp_path}")

    def read(self, pore_class_index: Optional[int] = None) -> ILPMetadata:
        """
        Le todos os metadados do arquivo .ilp.

        Parameters
        ----------
        pore_class_index : int, opcional
            Forca o indice da classe poro. Se None, detecta automaticamente.
        """
        logger.info(f"Lendo metadados de: {self.ilp_path}")
        meta = ILPMetadata(ilp_path=self.ilp_path)

        with h5py.File(self.ilp_path, "r") as f:
            meta.ilastik_version   = self._read_version(f)
            meta.workflow_type     = self._read_workflow(f)
            meta.label_names       = self._read_label_names(f)
            meta.n_classes         = len(meta.label_names)
            meta.n_input_channels  = self._read_n_channels(f)
            meta.selected_features = self._read_features(f)

        if pore_class_index is not None:
            meta.pore_class_index = pore_class_index
        else:
            meta.pore_class_index = self._auto_detect_pore_index(meta.label_names)

        logger.info(
            f"Leitura concluida: {meta.n_classes} classes, "
            f"canal poro = {meta.pore_class_index}"
        )
        return meta

    # ------------------------------------------------------------------

    @staticmethod
    def _read_version(f: h5py.File) -> str:
        for key in ["ilastikVersion", "IlastikVersion"]:
            if key in f:
                v = f[key][()]
                return v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
        return "unknown"

    @staticmethod
    def _read_workflow(f: h5py.File) -> str:
        for key in ["workflowName", "WorkflowName"]:
            if key in f:
                v = f[key][()]
                return v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
        if "PixelClassification" in f:
            return "PixelClassificationWorkflow"
        return "unknown"

    @staticmethod
    def _read_label_names(f: h5py.File) -> List[str]:
        """Extrai nomes das classes de PixelClassification/LabelNames."""
        if "PixelClassification/LabelNames" in f:
            raw = f["PixelClassification/LabelNames"][()]
            if raw.ndim == 0:
                raw = [raw]
            names = []
            for item in raw:
                names.append(item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item))
            if names:
                logger.debug(f"Classes: {names}")
                return names

        # Fallback: conta LabelSets
        if "PixelClassification/LabelSets" in f:
            n = len([k for k in f["PixelClassification/LabelSets"].keys() if k.startswith("labels")])
            if n:
                logger.warning("Nomes de classes nao encontrados — usando 'Class_N'")
                return [f"Class_{i}" for i in range(n)]

        logger.warning("Nomes de classes nao encontrados no .ilp.")
        return []

    @staticmethod
    def _read_n_channels(f: h5py.File) -> int:
        """Infere numero de canais de entrada a partir dos metadados de Input Data."""
        base = "Input Data/infos"
        if base not in f:
            return 1
        group = f[base]
        for lane_key in group.keys():
            lane = group[lane_key]
            for role_key in lane.keys():
                role = lane[role_key]
                if "axisorder" in role:
                    axisorder = role["axisorder"][()]
                    if isinstance(axisorder, (bytes, np.bytes_)):
                        axisorder = axisorder.decode("utf-8")
                    if "c" in axisorder and "shape" in role:
                        shape   = role["shape"][()]
                        c_idx   = axisorder.index("c")
                        if c_idx < len(shape):
                            return int(shape[c_idx])
        return 1

    @staticmethod
    def _read_features(f: h5py.File) -> List[str]:
        """Le features selecionadas de FeatureSelections/."""
        if "FeatureSelections" not in f:
            return []
        fs = f["FeatureSelections"]
        feature_ids, scales, features = [], [], []

        if "FeatureIds" in fs:
            for item in fs["FeatureIds"][()]:
                feature_ids.append(item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item))
        if "Scales" in fs:
            scales = list(fs["Scales"][()])
        if "SelectionMatrix" in fs:
            matrix = fs["SelectionMatrix"][()].astype(bool)
            for fi, fname in enumerate(feature_ids):
                for si, scale in enumerate(scales):
                    if fi < matrix.shape[0] and si < matrix.shape[1] and matrix[fi, si]:
                        features.append(f"{fname} sigma={scale:.1f}")
        return features

    def _auto_detect_pore_index(self, label_names: List[str]) -> int:
        for i, name in enumerate(label_names):
            if any(kw in name.lower() for kw in self._PORE_KEYWORDS):
                logger.info(f"Classe poro detectada: indice {i} ('{name}')")
                return i
        logger.warning(
            f"Classe poro nao detectada em {label_names}. "
            "Usando indice 0. Corrija 'pore_class_index' no config.yaml."
        )
        return 0


def inspect_ilp(ilp_path: str) -> None:
    """Imprime resumo do .ilp — util para descobrir o indice do poro."""
    reader = ILPReader(ilp_path)
    meta   = reader.read()
    print("\n" + "=" * 60)
    print("METADADOS DO ARQUIVO .ILP")
    print("=" * 60)
    print(meta.describe())
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python ilp_reader.py <arquivo.ilp>")
        sys.exit(1)
    inspect_ilp(sys.argv[1])
