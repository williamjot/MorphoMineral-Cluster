"""
core/segmentor.py (versão com EasIlastik)

Executa a inferência do Ilastik via EasIlastik.

Requerimentos:
  - Ilastik instalado como programa Windows (ilastik.org/download)
  - pip install easilastik h5py
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)


class IlastikSegmentor:
    """
    Wrapper sobre EasIlastik para segmentação de poros.

    Parameters
    ----------
    ilp_path      : caminho para o arquivo .ilp treinado
    pore_channel  : índice do canal de saída que representa os poros
    ilastik_exe   : caminho para o executável do Ilastik no Windows.
                    Se None, o EasIlastik tenta detectar automaticamente.
    """

    def __init__(
        self,
        ilp_path: str | Path,
        pore_channel: int = 0,
        ilastik_exe: str | Path | None = None,
    ) -> None:
        self.ilp_path     = Path(ilp_path).resolve()
        self.pore_channel = pore_channel
        self.ilastik_exe  = ilastik_exe

        try:
            import easilastik as EasIlastik
            self._eil = EasIlastik
        except ImportError:
            raise ImportError(
                "easilastik não instalado.\n"
                "Execute: pip install easilastik\n"
                "E certifique-se de que o Ilastik está instalado em ilastik.org/download"
            )

        logger.info(f"IlastikSegmentor pronto — modelo: {self.ilp_path}")

    def predict(self, image: np.ndarray, image_path: str | Path) -> np.ndarray:
        """
        Executa a classificação e retorna o mapa de probabilidades.

        O EasIlastik trabalha com arquivos em disco, então salva o
        resultado como HDF5 temporário e lê de volta como numpy array.

        Parameters
        ----------
        image      : np.ndarray — não usado diretamente, mas mantido
                     para compatibilidade com a interface original
        image_path : caminho da imagem de entrada no disco

        Returns
        -------
        np.ndarray shape (H, W, n_classes), dtype float32
        """
        image_path = Path(image_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Roda o Ilastik headless via EasIlastik
            kwargs = dict(
                input_path=str(image_path),
                model_path=str(self.ilp_path),
                result_base_path=str(tmp_path),
                export_source="Probabilities",
                output_format="hdf5",
            )
            if self.ilastik_exe:
                kwargs["ilastik_path"] = str(self.ilastik_exe)

            self._eil.run_ilastik(**kwargs)

            # Localiza o arquivo HDF5 gerado
            h5_files = list(tmp_path.glob("*.h5")) + list(tmp_path.glob("*.hdf5"))
            if not h5_files:
                raise RuntimeError(
                    f"EasIlastik não gerou arquivo HDF5 em {tmp_path}. "
                    "Verifique se o Ilastik está instalado corretamente."
                )

            # Lê o mapa de probabilidades
            prob_map = self._read_probability_hdf5(h5_files[0])

        logger.debug(f"Inferência concluída — shape: {prob_map.shape}")
        return prob_map

    def pore_probability(self, image: np.ndarray, image_path: str | Path) -> np.ndarray:
        """
        Retorna apenas o canal de probabilidade do poro.

        Returns
        -------
        np.ndarray shape (H, W), dtype float32
        """
        prob_map = self.predict(image, image_path)

        if self.pore_channel >= prob_map.shape[2]:
            raise ValueError(
                f"pore_channel={self.pore_channel} fora do intervalo: "
                f"o mapa tem {prob_map.shape[2]} canais."
            )
        return prob_map[:, :, self.pore_channel]

    @staticmethod
    def _read_probability_hdf5(h5_path: Path) -> np.ndarray:
        """
        Lê o mapa de probabilidades do HDF5 gerado pelo Ilastik.

        O Ilastik salva em formato (C, H, W) ou (H, W, C) dependendo
        da versão. Esta função normaliza para (H, W, C) sempre.
        """
        with h5py.File(h5_path, "r") as f:
            # Chave padrão do Ilastik
            key = "exported_data"
            if key not in f:
                key = list(f.keys())[0]
            data = f[key][()].astype(np.float32)

        # Normaliza shape para (H, W, C)
        if data.ndim == 3:
            # Se o menor eixo está na posição 0, é (C, H, W) → transpõe
            if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
                data = np.moveaxis(data, 0, -1)
        elif data.ndim == 4:
            # (1, H, W, C) ou (1, C, H, W) — remove dimensão de batch
            data = data[0]
            if data.shape[0] < data.shape[1]:
                data = np.moveaxis(data, 0, -1)

        return data