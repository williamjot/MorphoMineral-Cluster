"""
validator.py
============
Validacao de imagens antes da inferencia.

Verifica compatibilidade de shape e tipo com o modelo carregado do .ilp.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ImageValidator:
    """
    Valida imagens contra os requisitos do modelo Ilastik.

    Parameters
    ----------
    expected_channels : int
        Numero de canais esperado pelo modelo (de ILPMetadata.n_input_channels).
        Use None para desativar a verificacao de canais.
    """

    def __init__(self, expected_channels: Optional[int] = None) -> None:
        self.expected_channels = expected_channels

    def validate(self, image: np.ndarray, name: str = "imagem") -> None:
        """
        Valida a imagem e levanta ValueError se houver incompatibilidade.

        Parameters
        ----------
        image : np.ndarray
            Imagem a ser validada (antes ou depois da normalizacao).
        name  : str
            Nome para usar nas mensagens de erro (ex.: nome do arquivo).

        Raises
        ------
        ValueError  : se a imagem for invalida ou incompativel com o modelo
        """
        if not isinstance(image, np.ndarray):
            raise ValueError(f"[{name}] Esperado np.ndarray, obtido {type(image)}")

        if image.ndim not in (2, 3):
            raise ValueError(
                f"[{name}] Dimensoes invalidas: {image.ndim}D. "
                "Esperado 2D (H,W) ou 3D (H,W,C)."
            )

        if image.size == 0:
            raise ValueError(f"[{name}] Imagem vazia (size=0).")

        if image.shape[0] < 10 or image.shape[1] < 10:
            raise ValueError(
                f"[{name}] Imagem muito pequena: {image.shape}. "
                "Minimo recomendado: 10x10 pixels."
            )

        # Verificacao de canais
        if self.expected_channels is not None:
            actual_channels = image.shape[2] if image.ndim == 3 else 1
            if actual_channels != self.expected_channels:
                raise ValueError(
                    f"[{name}] Numero de canais incompativel: "
                    f"imagem tem {actual_channels} canal(is), "
                    f"modelo espera {self.expected_channels}. "
                    "Verifique se a imagem e do mesmo tipo que as usadas no treino."
                )

        # Avisos nao-bloqueadores
        self._warn_if_unusual(image, name)
        logger.debug(
            f"[{name}] Validacao OK — shape={image.shape}, dtype={image.dtype}"
        )

    @staticmethod
    def _warn_if_unusual(image: np.ndarray, name: str) -> None:
        """Emite avisos para situacoes incomuns mas nao invalidas."""
        if np.all(image == 0):
            logger.warning(f"[{name}] Imagem totalmente preta (todos zeros).")
        elif np.all(image == image.flat[0]):
            logger.warning(f"[{name}] Imagem com intensidade constante.")

        if image.dtype == np.float32 or image.dtype == np.float64:
            vmax = float(image.max())
            if vmax > 1.0 + 1e-5:
                logger.warning(
                    f"[{name}] Imagem float com max={vmax:.3f} > 1.0. "
                    "Normalize antes da inferencia."
                )
