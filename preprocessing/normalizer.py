"""
normalizer.py
=============
Normalizacao e conversao de tipo para preparo de imagens antes da inferencia.

O Ilastik espera float32 em [0, 1].
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

NormMethod = Literal["percentile", "minmax", "none"]


def normalize(
    image: np.ndarray,
    method: NormMethod = "percentile",
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """
    Normaliza uma imagem para float32 com valores em [0, 1].

    Parameters
    ----------
    image  : np.ndarray — shape (H,W) ou (H,W,C)
    method : "percentile" | "minmax" | "none"
        - percentile: usa percentis p_low e p_high para clipping robusto
          (recomendado para imagens com outliers de brilho)
        - minmax    : usa min e max absolutos
        - none      : apenas converte para float32 sem reescalar
    p_low  : percentil inferior (usado com method="percentile")
    p_high : percentil superior (usado com method="percentile")

    Returns
    -------
    np.ndarray
        float32, shape igual ao input, valores em [0, 1].
    """
    image = np.asarray(image, dtype=np.float64)

    if method == "none":
        return image.astype(np.float32)

    if method == "percentile":
        lo = np.percentile(image, p_low)
        hi = np.percentile(image, p_high)
    elif method == "minmax":
        lo = float(image.min())
        hi = float(image.max())
    else:
        raise ValueError(f"Metodo de normalizacao desconhecido: '{method}'")

    if hi == lo:
        logger.warning(
            "Imagem com intensidade constante (hi == lo). "
            "Retornando array de zeros."
        )
        return np.zeros_like(image, dtype=np.float32)

    normalized = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    logger.debug(
        f"Normalizacao '{method}': [{lo:.2f}, {hi:.2f}] -> [0, 1]"
    )
    return normalized.astype(np.float32)


def normalize_per_channel(
    image: np.ndarray,
    method: NormMethod = "percentile",
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """
    Normaliza cada canal de uma imagem multicanal independentemente.

    Para imagens (H, W) delega para normalize(). Para (H, W, C)
    normaliza cada canal separado.

    Parameters
    ----------
    image  : np.ndarray — shape (H,W) ou (H,W,C)

    Returns
    -------
    np.ndarray float32
    """
    if image.ndim == 2:
        return normalize(image, method, p_low, p_high)

    if image.ndim == 3:
        channels = []
        for c in range(image.shape[2]):
            ch = normalize(image[:, :, c], method, p_low, p_high)
            channels.append(ch)
        return np.stack(channels, axis=-1).astype(np.float32)

    raise ValueError(f"Shape nao suportado: {image.shape}")
