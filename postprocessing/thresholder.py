"""
thresholder.py
==============
Binarizacao do mapa de probabilidade de poros.

Tres estrategias:
  - otsu     : threshold automatico de Otsu (recomendado para dados variados)
  - fixed    : valor fixo definido no config.yaml
  - adaptive : threshold por blocos (util para imagens com iluminacao irregular)
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from skimage.filters import threshold_otsu, threshold_local

logger = logging.getLogger(__name__)

ThresholdMethod = Literal["otsu", "fixed", "adaptive"]


def threshold_probability_map(
    prob_map: np.ndarray,
    method: ThresholdMethod = "otsu",
    fixed_value: float = 0.5,
    adaptive_block_size: int = 51,
) -> np.ndarray:
    """
    Binariza o mapa de probabilidade do poro.

    Parameters
    ----------
    prob_map           : np.ndarray, shape (H, W), dtype float32, valores [0,1]
    method             : "otsu" | "fixed" | "adaptive"
    fixed_value        : valor do threshold (usado quando method="fixed")
    adaptive_block_size: tamanho do bloco local (metodo adaptive, deve ser impar)

    Returns
    -------
    np.ndarray
        Mascara binaria, shape (H, W), dtype bool.
        True = pixel classificado como poro.
    """
    if prob_map.ndim != 2:
        raise ValueError(
            f"prob_map deve ser 2D, obtido shape {prob_map.shape}"
        )

    if method == "otsu":
        thresh = threshold_otsu(prob_map)
        binary = prob_map > thresh
        logger.info(f"Threshold Otsu = {thresh:.4f}")

    elif method == "fixed":
        binary = prob_map > fixed_value
        logger.info(f"Threshold fixo = {fixed_value:.4f}")

    elif method == "adaptive":
        # Garante block_size impar
        bs = adaptive_block_size if adaptive_block_size % 2 == 1 else adaptive_block_size + 1
        local_thresh = threshold_local(prob_map, block_size=bs)
        binary = prob_map > local_thresh
        logger.info(f"Threshold adaptativo (block_size={bs})")

    else:
        raise ValueError(f"Metodo de threshold desconhecido: '{method}'")

    n_pore_px = int(binary.sum())
    total_px  = binary.size
    logger.info(
        f"Binarizacao concluida: {n_pore_px:,} pixels de poro "
        f"({100 * n_pore_px / total_px:.2f}% da area total)"
    )
    return binary
