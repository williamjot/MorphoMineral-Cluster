"""
morphology.py
=============
Operacoes morfologicas binarias para refinamento da mascara de poros.

Sequencia recomendada:
  1. opening  — remove ruido (ilhas pequenas fora dos poros)
  2. closing  — fecha buracos dentro dos poros
  3. fill_holes — preenche cavidades internas
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
)

logger = logging.getLogger(__name__)


def apply_morphology(
    binary: np.ndarray,
    opening_radius: int = 2,
    closing_radius: int = 2,
    fill_holes: bool = True,
) -> np.ndarray:
    """
    Aplica morfologia binaria a uma mascara de poros.

    Parameters
    ----------
    binary          : np.ndarray bool, shape (H, W)
    opening_radius  : raio do elemento estruturante para opening.
                      0 = operacao desativada.
    closing_radius  : raio do elemento estruturante para closing.
                      0 = operacao desativada.
    fill_holes      : se True, preenche cavidades internas dos poros.

    Returns
    -------
    np.ndarray bool, shape (H, W)
    """
    if binary.ndim != 2:
        raise ValueError(f"Esperado array 2D, obtido {binary.ndim}D")

    result = binary.astype(bool)

    before = int(result.sum())

    if opening_radius > 0:
        selem  = disk(opening_radius)
        result = binary_opening(result, footprint=selem)
        logger.debug(
            f"Opening (r={opening_radius}): {before:,} -> {int(result.sum()):,} pixels"
        )

    if closing_radius > 0:
        selem  = disk(closing_radius)
        result = binary_closing(result, footprint=selem)
        logger.debug(
            f"Closing (r={closing_radius}): {int(result.sum()):,} pixels"
        )

    if fill_holes:
        result = binary_fill_holes(result)
        logger.debug(
            f"Fill holes: {int(result.sum()):,} pixels"
        )

    after = int(result.sum())
    logger.info(
        f"Morfologia concluida: {before:,} -> {after:,} pixels de poro "
        f"(delta: {after - before:+,})"
    )
    return result
