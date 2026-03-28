"""
labeler.py
==========
Identificacao de poros individuais via componentes conectados.

Gera o mapa de labels que sera percorrido pelo PARTISAN,
alem de filtrar poros por area minima/maxima.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.ndimage import label as scipy_label

logger = logging.getLogger(__name__)


@dataclass
class PoreRegion:
    """
    Representacao de um poro individual apos labeling.

    Atributos
    ---------
    label_id   : ID unico no mapa de labels (inteiro >= 1)
    area_px    : area em pixels
    bbox       : (row_min, col_min, row_max, col_max)
    binary_mask: mascara binaria do poro isolado, shape (H, W) da imagem original
    """
    label_id:    int
    area_px:     int
    bbox:        tuple  # (r0, c0, r1, c1)
    binary_mask: np.ndarray


@dataclass
class LabelingResult:
    """
    Resultado completo do labeling de uma imagem.

    Atributos
    ---------
    label_map    : mapa de inteiros shape (H, W); 0 = fundo, 1..N = poros
    n_total      : total de componentes encontrados antes do filtro
    n_accepted   : componentes apos filtro de area
    total_area_px: soma das areas dos poros aceitos (pixels)
    porosity_pct : porosidade percentual = total_area_px / pixels_totais * 100
    pores        : lista de PoreRegion para cada poro aceito
    """
    label_map:     np.ndarray
    n_total:       int
    n_accepted:    int
    total_area_px: int
    porosity_pct:  float
    pores:         List[PoreRegion]


def label_pores(
    binary: np.ndarray,
    min_area_px: int = 50,
    max_area_px: int = 0,
    connectivity: int = 2,
) -> LabelingResult:
    """
    Identifica e filtra poros individuais na mascara binaria.

    Parameters
    ----------
    binary       : np.ndarray bool, shape (H, W)
    min_area_px  : area minima de um poro para ser incluido (pixels)
    max_area_px  : area maxima (0 = sem limite)
    connectivity : 1 = 4-conectividade, 2 = 8-conectividade

    Returns
    -------
    LabelingResult
    """
    if binary.ndim != 2:
        raise ValueError(f"Esperado array 2D, obtido {binary.ndim}D")

    # Estrutura de conectividade para scipy.ndimage.label
    if connectivity == 2:
        structure = np.ones((3, 3), dtype=int)  # 8-conectado
    else:
        structure = None  # padrao: 4-conectado

    label_map, n_total = scipy_label(binary.astype(bool), structure=structure)
    logger.info(f"Componentes conectados encontrados: {n_total}")

    pores: List[PoreRegion] = []
    accepted_map = np.zeros_like(label_map)

    for lbl_id in range(1, n_total + 1):
        mask = (label_map == lbl_id)
        area = int(mask.sum())

        # Filtro de area
        if area < min_area_px:
            continue
        if max_area_px > 0 and area > max_area_px:
            continue

        rows, cols = np.where(mask)
        bbox = (int(rows.min()), int(cols.min()),
                int(rows.max()), int(cols.max()))

        pores.append(PoreRegion(
            label_id=lbl_id,
            area_px=area,
            bbox=bbox,
            binary_mask=mask,
        ))
        accepted_map[mask] = lbl_id

    total_area = sum(p.area_px for p in pores)
    porosity   = 100.0 * total_area / binary.size

    result = LabelingResult(
        label_map=accepted_map,
        n_total=n_total,
        n_accepted=len(pores),
        total_area_px=total_area,
        porosity_pct=porosity,
        pores=pores,
    )

    logger.info(
        f"Poros aceitos: {result.n_accepted} / {n_total} "
        f"(filtro: area >= {min_area_px}px)"
    )
    logger.info(
        f"Porosidade total: {porosity:.3f}% "
        f"({total_area:,} / {binary.size:,} pixels)"
    )
    return result


def extract_pore_crop(
    pore: PoreRegion,
    padding: int = 5,
) -> np.ndarray:
    """
    Recorta a mascara binaria de um poro com padding.

    Util para passar ao PARTISAN um crop justo ao redor do poro
    (reduz tempo de calculo das geometrias).

    Parameters
    ----------
    pore    : PoreRegion com binary_mask de shape (H, W) da imagem completa
    padding : pixels de margem ao redor do bbox

    Returns
    -------
    np.ndarray bool, shape do crop com padding
    """
    H, W = pore.binary_mask.shape
    r0, c0, r1, c1 = pore.bbox
    r0 = max(0, r0 - padding)
    c0 = max(0, c0 - padding)
    r1 = min(H - 1, r1 + padding)
    c1 = min(W - 1, c1 + padding)
    return pore.binary_mask[r0:r1 + 1, c0:c1 + 1]
