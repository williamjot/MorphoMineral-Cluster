"""
loader.py
=========
Carregamento de imagens EDS/CBS/QEMSCAN.

Suporta: TIFF (single e multi-page), PNG, BMP, JPEG.
Retorna sempre np.ndarray com shape (H, W) ou (H, W, C).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Extensoes suportadas em ordem de preferencia de leitor
_TIFF_EXTS = {".tif", ".tiff"}
_IMG_EXTS  = {".png", ".bmp", ".jpg", ".jpeg"}


def load_image(path: str | Path) -> np.ndarray:
    """
    Carrega uma imagem de qualquer formato suportado.

    Parameters
    ----------
    path : str | Path
        Caminho para o arquivo de imagem.

    Returns
    -------
    np.ndarray
        Shape (H, W) para grayscale ou (H, W, C) para multicanal.
        Dtype preservado conforme o arquivo original (uint8, uint16, float32...).
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de imagem nao encontrado: {path}")

    ext = path.suffix.lower()
    logger.info(f"Carregando imagem: {path}")

    if ext in _TIFF_EXTS:
        image = _load_tiff(path)
    else:
        image = _load_generic(path)

    logger.info(
        f"Imagem carregada — shape: {image.shape}, dtype: {image.dtype}, "
        f"min={image.min():.4f}, max={image.max():.4f}"
    )
    return image


def _load_tiff(path: Path) -> np.ndarray:
    """Carrega TIFF com tifffile (preferido para dados cientificos)."""
    try:
        import tifffile
        image = tifffile.imread(str(path))
        return _fix_shape(image)
    except Exception as e:
        logger.warning(f"tifffile falhou ({e}), tentando imageio...")
        return _load_generic(path)


def _load_generic(path: Path) -> np.ndarray:
    """Carrega com imageio (fallback generico)."""
    try:
        import imageio.v3 as iio
        image = iio.imread(str(path))
        return _fix_shape(image)
    except Exception as e:
        raise OSError(
            f"Nao foi possivel carregar a imagem '{path}'.\n"
            f"Erro: {e}"
        ) from e


def _fix_shape(image: np.ndarray) -> np.ndarray:
    """
    Normaliza o shape da imagem.

    Regras:
      - 2D (H, W)       → grayscale, mantido como esta
      - 3D (H, W, C)    → multicanal, mantido
      - 3D (1, H, W)    → single-page TIFF, converte para (H, W)
      - 3D (C, H, W)    → TIFF multicanal first-axis, converte para (H, W, C)
      - 4D (1, H, W, C) → TIFF com dim de pagina, converte para (H, W, C)
    """
    if image.ndim == 2:
        return image

    if image.ndim == 3:
        # (1, H, W) → (H, W)
        if image.shape[0] == 1:
            return image[0]
        # (H, W, C): ja no formato correto
        if image.shape[2] <= 4:
            return image
        # (C, H, W) onde C eh pequeno — casos como RGB salvo em first-axis
        if image.shape[0] <= 4:
            return np.moveaxis(image, 0, -1)
        return image

    if image.ndim == 4:
        # (1, H, W, C) ou (N, H, W, C): pega primeiro frame
        if image.shape[0] == 1:
            return image[0]
        logger.warning(
            f"Imagem 4D com {image.shape[0]} frames — usando apenas o primeiro."
        )
        return image[0]

    raise ValueError(f"Shape nao suportado: {image.shape}")


def list_images(directory: str | Path, extensions: Optional[set] = None) -> list[Path]:
    """
    Lista todas as imagens em um diretorio.

    Parameters
    ----------
    directory  : diretorio a varrer
    extensions : conjunto de extensoes (ex.: {'.tif', '.png'}). Se None, usa todas.

    Returns
    -------
    Lista de Path ordenada por nome.
    """
    if extensions is None:
        extensions = _TIFF_EXTS | _IMG_EXTS

    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Diretorio nao encontrado: {directory}")

    files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )
    logger.info(f"Encontradas {len(files)} imagens em {directory}")
    return files
