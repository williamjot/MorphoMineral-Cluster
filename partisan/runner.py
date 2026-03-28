"""
partisan/runner.py
==================
Executa o PARTISAN em cada poro segmentado e agrega os resultados
em um pandas.DataFrame.

Interface com o PARTISAN
------------------------
O PARTISAN espera uma mascara binaria de UM UNICO poro por chamada.
A funcao analisePARTISAN(silhueta, do_plots, filename) retorna um dict
com 41 parametros morfometricos.

Este modulo itera sobre todos os poros do LabelingResult e chama
o PARTISAN para cada um, construindo um DataFrame de resultados.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _import_partisan(partisan_path: Optional[str | Path] = None):
    """
    Importa o modulo partisan.py.

    Tenta importar como modulo do pacote primeiro. Se nao encontrar,
    aceita um caminho explicito para o arquivo .py.
    """
    # 1) Tenta importar normalmente (partisan esta no PYTHONPATH)
    try:
        from partisan import analisePARTISAN
        logger.debug("PARTISAN importado via modulo Python")
        return analisePARTISAN
    except ImportError:
        pass

    # 2) Tenta importar do caminho explicito
    if partisan_path is not None:
        partisan_path = Path(partisan_path).resolve()
        if not partisan_path.exists():
            raise FileNotFoundError(f"partisan.py nao encontrado em: {partisan_path}")
        import importlib.util
        spec   = importlib.util.spec_from_file_location("partisan", partisan_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logger.info(f"PARTISAN carregado de: {partisan_path}")
        return module.analisePARTISAN

    raise ImportError(
        "Nao foi possivel importar o PARTISAN.\n"
        "Opcoes:\n"
        "  1. Coloque partisan.py na mesma pasta do projeto\n"
        "  2. Passe partisan_path='caminho/para/partisan.py' para run_partisan()"
    )


def run_partisan(
    labeling_result,
    image_name: str = "image",
    partisan_path: Optional[str | Path] = None,
    feret_angle_step: float = 0.5,
    min_area_px: int = 50,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Executa o PARTISAN em todos os poros do LabelingResult.

    Parameters
    ----------
    labeling_result  : LabelingResult de postprocessing.labeler
    image_name       : nome da imagem (usado como coluna no DataFrame)
    partisan_path    : caminho para partisan.py se nao estiver no PYTHONPATH
    feret_angle_step : passo angular Feret em graus (0.5 = maior precisao)
    min_area_px      : poros com area < este valor sao ignorados (seguranca extra)
    show_progress    : exibe barra de progresso tqdm

    Returns
    -------
    pd.DataFrame
        Uma linha por poro. Colunas:
          image_name, pore_id, area_px, bbox_r0, bbox_c0, bbox_r1, bbox_c1,
          + todas as 41 metricas do PARTISAN
    """
    from postprocessing.labeler import extract_pore_crop

    analisePARTISAN = _import_partisan(partisan_path)

    pores = labeling_result.pores
    if not pores:
        logger.warning("Nenhum poro encontrado para analise PARTISAN.")
        return pd.DataFrame()

    logger.info(f"Iniciando analise PARTISAN: {len(pores)} poros")

    rows = []
    iterator = tqdm(pores, desc="PARTISAN", unit="poro") if show_progress else pores

    for pore in iterator:
        if pore.area_px < min_area_px:
            continue

        # Recorta a mascara do poro com margem (reduz tempo de calculo)
        crop = extract_pore_crop(pore, padding=5)

        try:
            metrics = analisePARTISAN(
                silhueta=crop.astype(np.uint8) * 255,
                do_plots=False,
                filename=f"{image_name}_pore_{pore.label_id:04d}",
            )
        except Exception as e:
            logger.warning(
                f"PARTISAN falhou no poro {pore.label_id} "
                f"(area={pore.area_px}px): {e}"
            )
            continue

        # Mescla metadados do poro com metricas PARTISAN
        row = {
            "image_name": image_name,
            "pore_id":    pore.label_id,
            "area_px":    pore.area_px,
            "bbox_r0":    pore.bbox[0],
            "bbox_c0":    pore.bbox[1],
            "bbox_r1":    pore.bbox[2],
            "bbox_c1":    pore.bbox[3],
        }
        row.update(metrics)
        rows.append(row)

    if not rows:
        logger.warning("PARTISAN nao retornou resultados para nenhum poro.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Reordena colunas: metadados primeiro, depois metricas PARTISAN
    meta_cols    = ["image_name", "pore_id", "area_px",
                    "bbox_r0", "bbox_c0", "bbox_r1", "bbox_c1"]
    partisan_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + partisan_cols]

    logger.info(
        f"PARTISAN concluido: {len(df)} poros analisados, "
        f"{len(partisan_cols)} metricas cada"
    )
    return df


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estatisticas descritivas das metricas PARTISAN.

    Parameters
    ----------
    df : DataFrame retornado por run_partisan()

    Returns
    -------
    pd.DataFrame
        Estatisticas (mean, std, median, min, max, count) para cada metrica.
    """
    if df.empty:
        return pd.DataFrame()

    # Seleciona apenas colunas numericas (exclui metadados de string)
    numeric = df.select_dtypes(include=[np.number]).drop(
        columns=["pore_id", "area_px", "bbox_r0", "bbox_c0", "bbox_r1", "bbox_c1"],
        errors="ignore",
    )

    stats = numeric.agg(["count", "mean", "std", "median", "min", "max"])
    return stats.T.reset_index().rename(columns={"index": "metric"})
