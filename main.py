"""
main.py
=======
Entry point principal do pipeline de segmentacao de poros EDS/CBS/QEMSCAN.

Uso rapido
----------
  python main.py --ilp model.ilp --image imagem.tif
  python main.py --ilp model.ilp --dir pasta_com_imagens/
  python main.py --inspect model.ilp      (apenas inspeciona o .ilp)
  python main.py --help

Fluxo completo
--------------
  1. Leitura dos metadados do .ilp  (core.ilp_reader)
  2. Carregamento da imagem          (preprocessing.loader)
  3. Validacao                       (preprocessing.validator)
  4. Normalizacao                    (preprocessing.normalizer)
  5. Segmentacao via Ilastik         (core.segmentor)
  6. Threshold + morfologia          (postprocessing)
  7. Labeling de poros individuais   (postprocessing.labeler)
  8. Analise morfometrica PARTISAN   (partisan.runner)
  9. Exportacao de tabelas           (output.exporter)
  10. Visualizacoes                  (output.visualizer)
  11. Relatorio PDF                  (output.reporter)
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import yaml

# ──────────────────────────────────────────────────────────────────────────────
# Configuracao de logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Carregamento de configuracao
# ──────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline para uma imagem
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    image_path: Path,
    ilp_path: Path,
    cfg: dict,
    segmentor,         # IlastikSegmentor ja inicializado
    meta,              # ILPMetadata
    partisan_path: Path | None,
) -> dict:
    """
    Executa o pipeline completo para uma unica imagem.

    Retorna um dict com os paths dos arquivos gerados.
    """
    logger = logging.getLogger("pipeline")

    from preprocessing.loader    import load_image
    from preprocessing.normalizer import normalize_per_channel
    from preprocessing.validator  import ImageValidator
    from postprocessing.thresholder import threshold_probability_map
    from postprocessing.morphology  import apply_morphology
    from postprocessing.labeler     import label_pores
    from partisan.runner            import run_partisan, summary_statistics
    from output.exporter            import ResultExporter
    from output.visualizer          import Visualizer
    from output.reporter            import PDFReporter
    import pandas as pd

    stem       = image_path.stem
    output_dir = Path(cfg["output"]["output_dir"]) / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    pp_cfg   = cfg["preprocessing"]
    post_cfg = cfg["postprocessing"]
    out_cfg  = cfg["output"]

    # ── Passo 1: Carregamento ────────────────────────────────────────────────
    logger.info(f"[{stem}] Carregando imagem...")
    image = load_image(image_path)

    # ── Passo 2: Validacao ───────────────────────────────────────────────────
    validator = ImageValidator(expected_channels=meta.n_input_channels)
    try:
        validator.validate(image, name=stem)
    except ValueError as exc:
        # Se falhou apenas por canais, tentamos converter automaticamente
        if "Numero de canais incompativel" in str(exc):
            logger.warning(f"[{stem}] {exc} -> Tentando conversao automatica...")
            if meta.n_input_channels == 1:
                # Converte para grayscale
                if image.ndim == 3:
                    if image.shape[2] == 4: # RGBA -> RGB
                        image = image[:, :, :3]
                    # Simples media ou luminancia
                    image = np.mean(image, axis=2).astype(image.dtype)
                    logger.info(f"[{stem}] Imagem convertida para Grayscale (1 canal).")
                else:
                    raise exc
            elif meta.n_input_channels == 3 and (image.ndim == 2 or image.shape[2] == 1):
                # Grayscale -> RGB
                if image.ndim == 2:
                    image = np.stack([image]*3, axis=-1)
                else:
                    image = np.tile(image, (1, 1, 3))
                logger.info(f"[{stem}] Imagem convertida para RGB (3 canais).")
            else:
                raise exc
            # Re-valida apos conversao
            validator.validate(image, name=stem)
        else:
            raise exc

    # ── Passo 3: Normalizacao ────────────────────────────────────────────────
    logger.info(f"[{stem}] Normalizando...")
    norm_image = normalize_per_channel(
        image,
        method=pp_cfg.get("normalization", "percentile"),
        p_low=pp_cfg.get("norm_percentile_low", 1),
        p_high=pp_cfg.get("norm_percentile_high", 99),
    )

    # ── Passo 4: Segmentacao Ilastik ─────────────────────────────────────────
    logger.info(f"[{stem}] Executando segmentacao Ilastik...")
    t0 = time.time()
    prob_map_all = segmentor.predict(norm_image, image_path=image_path)
    if prob_map_all.ndim != 3:
        raise ValueError(
            f"Mapa de probabilidade invalido: esperado 3D (H, W, C), obtido {prob_map_all.shape}"
        )
    if meta.pore_class_index >= prob_map_all.shape[2]:
        raise ValueError(
            f"pore_class_index={meta.pore_class_index} fora do intervalo para {prob_map_all.shape[2]} classes."
        )
    prob_map = prob_map_all[:, :, meta.pore_class_index]
    class_ids = np.argmax(prob_map_all, axis=2)
    n_classes = prob_map_all.shape[2]
    class_names = meta.label_names if meta.label_names else [f"Class_{i}" for i in range(n_classes)]
    if len(class_names) < n_classes:
        class_names = class_names + [f"Class_{i}" for i in range(len(class_names), n_classes)]
    logger.info(f"[{stem}] Inferencia concluida em {time.time() - t0:.1f}s")

    import unicodedata
    import re

    def _class_dir_label(name: str, idx: int) -> str:
        txt = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
        txt = re.sub(r"[^A-Za-z0-9]+", "_", txt).strip("_")
        if not txt:
            txt = f"Class_{idx}"
        low = txt.lower()
        if low in {"poro", "pore"}:
            return "Poro"
        if low == "matriz":
            return "Matriz"
        if low in {"particula", "particle"}:
            return "Particula"
        return txt[:1].upper() + txt[1:]

    class_output_dirs = {}
    used_labels = set()
    for class_idx, class_name in enumerate(class_names):
        label = _class_dir_label(class_name, class_idx)
        unique_label = label
        suffix = 1
        while unique_label.lower() in used_labels:
            unique_label = f"{label}_{suffix}"
            suffix += 1
        used_labels.add(unique_label.lower())

        class_dir = output_dir / unique_label
        class_dir.mkdir(parents=True, exist_ok=True)
        class_output_dirs[class_idx] = class_dir

    class_colors = [
        (1.0, 0.15, 0.15),
        (0.20, 0.70, 0.25),
        (0.20, 0.45, 0.95),
        (0.95, 0.65, 0.20),
    ]

    # ── Processamento por Classe ─────────────────────────────────────────────
    logger.info(f"[{stem}] Iniciando analise multi-classe para {n_classes} classes...")
    
    generated_files = {}
    total_px = class_ids.size

    for class_idx in range(n_classes):
        class_name = class_names[class_idx]
        class_dir = class_output_dirs[class_idx]
        logger.info(f"[{stem}] >>> Analisando Classe {class_idx}: {class_name}")

        # ── Passo 5: Threshold (especifico para a classe) ─────────────────────
        class_prob = prob_map_all[:, :, class_idx]
        binary = threshold_probability_map(
            class_prob,
            method=post_cfg.get("threshold_method", "otsu"),
            fixed_value=post_cfg.get("threshold_value", 0.5),
        )

        # ── Passo 6: Morfologia ──────────────────────────────────────────────
        binary = apply_morphology(
            binary,
            opening_radius=post_cfg.get("morphology_opening_radius", 2),
            closing_radius=post_cfg.get("morphology_closing_radius", 2),
            fill_holes=post_cfg.get("morphology_fill_holes", True),
        )

        # ── Passo 7: Labeling ────────────────────────────────────────────────
        labeling = label_pores(
            binary,
            min_area_px=post_cfg.get("min_pore_area_px", 50),
            max_area_px=post_cfg.get("max_pore_area_px", 0),
        )

        # ── Passo 8: PARTISAN ────────────────────────────────────────────────
        if labeling.n_accepted > 0:
            logger.info(f"[{stem}] Executando PARTISAN para {class_name} ({labeling.n_accepted} objetos)...")
            df = run_partisan(
                labeling_result=labeling,
                image_name=f"{stem}_{class_dir.name}",
                partisan_path=partisan_path,
                show_progress=False,
            )
            stats_df = summary_statistics(df) if not df.empty else None
        else:
            logger.warning(f"[{stem}] Nenhum objeto aceito para a classe {class_name}.")
            df = pd.DataFrame()
            stats_df = None

        # ── Passo 9: Exportação de tabelas da classe ──────────────────────────
        exporter = ResultExporter(
            output_dir=class_dir,
            formats=out_cfg.get("export_formats", ["csv", "excel"]),
        )
        exported = exporter.export(df, stem=f"{stem}_{class_dir.name}", stats_df=stats_df)
        generated_files.update({f"{class_name}_{k}": v for k, v in exported.items()})

        # Resumo de pixels por classe
        count = int((class_ids == class_idx).sum())
        summary_df = pd.DataFrame([{
            "class_index": class_idx,
            "class_name": class_name,
            "count_objects": labeling.n_accepted,
            "pixel_count": count,
            "area_pct": (100.0 * count / total_px) if total_px > 0 else 0.0,
            "prob_mean": float(class_prob.mean()),
        }])
        summary_csv = class_dir / f"{stem}_{class_dir.name}_summary.csv"
        summary_df.to_csv(summary_csv, index=False)

        # ── Passo 10: Visualizações da classe ─────────────────────────────────
        image_paths_for_class = {}
        if out_cfg.get("generate_overlay", True):
            viz = Visualizer(
                output_dir=class_dir,
                dpi=out_cfg.get("figure_dpi", 150),
                cmap_overlay=out_cfg.get("overlay_colormap", "jet"),
            )
            color = class_colors[class_idx % len(class_colors)]

            p_prob = viz.save_probability_map(class_prob, stem=f"{stem}_{class_dir.name}_prob", class_name=class_name)
            p_over = viz.save_overlay(image, class_ids == class_idx, stem=f"{stem}_{class_dir.name}_overlay", color=color, class_name=class_name)
            p_lab  = viz.save_label_overlay(image, labeling.label_map, stem=f"{stem}_{class_dir.name}_labels")
            
            image_paths_for_class = {
                "Probabilidade": p_prob,
                "Overlay": p_over,
                "Labels": p_lab
            }
            
            if not df.empty:
                image_paths_for_class["Histogramas"] = viz.save_histograms(df, stem=f"{stem}_{class_dir.name}_hist")

        # ── Passo 11: Relatório PDF Individual ───────────────────────────────
        if out_cfg.get("generate_report", True):
            reporter = PDFReporter(output_dir=class_dir)
            pdf_path = reporter.generate(
                image_name=f"{image_path.name} ({class_name})",
                porosity_pct=(100.0 * count / total_px),
                n_pores=labeling.n_accepted,
                df=df,
                stats_df=stats_df,
                image_paths=image_paths_for_class,
                config=cfg,
                stem=f"{stem}_{class_dir.name}_report",
            )
            generated_files[f"pdf_{class_name}"] = pdf_path

    logger.info(f"[{stem}] Pipeline multi-classe concluido. Resultados em: {output_dir}")

    logger.info(
        f"[{stem}] Pipeline concluido — "
        f"porosidade: {labeling.porosity_pct:.3f}%, "
        f"poros: {labeling.n_accepted}, "
        f"pasta: {output_dir}"
    )
    return generated_files


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Segmentacao de poros EDS/CBS/QEMSCAN com Ilastik + PARTISAN",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--ilp", type=Path,
        help="Caminho para o arquivo .ilp treinado no Ilastik",
    )
    p.add_argument(
        "--image", type=Path, default=None,
        help="Imagem unica para processar (.tif, .png, ...)",
    )
    p.add_argument(
        "--dir", type=Path, default=None,
        help="Diretorio com multiplas imagens para processar em batch",
    )
    p.add_argument(
        "--config", type=Path, default=Path("config.yaml"),
        help="Arquivo de configuracao YAML (default: config.yaml)",
    )
    p.add_argument(
        "--partisan", type=Path, default=None,
        help="Caminho para partisan.py (se nao estiver no PYTHONPATH)",
    )
    p.add_argument(
        "--pore-index", type=int, default=None,
        help="Forca o indice da classe poro (sobreescreve deteccao automatica)",
    )
    p.add_argument(
        "--inspect", type=Path, default=None, metavar="ILP",
        help="Apenas inspeciona o arquivo .ilp e imprime os metadados",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Sobreescreve output_dir do config.yaml",
    )
    return p


def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()

    # ── Modo inspecao ─────────────────────────────────────────────────────────
    if args.inspect:
        from core.ilp_reader import inspect_ilp
        inspect_ilp(args.inspect)
        return 0

    # ── Validacao de argumentos ───────────────────────────────────────────────
    if not args.ilp:
        parser.error("--ilp e obrigatorio (exceto com --inspect)")
    if not args.image and not args.dir:
        parser.error("Informe --image ou --dir")

    # ── Configuracao ──────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    logger = logging.getLogger("main")

    if args.output:
        cfg["output"]["output_dir"] = str(args.output)

    # ── Leitura do .ilp ───────────────────────────────────────────────────────
    from core.ilp_reader import ILPReader
    reader = ILPReader(args.ilp)
    meta   = reader.read(pore_class_index=args.pore_index)
    logger.info("\n" + meta.describe())

    # ── Inicializa segmentador (uma vez para todas as imagens) ────────────────
    from core.segmentor import IlastikSegmentor
    ilastik_cfg = cfg.get("ilastik", {})
    segmentor = IlastikSegmentor(
        ilp_path=args.ilp,
        pore_channel=meta.pore_class_index,
        ilastik_exe=ilastik_cfg.get("ilastik_exe", None),
    )

    # ── Lista de imagens ──────────────────────────────────────────────────────
    if args.image:
        images = [args.image]
    else:
        from preprocessing.loader import list_images
        images = list_images(args.dir)
        logger.info(f"Encontradas {len(images)} imagens em {args.dir}")

    if not images:
        logger.error("Nenhuma imagem encontrada.")
        return 1

    # ── Processamento em batch ────────────────────────────────────────────────
    t_start = time.time()
    errors  = []

    for i, img_path in enumerate(images):
        logger.info(f"\n{'='*60}")
        logger.info(f"Imagem {i+1}/{len(images)}: {img_path.name}")
        logger.info("="*60)
        try:
            run_pipeline(
                image_path=img_path,
                ilp_path=args.ilp,
                cfg=cfg,
                segmentor=segmentor,
                meta=meta,
                partisan_path=args.partisan,
            )
        except Exception as exc:
            logger.error(f"Erro ao processar '{img_path.name}': {exc}", exc_info=True)
            errors.append((img_path.name, str(exc)))

    # ── Resumo final ──────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"Processamento concluido em {elapsed:.1f}s")
    logger.info(f"  Sucesso : {len(images) - len(errors)}/{len(images)}")
    if errors:
        logger.warning(f"  Erros   : {len(errors)}")
        for name, msg in errors:
            logger.warning(f"    - {name}: {msg}")
    logger.info("="*60)

    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
