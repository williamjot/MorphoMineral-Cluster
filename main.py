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
    validator = ImageValidator(expected_channels=None)  # flexivel: aceita gray e multi
    validator.validate(image, name=stem)

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

    # ── Passo 5: Threshold ───────────────────────────────────────────────────
    logger.info(f"[{stem}] Aplicando threshold...")
    binary = threshold_probability_map(
        prob_map,
        method=post_cfg.get("threshold_method", "otsu"),
        fixed_value=post_cfg.get("threshold_value", 0.5),
    )

    # ── Passo 6: Morfologia ──────────────────────────────────────────────────
    logger.info(f"[{stem}] Aplicando morfologia...")
    binary = apply_morphology(
        binary,
        opening_radius=post_cfg.get("morphology_opening_radius", 2),
        closing_radius=post_cfg.get("morphology_closing_radius", 2),
        fill_holes=post_cfg.get("morphology_fill_holes", True),
    )

    # ── Passo 7: Labeling ────────────────────────────────────────────────────
    logger.info(f"[{stem}] Identificando poros individuais...")
    labeling = label_pores(
        binary,
        min_area_px=post_cfg.get("min_pore_area_px", 50),
        max_area_px=post_cfg.get("max_pore_area_px", 0),
    )

    # ── Passo 8: PARTISAN ────────────────────────────────────────────────────
    logger.info(f"[{stem}] Executando analise PARTISAN ({labeling.n_accepted} poros)...")
    df = run_partisan(
        labeling_result=labeling,
        image_name=stem,
        partisan_path=partisan_path,
        show_progress=True,
    )
    stats_df = summary_statistics(df) if not df.empty else None

    generated_files = {}

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

    # ── Passo 9: Exportacao de tabelas ───────────────────────────────────────
    logger.info(f"[{stem}] Exportando resultados...")
    pore_dir = class_output_dirs.get(meta.pore_class_index, output_dir / "Poro")
    pore_dir.mkdir(parents=True, exist_ok=True)
    exporter = ResultExporter(
        output_dir=pore_dir,
        formats=out_cfg.get("export_formats", ["csv", "excel"]),
    )
    exported = exporter.export(df, stem=stem, stats_df=stats_df)
    generated_files.update(exported)

    # Exporta resumo por classe (poro/matriz/particula/etc.) em cada pasta de classe
    total_px = class_ids.size
    for class_idx in range(n_classes):
        class_name = class_names[class_idx]
        class_dir = class_output_dirs[class_idx]
        class_prob = prob_map_all[:, :, class_idx]
        class_mask = class_ids == class_idx
        count = int((class_ids == class_idx).sum())

        class_df = pd.DataFrame([{
            "class_index": class_idx,
            "class_name": class_name,
            "pixel_count": count,
            "pixel_pct": (100.0 * count / total_px) if total_px > 0 else 0.0,
            "prob_mean": float(class_prob.mean()),
            "prob_std": float(class_prob.std()),
            "prob_min": float(class_prob.min()),
            "prob_max": float(class_prob.max()),
        }])

        class_csv = class_dir / f"{stem}_{class_dir.name}_summary.csv"
        class_df.to_csv(class_csv, index=False, float_format="%.6f")
        generated_files[f"csv_class_{class_idx}"] = class_csv
        logger.info(f"CSV salvo: {class_csv}")

    # ── Passo 10: Visualizacoes ──────────────────────────────────────────────
    image_paths_for_report = {}
    if out_cfg.get("generate_overlay", True):
        logger.info(f"[{stem}] Gerando visualizacoes...")
        class_colors = [
            (1.0, 0.15, 0.15),
            (0.20, 0.70, 0.25),
            (0.20, 0.45, 0.95),
            (0.95, 0.65, 0.20),
        ]

        for class_idx in range(n_classes):
            class_name = class_names[class_idx]
            class_dir = class_output_dirs[class_idx]
            class_prob = prob_map_all[:, :, class_idx]
            class_mask = class_ids == class_idx
            viz_class = Visualizer(
                output_dir=class_dir,
                dpi=out_cfg.get("figure_dpi", 150),
                cmap_overlay=out_cfg.get("overlay_colormap", "jet"),
            )
            color = class_colors[class_idx % len(class_colors)]

            p_prob = viz_class.save_probability_map(
                class_prob,
                stem=f"{stem}_{class_dir.name}_prob",
                class_name=class_name,
            )
            p_overlay = viz_class.save_overlay(
                image,
                class_mask,
                stem=f"{stem}_{class_dir.name}_overlay",
                color=color,
                class_name=class_name,
            )

            if class_idx == meta.pore_class_index:
                image_paths_for_report["Mapa de Probabilidade (Poro)"] = p_prob
                image_paths_for_report["Overlay (Poro)"] = p_overlay

        viz_pore = Visualizer(
            output_dir=pore_dir,
            dpi=out_cfg.get("figure_dpi", 150),
            cmap_overlay=out_cfg.get("overlay_colormap", "jet"),
        )
        p = viz_pore.save_label_overlay(image, labeling.label_map, stem=f"{stem}_labels")
        image_paths_for_report["Poros Individuais"] = p

        if not df.empty:
            p = viz_pore.save_histograms(df, stem=f"{stem}_histograms")
            image_paths_for_report["Histogramas PARTISAN"] = p

            p = viz_pore.save_scatter(df, stem=f"{stem}_scatter")
            image_paths_for_report["Scatter CI_Circ vs CI_AR"] = p

    # ── Passo 11: Relatorio PDF ──────────────────────────────────────────────
    if out_cfg.get("generate_report", True):
        logger.info(f"[{stem}] Gerando relatorio PDF...")
        reporter = PDFReporter(output_dir=output_dir)
        pdf_path = reporter.generate(
            image_name=image_path.name,
            porosity_pct=labeling.porosity_pct,
            n_pores=labeling.n_accepted,
            df=df,
            stats_df=stats_df,
            image_paths=image_paths_for_report,
            config=cfg,
            stem=f"{stem}_report",
        )
        generated_files["pdf"] = pdf_path

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
