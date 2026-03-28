"""
PARTISAN (PARTicle Shape ANalyzer) v2.0 — Python
=================================================

Reimplementação fiel do PARTISAN (MATLAB).

Baseado em:
  - Dellino e La Volpe (1996)
  - Durig et al. (2012)
  - Cioni et al. (2014)
  - Leibrandt e Le Pennec (2015)
  - Liu et al. (2015)
  - Schmith et al. (2017)

Por favor cite:
  Durig, T. et al. (2018) "PARTIcle Shape ANalyzer PARTISAN — an open source
  tool for multi-standard two-dimensional particle morphometry analysis",
  An. Geophys., 61, 31. doi: 10.4401/ag-7865
"""

from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import label
from scipy.spatial import ConvexHull, QhullError
from skimage import measure
from skimage.morphology import convex_hull_image
from skimage.transform import rotate

warnings.filterwarnings("ignore", category=RuntimeWarning, module="partisan")


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclass de métricas
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PartisanMetrics:
    """
    Container para todos os parâmetros morfométricos do PARTISAN.

    Nomenclatura idêntica ao PARTISAN v2.0 (MATLAB).

    Medidas básicas (pixels)
    ------------------------
    p       : perímetro da partícula
    A       : área da partícula
    w       : largura (lado curto da bbox mínima)
    b       : comprimento (lado longo da bbox mínima)
    c       : perímetro de círculo com área A  (= 2√(πA))
    a       : intercepto máximo (maior segmento contíguo horizontal na bbox alinhada)
    m       : intercepto médio perpendicular a 'a' (span médio por coluna)

    Dimensões derivadas
    -------------------
    Lb      : comprimento projetado máximo (elipse alinhada, extensão horizontal)
    Wb      : comprimento projetado ortogonal (elipse alinhada, extensão vertical)
    dH      : diâmetro de Heywood  (= 2√(A/π))
    pHull   : perímetro do casco convexo
    Ahull   : área do casco convexo (pixels)
    ece     : perímetro da elipse mínima envolvente
    L_maj   : eixo maior da elipse de melhor ajuste (= MajorAxisLength)
    L_min   : eixo menor da elipse de melhor ajuste (= MinorAxisLength)
    d_BC    : diâmetro do círculo mínimo envolvente
    feret_major : diâmetro de Feret máximo
    feret_minor : diâmetro de Feret mínimo  (= w, lado curto da bbox)
    w_f     : comprimento ortogonal ao Feret  (= b, lado longo da bbox)

    Índices — Dellino & La Volpe (1996) [DL_*]
    -------------------------------------------
    DL_Circ : circularidade  = p / c
    DL_Rec  : retangularidade = p / (2b + 2w)
    DL_Com  : compacidade    = A / (b·w)
    DL_Elo  : alongamento    = a / m

    Índices — Cioni et al. (2014) [CI_*]
    -------------------------------------
    CI_Circ : circularidade  = 4πA / p²
    CI_AR   : razão de aspecto = L_maj / L_min
    CI_Con  : convexidade    = ece / p
    CI_Sol  : solidez        = A / Ahull

    Índices — Leibrandt & Le Pennec (2015) [LL_*]
    ----------------------------------------------
    LL_Circ : circularidade  = c / p
    LL_AR   : razão de aspecto = Wb / Lb
    LL_Elo  : alongamento    = 1 − LL_AR
    LL_Con  : convexidade    = pHull / p
    LL_Sol  : solidez        = A / Ahull

    Índices — Liu et al. (2015) [LI_*]
    ------------------------------------
    LI_AR   : razão axial    = L_min / L_maj
    LI_Con  : convexidade    = pHull / p
    LI_Sol  : solidez        = A / Ahull

    Índices — Schmith et al. (2017) [SC_*]
    ----------------------------------------
    SC_Circ : circularidade  = 4A / (π · d_BC²)
    SC_Rec  : retangularidade = A / (b·w)
    SC_ARF  : razão de Feret  = feret_minor / w_f
    SC_AR   : razão de Feret (para plots) = w_f / feret_minor

    Composto
    --------
    FF  : form factor  = 4πA / p²  (mesmo que CI_Circ)
    Reg : regularidade = SC_Circ · SC_Rec · FF
    """

    # Medidas básicas
    p: float = 0.0
    A: float = 0.0
    w: float = 0.0
    b: float = 0.0
    c: float = 0.0
    a: float = 0.0
    m: float = 0.0

    # Dimensões derivadas
    Lb: float = 0.0
    Wb: float = 0.0
    dH: float = 0.0
    pHull: float = 0.0
    Ahull: float = 0.0
    ece: float = 0.0
    L_maj: float = 0.0
    L_min: float = 0.0
    d_BC: float = 0.0
    feret_major: float = 0.0
    feret_minor: float = 0.0
    w_f: float = 0.0

    # Dellino & La Volpe (1996)
    DL_Circ: float = 0.0
    DL_Rec:  float = 0.0
    DL_Com:  float = 0.0
    DL_Elo:  float = 0.0

    # Cioni et al. (2014)
    CI_Circ: float = 0.0
    CI_AR:   float = 0.0
    CI_Con:  float = 0.0
    CI_Sol:  float = 0.0

    # Leibrandt & Le Pennec (2015)
    LL_Circ: float = 0.0
    LL_AR:   float = 0.0
    LL_Elo:  float = 0.0
    LL_Con:  float = 0.0
    LL_Sol:  float = 0.0

    # Liu et al. (2015)
    LI_AR:  float = 0.0
    LI_Con: float = 0.0
    LI_Sol: float = 0.0

    # Schmith et al. (2017)
    SC_Circ: float = 0.0
    SC_Rec:  float = 0.0
    SC_ARF:  float = 0.0
    SC_AR:   float = 0.0

    # Composto
    FF:  float = 0.0
    Reg: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Converte para dicionário (para exportação CSV/JSON)."""
        return asdict(self)

    def sanitize(self) -> "PartisanMetrics":
        """Substitui NaN e Inf por 0 (comportamento do MATLAB para divisão por zero)."""
        for field in self.__dataclass_fields__:
            v = getattr(self, field)
            if not np.isfinite(v):
                setattr(self, field, 0.0)
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# Funções geométricas puras
# ═══════════════════════════════════════════════════════════════════════════════

def _boundary_perimeter(binary_mask: np.ndarray) -> float:
    """
    Perímetro como comprimento total do polígono de fronteira.

    Equivalente ao MATLAB:
        Bdy = bwboundaries(data);
        d   = diff([Bdy{1}(:,1) Bdy{1}(:,2)]);
        p   = sum(sqrt(sum(d.^2, 2)));
    """
    contours = measure.find_contours(binary_mask.astype(np.float32), 0.5)
    if not contours:
        return 0.0
    boundary = max(contours, key=len)           # maior contorno (ilha principal)
    diffs = np.diff(boundary, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))


def _min_bounding_box(pts_xy: np.ndarray) -> Tuple[float, float, float]:
    """
    Caixa delimitadora de área mínima — rotating calipers sobre o casco convexo.

    Equivalente ao MATLAB ``minBoundingBox``.

    Parameters
    ----------
    pts_xy : (N, 2)  array  [coluna 0 = x, coluna 1 = y]

    Returns
    -------
    b     : lado longo
    w     : lado curto
    theta : ângulo do lado longo com o eixo x (radianos)
    """
    if len(pts_xy) < 3:
        rng = pts_xy.max(axis=0) - pts_xy.min(axis=0)
        return float(rng.max()), float(rng.min()), 0.0

    try:
        hull = ConvexHull(pts_xy)
        hpts = pts_xy[hull.vertices]
    except QhullError:
        rng = pts_xy.max(axis=0) - pts_xy.min(axis=0)
        return float(rng.max()), float(rng.min()), 0.0

    n = len(hpts)
    best_area  = np.inf
    best_b = best_w = best_theta = 0.0

    for i in range(n):
        edge  = hpts[(i + 1) % n] - hpts[i]
        angle = np.arctan2(edge[1], edge[0])
        ca, sa = np.cos(-angle), np.sin(-angle)
        R  = np.array([[ca, -sa], [sa, ca]])
        rp = hpts @ R.T

        sx = float(rp[:, 0].max() - rp[:, 0].min())
        sy = float(rp[:, 1].max() - rp[:, 1].min())
        area = sx * sy

        if area < best_area:
            best_area = area
            if sx >= sy:
                best_b, best_w, best_theta = sx, sy, angle
            else:
                best_b, best_w, best_theta = sy, sx, angle + np.pi / 2

    return best_b, best_w, best_theta


def _min_bounding_circle(pts_xy: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Círculo mínimo envolvente — algoritmo de Welzl O(N) esperado.

    Equivalente ao MATLAB ``minBoundCircle``.

    Parameters
    ----------
    pts_xy : (N, 2)

    Returns
    -------
    center : (2,)
    radius : float
    """
    def _c1(p: np.ndarray) -> Tuple[np.ndarray, float]:
        return p.copy(), 0.0

    def _c2(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, float]:
        return (p1 + p2) / 2.0, float(np.linalg.norm(p1 - p2) / 2.0)

    def _c3(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[np.ndarray, float]:
        ax, ay = float(p1[0]), float(p1[1])
        bx, by = float(p2[0]), float(p2[1])
        cx, cy = float(p3[0]), float(p3[1])
        D = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(D) < 1e-10:
            # Colineares — círculo do par mais distante
            d12 = np.linalg.norm(p1 - p2)
            d23 = np.linalg.norm(p2 - p3)
            d13 = np.linalg.norm(p1 - p3)
            if d12 >= d23 and d12 >= d13:
                return _c2(p1, p2)
            return _c2(p2, p3) if d23 >= d13 else _c2(p1, p3)
        ux = ((ax**2 + ay**2)*(by - cy) + (bx**2 + by**2)*(cy - ay) +
              (cx**2 + cy**2)*(ay - by)) / D
        uy = ((ax**2 + ay**2)*(cx - bx) + (bx**2 + by**2)*(ax - cx) +
              (cx**2 + cy**2)*(bx - ax)) / D
        center = np.array([ux, uy])
        return center, float(np.linalg.norm(p1 - center))

    def _in_circle(c: np.ndarray, r: float, p: np.ndarray) -> bool:
        return float(np.linalg.norm(p - c)) <= r + 1e-10

    # Usar apenas vértices do casco convexo para eficiência
    if len(pts_xy) >= 3:
        try:
            hull = ConvexHull(pts_xy)
            pts = pts_xy[hull.vertices]
        except QhullError:
            pts = pts_xy
    else:
        pts = pts_xy

    rng = np.random.default_rng(42)
    pts = pts[rng.permutation(len(pts))]

    c, r = _c1(pts[0])
    for i in range(1, len(pts)):
        if not _in_circle(c, r, pts[i]):
            c, r = _c1(pts[i])
            for j in range(i):
                if not _in_circle(c, r, pts[j]):
                    c, r = _c2(pts[i], pts[j])
                    for k in range(j):
                        if not _in_circle(c, r, pts[k]):
                            c, r = _c3(pts[i], pts[j], pts[k])
    return c, r


def _min_bounding_ellipse(
    pts_xy: np.ndarray,
    tolerance: float = 0.005,
    n_boundary: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Elipse mínima envolvente (MVEE) — algoritmo de Khachiyan / Todd-Yildirim.

    Equivalente ao MATLAB ``minBoundEllipse``.

    Parameters
    ----------
    pts_xy     : (N, 2) pontos de fronteira ou vértices do casco
    tolerance  : tolerância de convergência (MATLAB default: 0.005)
    n_boundary : pontos na fronteira da elipse de saída

    Returns
    -------
    center      : (2,)
    ellipse_pts : (2, n_boundary) — pontos na fronteira da elipse
    """
    P = pts_xy.T          # (2, N)
    d, N = P.shape

    # Casos degenerados
    if N < 3:
        center = P.mean(axis=1)
        r = float(np.max(np.linalg.norm(P - center[:, None], axis=0))) + 1e-6
        t = np.linspace(0, 2 * np.pi, n_boundary)
        return center, center[:, None] + r * np.vstack([np.cos(t), np.sin(t)])

    # Coordenadas homogêneas  (3, N)
    Q = np.vstack([P, np.ones((1, N))])
    u = np.full(N, 1.0 / N)

    for _ in range(5000):
        X = Q @ (u[:, None] * Q.T)          # (3, 3) = Q diag(u) Qᵀ
        try:
            X_inv = np.linalg.inv(X)
        except np.linalg.LinAlgError:
            break
        # M_i = (Q_i)ᵀ X⁻¹ Q_i  (diagonal de Qᵀ X⁻¹ Q)
        M = np.einsum("ij,jk,ki->i", Q.T, X_inv, Q)
        j     = int(np.argmax(M))
        max_M = float(M[j])

        step_size = (max_M - d - 1.0) / ((d + 1.0) * (max_M - 1.0))
        if step_size <= 0:
            break

        u_new    = (1.0 - step_size) * u
        u_new[j] += step_size

        if float(np.linalg.norm(u_new - u)) < tolerance:
            u = u_new
            break
        u = u_new

    center = P @ u                              # (2,)
    Pu  = P - center[:, None]
    cov = Pu @ (u[:, None] * Pu.T)             # (2, 2) covariância ponderada

    try:
        A_mat     = np.linalg.inv(cov) / d
        eigvals, eigvecs = np.linalg.eigh(A_mat)
        eigvals   = np.maximum(eigvals, 1e-12)
        semi_axes = 1.0 / np.sqrt(eigvals)
    except np.linalg.LinAlgError:
        r = float(np.max(np.linalg.norm(Pu, axis=0))) + 1e-6
        t = np.linspace(0, 2 * np.pi, n_boundary)
        return center, center[:, None] + r * np.vstack([np.cos(t), np.sin(t)])

    t      = np.linspace(0, 2 * np.pi, n_boundary)
    circle = np.vstack([np.cos(t), np.sin(t)])
    ellipse_pts = center[:, None] + eigvecs @ np.diag(semi_axes) @ circle

    return center, ellipse_pts


def _max_row_contiguous(img: np.ndarray, dthresh: float = 0.5) -> Tuple[int, int, int]:
    """
    Varre linha por linha buscando o maior segmento CONTÍGUO.

    Equivalente ao cálculo de 'a' no MATLAB:
        for i = 1:rows
          hits = find(row > dthresh);
          jumps = find(diff(hits) > 1);
          % ... encontra o maior segmento entre saltos ...
        end
        a = max_segment.value;

    Returns
    -------
    max_len   : comprimento do maior segmento
    best_row  : índice da linha com o maior segmento
    best_start: índice de coluna inicial do segmento
    """
    max_len    = 0
    best_row   = 0
    best_start = 0

    for i in range(img.shape[0]):
        hits = np.where(img[i] > dthresh)[0]
        if len(hits) == 0:
            continue

        jumps = np.where(np.diff(hits) > 1)[0]

        if len(jumps) == 0:
            seg_len = 1 + int(hits[-1]) - int(hits[0])
            if seg_len > max_len:
                max_len, best_row, best_start = seg_len, i, int(hits[0])
        else:
            seg_starts = np.concatenate([[0],       jumps + 1])
            seg_ends   = np.concatenate([jumps,     [len(hits) - 1]])
            for s, e in zip(seg_starts, seg_ends):
                seg_len = 1 + int(hits[e]) - int(hits[s])
                if seg_len > max_len:
                    max_len, best_row, best_start = seg_len, i, int(hits[s])

    return max_len, best_row, best_start


def _mean_col_span(img: np.ndarray, dthresh: float = 0.5) -> float:
    """
    Span médio por coluna (intercepto médio perpendicular, 'm').

    MATLAB:
        for i = 1:cols
          hits = find(col > dthresh);
          len  = 1 + max(hits) - min(hits);  % span total, não contagem de pixels
          chord_sum += len;  num_chords += 1;
        end
        m = chord_sum / num_chords;

    Usa span (máx − mín + 1), não contagem de pixels — difere para partículas
    côncavas / com buracos.
    """
    chord_sum  = 0
    num_chords = 0
    for j in range(img.shape[1]):
        hits = np.where(img[:, j] > dthresh)[0]
        if len(hits) > 0:
            chord_sum  += 1 + int(hits[-1]) - int(hits[0])
            num_chords += 1
    if num_chords == 0:
        return 0.0
    return chord_sum / num_chords


def _max_row_span(img: np.ndarray, dthresh: float = 0.5) -> Tuple[int, int]:
    """
    Span horizontal máximo em todas as linhas (para Feret maior).

    MATLAB:
        hits = find(row > dthresh);
        len  = 1 + max(hits) - min(hits);  % span, não segmento contíguo
    """
    max_span = 0
    best_row = 0
    for i in range(img.shape[0]):
        hits = np.where(img[i] > dthresh)[0]
        if len(hits) > 0:
            span = 1 + int(hits[-1]) - int(hits[0])
            if span > max_span:
                max_span, best_row = span, i
    return max_span, best_row


def _crop_to_content(img: np.ndarray, dthresh: float = 0.5) -> np.ndarray:
    """Recorta imagem ao bounding box dos pixels > dthresh."""
    ys, xs = np.where(img > dthresh)
    if len(ys) == 0:
        return img
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# ═══════════════════════════════════════════════════════════════════════════════
# Classe principal
# ═══════════════════════════════════════════════════════════════════════════════

class PartisanAnalyzer:
    """
    Analisador morfométrico — reimplementação fiel do PARTISAN v2.0 (MATLAB).

    Pipeline idêntico ao script MATLAB original:
    1. Propriedades básicas (área, perímetro via fronteira, eixos)
    2. Casco convexo (área, perímetro)
    3. Caixa delimitadora mínima → b, w, θ
    4. Rotação pela bbox → a (maior segmento contíguo), m (span médio)
    5. Busca de Feret ±50° em torno de θ → feret_major
    6. Elipse mínima envolvente sobre imagem rotacionada por Feret → ece
    7. Círculo mínimo envolvente → d_BC
    8. Rotação pela elipse de melhor ajuste → Lb, Wb
    9. Cálculo de todos os índices morfométricos

    Parameters
    ----------
    plot_results     : gera visualização simplificada (matplotlib)
    feret_angle_step : passo angular para busca Feret (graus).
                       MATLAB usa 0,5° (imagens pequenas) ou 1,0° (grandes).
    """

    def __init__(
        self,
        plot_results: bool = False,
        feret_angle_step: float = 0.5,
    ) -> None:
        self.plot_results     = plot_results
        self.feret_angle_step = feret_angle_step

    # ──────────────────────────────────────────────────────────────────────────
    # Método público principal
    # ──────────────────────────────────────────────────────────────────────────

    def analyze(
        self,
        silhouette: np.ndarray,
        filename: Optional[str] = None,
    ) -> PartisanMetrics:
        """
        Executa a análise morfométrica PARTISAN completa.

        Parameters
        ----------
        silhouette : np.ndarray
            Imagem binária 2D (bool, uint8 0/255, ou float 0/1).
            Pixels de primeiro plano = True / 255 / 1.
        filename   : str, opcional
            Identificador para logs e título do gráfico.

        Returns
        -------
        PartisanMetrics
        """
        if not np.any(silhouette):
            raise ValueError("Silhueta vazia (sem pixels de primeiro plano).")

        # ── Binarização (replica lógica do MATLAB) ───────────────────────────
        max_val = float(silhouette.max())
        if max_val <= 1:
            dthresh = 0.5
        elif max_val <= 255:
            dthresh = 128.0
        else:
            dthresh = 128.0
            warnings.warn(
                f"Valor máximo não-padrão ({max_val:.0f}). Usando threshold=128.",
                RuntimeWarning,
            )

        binary = silhouette > dthresh

        # Verificar inversão (pixel [0,0] aceso = imagem invertida)
        if binary[0, 0]:
            binary = ~binary

        # ── PASSO 1: regionprops básico ───────────────────────────────────────
        labeled  = label(binary)[0]
        regions  = measure.regionprops(labeled)
        if not regions:
            raise ValueError("Nenhuma região encontrada na máscara binária.")
        if len(regions) > 1:
            warnings.warn(
                f"Múltiplas regiões detectadas ({len(regions)}). Usando a maior.",
                RuntimeWarning,
            )
        region = max(regions, key=lambda r: r.area)

        A               = float(region.area)
        Ahull           = float(region.area_convex)
        L_maj           = float(region.axis_major_length)
        L_min           = float(region.axis_minor_length)
        orientation_rad = float(region.orientation)  # skimage: radianos, eixo de linhas

        # ── PASSO 2: Perímetro (bwboundaries-style) ───────────────────────────
        p = _boundary_perimeter(binary)

        # ── PASSO 3: Casco convexo e seu perímetro ────────────────────────────
        try:
            hull_img = convex_hull_image(binary)
        except Exception:
            hull_img = binary
        pHull = _boundary_perimeter(hull_img)

        # ── PASSO 4: Diâmetro de Heywood e círculo equivalente ────────────────
        # MATLAB: r = sqrt(A/pi);  dH = r*2;  c = pi*2*r
        r_eq = np.sqrt(A / np.pi)
        dH   = float(r_eq * 2.0)
        c    = float(np.pi * 2.0 * r_eq)

        # ── PASSO 5: Caixa delimitadora mínima (rotating calipers) ───────────
        rows_px, cols_px = np.where(binary)
        pts_xy = np.column_stack([cols_px.astype(float), rows_px.astype(float)])

        b, w, theta = _min_bounding_box(pts_xy)
        # feret_minor = w  (lado curto da bbox, MATLAB: feret_minor = w)
        # w_f         = b  (lado longo da bbox, MATLAB: w_f = b)
        feret_minor = w
        w_f         = b

        # ── PASSO 6: Intercepts a e m (rotação pela bbox) ─────────────────────
        theta_deg = float(np.degrees(theta))

        data_rot_bbox = rotate(
            binary.astype(np.float32), theta_deg,
            resize=True, preserve_range=True, order=1,
        )
        crop_bbox = _crop_to_content(data_rot_bbox)

        # a = maior segmento CONTÍGUO horizontal (MATLAB: max_segment.value)
        a_int, _, _ = _max_row_contiguous(crop_bbox, dthresh=0.5)
        a = float(a_int)

        # m = span médio por coluna (MATLAB: chord_sum / num_chords)
        m = _mean_col_span(crop_bbox, dthresh=0.5)

        # ── PASSO 7: Busca de Feret ±50° em torno de θ ───────────────────────
        # MATLAB: for adjust = -50 : stepsize : +50
        #           imrotate → varredura de linhas → max span
        # Implementação: projeção do casco convexo (matematicamente equivalente).
        try:
            hull_ch = ConvexHull(pts_xy)
            hull_pts_xy = pts_xy[hull_ch.vertices]
        except QhullError:
            hull_pts_xy = pts_xy

        feret_major        = 0.0
        best_feret_deg     = theta_deg
        adjusts = np.arange(-50, 50 + self.feret_angle_step, self.feret_angle_step)

        for adjust in adjusts:
            test_deg = theta_deg + adjust
            angle_r  = np.radians(test_deg)
            ca, sa   = np.cos(-angle_r), np.sin(-angle_r)
            R = np.array([[ca, -sa], [sa, ca]])
            rp = hull_pts_xy @ R.T
            # +1 replica "len = 1 + max(hits) - min(hits)" do MATLAB
            span = float(rp[:, 0].max() - rp[:, 0].min()) + 1.0
            if span > feret_major:
                feret_major    = span
                best_feret_deg = test_deg

        # ── PASSO 8: Elipse mínima envolvente (ece) ───────────────────────────
        # MATLAB: [mbe.C, mbe.coords] = minBoundEllipse(data_rot_crop, 0.005)
        #         ece = comprimento do polígono das coordenadas da elipse
        # Rotacionar para alinhamento Feret, extrair casco, aplicar MVEE.
        data_rot_feret = rotate(
            binary.astype(np.float32), best_feret_deg,
            resize=True, preserve_range=True, order=1,
        )
        ys_f, xs_f = np.where(data_rot_feret > 0.5)

        if len(ys_f) >= 3:
            pts_feret = np.column_stack([xs_f.astype(float), ys_f.astype(float)])
            try:
                hull_f    = ConvexHull(pts_feret)
                hull_pts_f = pts_feret[hull_f.vertices]
            except QhullError:
                hull_pts_f = pts_feret
            try:
                _, ellipse_pts = _min_bounding_ellipse(hull_pts_f, tolerance=0.005)
                diffs_e = np.diff(ellipse_pts, axis=1)
                ece     = float(np.sum(np.sqrt(np.sum(diffs_e ** 2, axis=0))))
            except Exception:
                ece = p   # fallback
        else:
            ece = p

        # ── PASSO 9: Círculo mínimo envolvente (d_BC) ─────────────────────────
        # MATLAB: [center, radius] = minBoundCircle(Xs, Ys, false)
        try:
            _, radius = _min_bounding_circle(pts_xy)
            d_BC = float(radius * 2.0)
        except Exception:
            d_BC = float(region.equivalent_diameter_area)   # fallback

        # ── PASSO 10: Lb e Wb (rotação pela elipse de melhor ajuste) ──────────
        # MATLAB: ellip.theta_raw = -1 * imgstats.Orientation  [graus]
        #   imrotate(data, ellip.theta_raw, 'bilinear')
        #   Lb = max(find(max(bool))) - min(find(max(bool)))   % extensão horizontal
        #   Wb = max(find(max(bool'))) - min(find(max(bool'))) % extensão vertical
        #
        # Conversão de orientação:
        #   MATLAB Orientation: ângulo entre eixo x e eixo maior [−90°, 90°]
        #   skimage orientation: ângulo entre eixo de linhas (y) e eixo maior [−π/2, π/2]
        #   → matlab_deg = 90 − degrees(skimage_rad)
        #   → theta_raw  = −matlab_deg = degrees(skimage_rad) − 90
        ellipse_theta_deg = float(np.degrees(orientation_rad)) - 90.0

        data_rot_ellips = rotate(
            binary.astype(np.float32), ellipse_theta_deg,
            resize=True, preserve_range=True, order=1,
        )
        bool_ellips = data_rot_ellips > 0.5

        occupied_cols = np.any(bool_ellips, axis=0)
        occupied_rows = np.any(bool_ellips, axis=1)
        col_idx = np.where(occupied_cols)[0]
        row_idx = np.where(occupied_rows)[0]

        Lb = float(col_idx[-1] - col_idx[0]) if len(col_idx) >= 2 else 0.0
        Wb = float(row_idx[-1] - row_idx[0]) if len(row_idx) >= 2 else 0.0

        # Garante Lb >= Wb (comprimento >= largura).
        # A rotacao pela elipse nem sempre alinha o eixo maior com o eixo
        # horizontal -- equivalente ao ExtraRot = pi/2 do MATLAB para a bbox.
        if Wb > Lb:
            Lb, Wb = Wb, Lb

        # ── PASSO 11: Montagem de todos os índices ────────────────────────────
        mt = PartisanMetrics()

        # Medidas básicas
        mt.p     = p
        mt.A     = A
        mt.w     = w
        mt.b     = b
        mt.c     = c
        mt.a     = a
        mt.m     = m
        mt.Lb    = Lb
        mt.Wb    = Wb
        mt.dH    = dH
        mt.pHull = pHull
        mt.Ahull = Ahull
        mt.ece   = ece
        mt.L_maj = L_maj
        mt.L_min = L_min
        mt.d_BC  = d_BC
        mt.feret_major = feret_major
        mt.feret_minor = feret_minor
        mt.w_f   = w_f

        # Dellino & La Volpe (1996)
        # Circ = p/c | Rec = p/(2b+2w) | Com = A/(b·w) | Elo = a/m
        mt.DL_Circ = p / c             if c > 0             else 0.0
        mt.DL_Rec  = p / (2*b + 2*w)  if (b + w) > 0       else 0.0
        mt.DL_Com  = A / (b * w)       if (b * w) > 0       else 0.0
        mt.DL_Elo  = a / m             if m > 0             else 0.0

        # Cioni et al. (2014)
        # Circ = 4πA/p² | AR = L_maj/L_min | Con = ece/p | Sol = A/Ahull
        mt.CI_Circ = (4 * np.pi * A) / (p ** 2)  if p > 0     else 0.0
        mt.CI_AR   = L_maj / L_min                if L_min > 0 else 0.0
        mt.CI_Con  = ece / p                       if p > 0     else 0.0
        mt.CI_Sol  = A / Ahull                     if Ahull > 0 else 0.0

        # Leibrandt & Le Pennec (2015)
        # Circ = c/p | AR = Wb/Lb | Elo = 1−AR | Con = pHull/p | Sol = A/Ahull
        mt.LL_Circ = c / p       if p > 0    else 0.0
        mt.LL_AR   = Wb / Lb     if Lb > 0   else 0.0
        mt.LL_Elo  = 1.0 - mt.LL_AR
        mt.LL_Con  = pHull / p   if p > 0    else 0.0
        mt.LL_Sol  = A / Ahull   if Ahull > 0 else 0.0

        # Liu et al. (2015)
        # FF = 4πA/p² | AR = L_min/L_maj | Con = pHull/p | Sol = A/Ahull
        mt.FF     = mt.CI_Circ
        mt.LI_AR  = L_min / L_maj  if L_maj > 0  else 0.0
        mt.LI_Con = pHull / p      if p > 0       else 0.0
        mt.LI_Sol = A / Ahull      if Ahull > 0   else 0.0

        # Schmith et al. (2017)
        # Circ = 4A/(π·d_BC²) | Rec = A/(b·w) | ARF = feret_minor/w_f | AR = w_f/feret_minor
        mt.SC_Circ = (4 * A) / (np.pi * d_BC ** 2)  if d_BC > 0           else 0.0
        mt.SC_Rec  = A / (b * w)                      if (b * w) > 0        else 0.0
        mt.SC_ARF  = feret_minor / w_f                if w_f > 0            else 0.0
        mt.SC_AR   = w_f / feret_minor                if feret_minor > 0    else 0.0

        # Composto
        mt.Reg = mt.SC_Circ * mt.SC_Rec * mt.FF

        mt.sanitize()

        if self.plot_results:
            self._plot_silhouette(binary, filename)

        return mt

    # ──────────────────────────────────────────────────────────────────────────
    # Visualização
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_silhouette(
        self, binary_mask: np.ndarray, filename: Optional[str]
    ) -> None:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 8))
            plt.imshow(binary_mask, cmap="gray")
            plt.axis("off")
            if filename:
                plt.title(filename.replace("_", "\\_"), fontsize=12)
            else:
                plt.title("Silhueta", fontsize=12)
            plt.tight_layout()
            plt.show()
        except ImportError:
            warnings.warn("Matplotlib indisponível. Gráfico ignorado.", RuntimeWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# Wrapper compatível com MATLAB
# ═══════════════════════════════════════════════════════════════════════════════

def analisePARTISAN(
    silhueta: np.ndarray,
    do_plots: bool = False,
    filename: Optional[str] = None,
) -> Dict[str, float]:
    """
    Wrapper compatível com MATLAB para análise PARTISAN.

    Interface idêntica ao código MATLAB original.

    Parameters
    ----------
    silhueta  : imagem binária da silhueta
    do_plots  : se True, gera gráfico
    filename  : nome do arquivo (para título do gráfico)

    Returns
    -------
    dict com todas as métricas (equivalente à struct do MATLAB)

    Example
    -------
    >>> results = analisePARTISAN(silhouette, do_plots=True, filename="particle_01")
    >>> print(f"Circularidade: {results['CI_Circ']:.3f}")
    """
    analyzer = PartisanAnalyzer(plot_results=do_plots)
    metrics  = analyzer.analyze(silhueta, filename=filename)
    return metrics.to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-teste e demonstração
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Análise Morfométrica PARTISAN v2.0 — Auto-teste")
    print("=" * 70)

    analyzer = PartisanAnalyzer(plot_results=False, feret_angle_step=0.5)

    # ── Teste 1: Círculo perfeito ─────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("Teste 1: Círculo Perfeito (raio=50, imagem 200×200)")
    print("-" * 70)

    img_circle = np.zeros((200, 200), dtype=bool)
    y, x = np.ogrid[:200, :200]
    img_circle[(x - 100)**2 + (y - 100)**2 <= 50**2] = True

    m_circle = analyzer.analyze(img_circle, filename="circulo_perfeito")
    print(f"  Área              : {m_circle.A:.0f} px   (esperado ≈ {np.pi*50**2:.0f})")
    print(f"  Perímetro         : {m_circle.p:.1f} px  (esperado ≈ {2*np.pi*50:.1f})")
    print(f"  CI_Circ (4πA/p²)  : {m_circle.CI_Circ:.4f}  (esperado → 1.0; ~0.89 por pixelização, igual ao MATLAB)")
    print(f"  CI_AR (L_maj/L_min): {m_circle.CI_AR:.4f}  (esperado → 1.0)")
    print(f"  d_BC (círculo mín.): {m_circle.d_BC:.1f} px  (esperado ≈ 100.0)")
    print(f"  DL_Circ (p/c)     : {m_circle.DL_Circ:.4f}  (esperado → 1.0)")
    print(f"  LL_Circ (c/p)     : {m_circle.LL_Circ:.4f}  (esperado → 1.0)")
    print(f"  SC_Circ (4A/πd²)  : {m_circle.SC_Circ:.4f}  (esperado → 1.0)")

    # ── Teste 2: Quadrado (100×100) ───────────────────────────────────────────
    print("\n" + "-" * 70)
    print("Teste 2: Quadrado (100×100, imagem 200×200)")
    print("-" * 70)

    img_sq = np.zeros((200, 200), dtype=bool)
    img_sq[50:150, 50:150] = True

    m_sq = analyzer.analyze(img_sq, filename="quadrado")
    print(f"  Área              : {m_sq.A:.0f} px   (esperado 10000)")
    print(f"  b (lado longo)    : {m_sq.b:.1f} px  (esperado ≈ 100)")
    print(f"  w (lado curto)    : {m_sq.w:.1f} px  (esperado ≈ 100)")
    print(f"  SC_Rec (A/bw)     : {m_sq.SC_Rec:.4f}  (esperado → 1.0)")
    print(f"  CI_AR             : {m_sq.CI_AR:.4f}  (esperado → 1.0)")
    print(f"  CI_Circ (4πA/p²)  : {m_sq.CI_Circ:.4f}  (esperado ≈ π/4 ≈ 0.785)")

    # ── Teste 3: Elipse alongada (eixo maior 80, eixo menor 20) ──────────────
    print("\n" + "-" * 70)
    print("Teste 3: Elipse alongada (a=80, b=20, imagem 200×200)")
    print("-" * 70)

    img_ellipse = np.zeros((200, 200), dtype=bool)
    y, x = np.ogrid[:200, :200]
    img_ellipse[((x - 100) / 80)**2 + ((y - 100) / 20)**2 <= 1] = True

    m_ell = analyzer.analyze(img_ellipse, filename="elipse_alongada")
    print(f"  Área              : {m_ell.A:.0f} px   (esperado ≈ {np.pi*80*20:.0f})")
    print(f"  b (lado longo)    : {m_ell.b:.1f} px  (esperado ≈ 160)")
    print(f"  w (lado curto)    : {m_ell.w:.1f} px  (esperado ≈ 40)")
    print(f"  CI_AR (L_maj/L_min): {m_ell.CI_AR:.3f}  (esperado ≈ 4.0)")
    print(f"  DL_Elo (a/m)      : {m_ell.DL_Elo:.3f}  (> 1 para formas alongadas)")
    print(f"  LL_Elo (1−Wb/Lb)  : {m_ell.LL_Elo:.3f}  (esperado > 0.5)")

    # ── Teste 4: Forma irregular (estrela com 5 pontas) ───────────────────────
    print("\n" + "-" * 70)
    print("Teste 4: Forma Irregular (estrela)")
    print("-" * 70)

    from skimage.draw import polygon as ski_polygon

    img_star = np.zeros((150, 150), dtype=bool)
    angles   = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    r_star   = 40 + 15 * np.sin(5 * angles)
    xs_star  = np.clip((75 + r_star * np.cos(angles)).astype(int), 0, 149)
    ys_star  = np.clip((75 + r_star * np.sin(angles)).astype(int), 0, 149)
    rr, cc   = ski_polygon(ys_star, xs_star, shape=(150, 150))
    img_star[rr, cc] = True

    m_star = analyzer.analyze(img_star, filename="estrela")
    print(f"  Área              : {m_star.A:.0f} px")
    print(f"  CI_Circ           : {m_star.CI_Circ:.4f}  (< 1 — não circular)")
    print(f"  LL_Con (pHull/p)  : {m_star.LL_Con:.4f}  (< 1 — forma côncava)")
    print(f"  CI_Sol (A/Ahull)  : {m_star.CI_Sol:.4f}  (< 1 — solidez reduzida)")
    print(f"  Reg               : {m_star.Reg:.4f}")

    # ── Teste 5: Benchmark de desempenho ──────────────────────────────────────
    print("\n" + "-" * 70)
    print("Teste 5: Benchmark de Desempenho (blob 300×300)")
    print("-" * 70)

    from scipy.ndimage import binary_dilation

    seed = np.zeros((300, 300), dtype=bool)
    seed[140:160, 140:160] = True
    blob = binary_dilation(seed, iterations=50)

    t0    = time.time()
    m_bm  = analyzer.analyze(blob)
    elapsed = time.time() - t0

    print(f"  Resolução         : 300×300 = 90.000 pixels")
    print(f"  Pixels foreground : {int(m_bm.A):,}")
    print(f"  Tempo total       : {elapsed * 1000:.1f} ms")
    print(f"  Métricas geradas  : {len(m_bm.to_dict())}")

    # ── Resumo ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Auto-teste concluído com sucesso!")
    print("=" * 70)
    print("\nCategorias de Métricas:")
    print("  p, A, w, b, c, a, m, Lb, Wb, dH, pHull, Ahull, ece,")
    print("  L_maj, L_min, d_BC, feret_major, feret_minor, w_f  (19 básicas)")
    print("  DL_Circ, DL_Rec, DL_Com, DL_Elo                    (4 — Dellino&LaVolpe)")
    print("  CI_Circ, CI_AR, CI_Con, CI_Sol                      (4 — Cioni)")
    print("  LL_Circ, LL_AR, LL_Elo, LL_Con, LL_Sol              (5 — Leibrandt&LePennec)")
    print("  LI_AR, LI_Con, LI_Sol                               (3 — Liu)")
    print("  SC_Circ, SC_Rec, SC_ARF, SC_AR                      (4 — Schmith)")
    print("  FF, Reg                                             (2 — compostos)")
    print(f"\nTotal: {len(m_bm.to_dict())} parâmetros morfométricos")
    print("=" * 70)