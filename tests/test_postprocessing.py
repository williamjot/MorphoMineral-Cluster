"""
tests/test_postprocessing.py
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from postprocessing.thresholder import threshold_probability_map
from postprocessing.morphology  import apply_morphology
from postprocessing.labeler     import label_pores, extract_pore_crop


def _circle_mask(r=30, size=100):
    img = np.zeros((size, size), dtype=bool)
    y, x = np.ogrid[:size, :size]
    img[(x - size//2)**2 + (y - size//2)**2 <= r**2] = True
    return img


class TestThresholder:
    def test_otsu(self):
        prob = np.zeros((100, 100), dtype=np.float32)
        prob[30:70, 30:70] = 0.9
        binary = threshold_probability_map(prob, method="otsu")
        assert binary.dtype == bool
        assert binary[50, 50] == True
        assert binary[5,  5]  == False

    def test_fixed(self):
        prob   = np.full((50, 50), 0.6, dtype=np.float32)
        binary = threshold_probability_map(prob, method="fixed", fixed_value=0.5)
        assert binary.all()

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            threshold_probability_map(np.zeros((10,10)), method="unknown")

    def test_requires_2d(self):
        with pytest.raises(ValueError):
            threshold_probability_map(np.zeros((10, 10, 3)))


class TestMorphology:
    def test_opening_removes_small_noise(self):
        noisy = _circle_mask(r=30)
        noisy[0, 0] = True   # pixel isolado
        cleaned = apply_morphology(noisy, opening_radius=2, closing_radius=0, fill_holes=False)
        assert cleaned[0, 0] == False
        assert cleaned[50, 50] == True

    def test_fill_holes(self):
        ring = np.zeros((60, 60), dtype=bool)
        ring[10:50, 10:50] = True
        ring[20:40, 20:40] = False   # buraco interno
        filled = apply_morphology(ring, opening_radius=0, closing_radius=0, fill_holes=True)
        assert filled[30, 30] == True

    def test_requires_2d(self):
        with pytest.raises(ValueError):
            apply_morphology(np.zeros((10, 10, 3), dtype=bool))


class TestLabeler:
    def test_single_pore(self):
        binary = _circle_mask()
        result = label_pores(binary, min_area_px=10)
        assert result.n_accepted == 1
        assert result.total_area_px > 0
        assert 0 < result.porosity_pct < 100

    def test_multiple_pores(self):
        binary = np.zeros((200, 200), dtype=bool)
        binary[20:50,  20:50]  = True
        binary[100:150, 100:150] = True
        result = label_pores(binary, min_area_px=10)
        assert result.n_accepted == 2

    def test_min_area_filter(self):
        binary = np.zeros((100, 100), dtype=bool)
        binary[10:15, 10:15] = True   # 25 px — abaixo do limite
        binary[50:80, 50:80] = True   # 900 px — acima do limite
        result = label_pores(binary, min_area_px=100)
        assert result.n_accepted == 1

    def test_extract_crop_shape(self):
        binary  = _circle_mask()
        result  = label_pores(binary, min_area_px=10)
        pore    = result.pores[0]
        crop    = extract_pore_crop(pore, padding=5)
        assert crop.ndim == 2
        assert crop.any()
