"""
tests/test_partisan_runner.py
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

PARTISAN_PATH = Path(__file__).parent.parent.parent / "partisan.py"
has_partisan  = PARTISAN_PATH.exists()


@pytest.mark.skipif(not has_partisan, reason="partisan.py nao encontrado")
class TestPartisanRunner:

    def _make_labeling(self):
        from postprocessing.labeler import label_pores
        binary = np.zeros((200, 200), dtype=bool)
        # Dois poros: circulo e quadrado
        y, x = np.ogrid[:200, :200]
        binary[(x - 60)**2 + (y - 60)**2 <= 30**2] = True
        binary[120:160, 120:160] = True
        return label_pores(binary, min_area_px=50)

    def test_run_returns_dataframe(self):
        from partisan.runner import run_partisan
        import pandas as pd
        labeling = self._make_labeling()
        df = run_partisan(labeling, image_name="test", partisan_path=PARTISAN_PATH, show_progress=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "CI_Circ" in df.columns
        assert "A" in df.columns

    def test_summary_statistics(self):
        from partisan.runner import run_partisan, summary_statistics
        labeling = self._make_labeling()
        df    = run_partisan(labeling, image_name="test", partisan_path=PARTISAN_PATH, show_progress=False)
        stats = summary_statistics(df)
        assert not stats.empty
        assert "mean" in stats.columns
