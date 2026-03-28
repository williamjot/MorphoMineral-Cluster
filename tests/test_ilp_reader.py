"""
tests/test_ilp_reader.py
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.ilp_reader import ILPReader


def _make_mock_ilp(tmp_path, label_names=None):
    import h5py
    path = tmp_path / "mock.ilp"
    names = label_names or [b"Pore", b"Particle", b"Matrix"]
    with h5py.File(path, "w") as f:
        f.create_dataset("ilastikVersion", data=b"1.4.0")
        f.create_dataset("workflowName",   data=b"PixelClassificationWorkflow")
        f.create_dataset("PixelClassification/LabelNames", data=np.array(names))
        role = f.require_group("Input Data/infos/lane0/Raw Data")
        role.create_dataset("axisorder", data=b"yxc")
        role.create_dataset("shape", data=np.array([256, 256, 1]))
        fs = f.require_group("FeatureSelections")
        fs.create_dataset("FeatureIds",      data=np.array([b"GaussianSmoothing", b"LaplacianOfGaussian"]))
        fs.create_dataset("Scales",          data=np.array([0.3, 0.7, 1.0]))
        fs.create_dataset("SelectionMatrix", data=np.array([[True, False, True],[False, True, False]]))
    return path


class TestILPReader:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ILPReader("nao_existe.ilp")

    def test_read_mock(self, tmp_path):
        meta = ILPReader(_make_mock_ilp(tmp_path)).read()
        assert meta.label_names == ["Pore", "Particle", "Matrix"]
        assert meta.n_classes == 3
        assert meta.pore_class_index == 0
        assert meta.ilastik_version == "1.4.0"

    def test_force_pore_index(self, tmp_path):
        meta = ILPReader(_make_mock_ilp(tmp_path, [b"Matrix", b"Particle", b"Void"])).read(pore_class_index=2)
        assert meta.pore_class_index == 2

    def test_features_extracted(self, tmp_path):
        meta = ILPReader(_make_mock_ilp(tmp_path)).read()
        assert len(meta.selected_features) > 0

    def test_describe(self, tmp_path):
        desc = ILPReader(_make_mock_ilp(tmp_path)).read().describe()
        assert "<- PORO" in desc
