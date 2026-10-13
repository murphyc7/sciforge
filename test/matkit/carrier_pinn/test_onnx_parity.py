"""Numerical parity tests between exported ONNX artefacts and the source PyTorch model."""

import numpy as np
import pytest
import torch

from sciforge.matkit.carrier_pinn.models import CarrierPINN
from sciforge.matkit.registry.manager import SimulationTracker
from sciforge.matkit.utils.db_client import DatabaseClient

# Skip the whole module cleanly if the optional ONNX runtime is not installed.
ort = pytest.importorskip("onnxruntime")


@pytest.mark.parametrize("output_dim", [1, 2, 3])
@pytest.mark.parametrize("batch_size", [1, 10, 64])
def test_onnx_export_reproduces_pytorch_predictions(
    mock_db: DatabaseClient, tmp_path, output_dim: int, batch_size: int
) -> None:
    """Verifies the exported graph matches PyTorch for every solver width and batch size."""
    # Arrange: Export a seeded model through the tracker's real export path
    torch.manual_seed(0)
    model = CarrierPINN(input_dim=1, output_dim=output_dim, hidden_dim=12)
    model.eval()
    tracker = SimulationTracker(db_client=mock_db, artifact_dir=str(tmp_path))
    onnx_path = tracker._export_to_onnx(model, "parity_check")

    # Act: Evaluate the same coordinates through both runtimes
    x = np.linspace(0.0, 1.0, batch_size, dtype=np.float32).reshape(-1, 1)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: x})[0]
    with torch.no_grad():
        torch_output = model(torch.from_numpy(x)).numpy()

    # Assert: The dynamic batch axis works and values agree to float32 precision
    assert onnx_output.shape == (batch_size, output_dim)
    np.testing.assert_allclose(onnx_output, torch_output, rtol=1e-5, atol=1e-6)
