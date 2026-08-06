"""Tests for murphyc7/sciforge/src/sciforge/matkit/registry/"""

import os
import time

from sciforge.matkit.carrier_pinn.models import CarrierPINN
from sciforge.matkit.registry.manager import SimulationTracker
from sciforge.matkit.utils.db_client import (
    DatabaseClient,
    MaterialModel,
    SimulationRegistryModel,
)


def test_simulation_tracker_saves_metadata_and_onnx_artifact(
    mock_db: DatabaseClient, tmp_path
) -> None:
    """Verifies that the tracker serialises a valid ONNX binary and logs metrics to SQL."""
    # Arrange: Seed a parent material so the foreign key constraint passes
    with mock_db.get_session() as session:
        parent_material = MaterialModel(
            material_id="mp-100",
            formula="GaAs",
            space_group=216,
            effective_mass_me=0.067,
            permittivity=12.9,
        )
        session.add(parent_material)
        session.commit()

    # Point the tracker to use pytest's isolated temporary directory for file output
    tracker = SimulationTracker(db_client=mock_db, artifact_dir=str(tmp_path))
    model = CarrierPINN(input_dim=1, output_dim=1, hidden_dim=8)
    start_clock = time.time()

    # Act: Register a mock training run execution loop
    generated_id = tracker.register_run(
        material_id="mp-100",
        engine_mode="constant",
        epochs=10,
        lr=0.001,
        total_loss=0.05,
        bc_loss=0.02,
        physics_loss=0.03,
        start_time=start_clock,
        model=model,
    )

    # Assert: Verify database primary key output generation
    assert generated_id == 1

    # Assert: Verify row data accuracy from inside the isolated SQL store
    with mock_db.get_session() as session:
        record = (
            session.query(SimulationRegistryModel)
            .filter_by(run_id=generated_id)
            .first()
        )
        assert record is not None
        assert record.engine_mode == "constant"
        assert record.epochs_trained == 10
        assert record.learning_rate == 0.001
        assert record.final_total_loss == 0.05

        # Verify that the physical file path matches and the binary exists on disk
        assert os.path.exists(record.model_artifact_path)
        assert os.path.getsize(record.model_artifact_path) > 0
