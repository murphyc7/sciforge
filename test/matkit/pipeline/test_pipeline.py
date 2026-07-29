"""Tests for murphyc7/sciforge/src/sciforge/matkit/pipeline/"""

from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import OperationalError

from sciforge.matkit.pipeline import api_client, clustering, etl_process
from sciforge.matkit.utils import db_client

###############################################################
# 1. murphyc7/sciforge/src/sciforge/matkit/utils/db_client.py #
###############################################################


def test_db_client_creates_tables_successfully(
    mock_db: db_client.DatabaseClient,
) -> None:
    """Validates that the initialisation engine maps schemas to the DB correctly."""
    with mock_db.get_session() as session:
        results = session.query(db_client.MaterialModel).all()
        assert len(results) == 0


def test_db_client_create_tables_method_executes(
    mock_db: db_client.DatabaseClient,
) -> None:
    """Explicitly triggers the wrapper table creation method to secure 100% test coverage."""
    # Calling this on our existing mock_db will safely run the line
    # against our in-memory engine without breaking existing schemas.
    mock_db.create_tables()

    with mock_db.get_session() as session:
        # Check that the database can query the table structure cleanly
        assert session.query(db_client.MaterialModel).all() == []


###################################################################
# 2. murphyc7/sciforge/src/sciforge/matkit/pipeline/api_client.py #
###################################################################


def test_mock_api_client_returns_valid_payload() -> None:
    """Ensures the mock API client provides the expected data architecture structures."""
    client = api_client.MockMaterialsProjectClient(api_key="test_key")
    payload = client.fetch_material_data("mp-999")

    assert payload["material_id"] == "mp-999"
    assert "electronic_properties" in payload
    assert payload["electronic_properties"]["effective_mass_electrons"] == 0.067


####################################################################
# 3. murphyc7/sciforge/src/sciforge/matkit/pipeline/etl_process.py #
####################################################################


def test_etl_pipeline_successfully_saves_to_database(mock_db) -> None:
    """Verifies unformatted dictionary data is parsed and stored matching schema standards."""
    client = api_client.MockMaterialsProjectClient(api_key="test_key")
    pipeline = etl_process.MaterialETLPipeline(api_client=client, db_client=mock_db)

    pipeline.run("mp-test")

    with mock_db.get_session() as session:
        record = (
            session.query(db_client.MaterialModel)
            .filter_by(material_id="mp-test")
            .first()
        )
        assert record is not None
        assert record.formula == "GaAs"
        assert record.effective_mass_me == 0.067


def test_etl_pipeline_handles_missing_nested_json_keys(
    mock_db: db_client.DatabaseClient,
) -> None:
    """Verifies the ETL defaults missing structural properties to 1.0 gracefully."""
    bad_client = MagicMock()
    bad_client.fetch_material_data.return_value = {
        "material_id": "mp-broken",
        "formula": "Si",
        "space_group": 227,
    }

    pipeline = etl_process.MaterialETLPipeline(api_client=bad_client, db_client=mock_db)
    pipeline.run("mp-broken")

    with mock_db.get_session() as session:
        record = (
            session.query(db_client.MaterialModel)
            .filter_by(material_id="mp-broken")
            .first()
        )
        assert record is not None
        assert record.effective_mass_me == 1.0  # Fallback triggered
        assert record.permittivity == 1.0  # Fallback triggered


def test_etl_pipeline_logs_and_raises_database_exceptions(
    mock_db: db_client.DatabaseClient,
) -> None:
    """Guarantees the pipeline bubbles up core operational errors if database fails."""
    client = api_client.MockMaterialsProjectClient(api_key="test_key")
    pipeline = etl_process.MaterialETLPipeline(api_client=client, db_client=mock_db)

    mock_session_context = MagicMock()
    mock_session_context.__enter__.side_effect = OperationalError(
        "Connection lost", None, None
    )
    mock_db.get_session = MagicMock(return_value=mock_session_context)

    with pytest.raises(OperationalError):
        pipeline.run("mp-1234")


###################################################################
# 4. murphyc7/sciforge/src/sciforge/matkit/pipeline/clustering.py #
###################################################################


def test_clustering_engine_raises_value_error_when_database_is_empty(mock_db) -> None:
    """Guarantees clustering fails gracefully with a ValueError if no records exist."""
    engine = clustering.ElectronicClusteringEngine(db_client=mock_db, n_clusters=2)
    with pytest.raises(ValueError, match="No material properties found"):
        engine.fit()


def test_clustering_assign_clusters_raises_runtime_error_if_not_fitted(
    mock_db: db_client.DatabaseClient,
) -> None:
    """Ensures calling calculations out of order triggers an explicit operational break."""
    engine = clustering.ElectronicClusteringEngine(db_client=mock_db, n_clusters=2)
    with pytest.raises(RuntimeError, match="fit\\(\\) must be called first"):
        engine.assign_clusters()


def test_clustering_engine_happy_path(mock_db: db_client.DatabaseClient) -> None:
    """Validates the execution flow of training, grouping, and outputting cluster indices."""
    with mock_db.get_session() as session:
        m1 = db_client.MaterialModel(
            material_id="m1",
            formula="A",
            space_group=1,
            effective_mass_me=0.1,
            permittivity=10.0,
        )
        m2 = db_client.MaterialModel(
            material_id="m2",
            formula="B",
            space_group=1,
            effective_mass_me=0.15,
            permittivity=11.0,
        )
        m3 = db_client.MaterialModel(
            material_id="m3",
            formula="C",
            space_group=1,
            effective_mass_me=1.5,
            permittivity=2.0,
        )
        session.add_all([m1, m2, m3])
        session.commit()

    engine = clustering.ElectronicClusteringEngine(db_client=mock_db, n_clusters=2)
    engine.fit()
    results = engine.assign_clusters()

    assert len(results) == 3
    assert "material_id" in results[0]
