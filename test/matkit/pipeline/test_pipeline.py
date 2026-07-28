import pytest

from sciforge.matkit.pipeline import api_client, clustering, etl_process
from sciforge.matkit.utils import db_client


def test_etl_pipeline_successfully_saves_to_database(mock_db):
    """Verifies unformatted dictionary data is parsed and stored matching schema standards."""
    client = api_client.MockMaterialsProjectClient(api_key="test_key")
    pipeline = etl_process.MaterialETLPipeline(api_client=client, db_client=mock_db)

    pipeline.run("mp-test")

    with mock_db.get_session() as session:
        record = session.query(db_client.MaterialModel).filter_by(material_id="mp-test").first()
        assert record is not None
        assert record.formula == "GaAs"
        assert record.effective_mass_me == 0.067

def test_clustering_engine_raises_value_error_when_database_is_empty(mock_db):
    """Guarantees clustering fails gracefully with a ValueError if no records exist."""
    engine = clustering.ElectronicClusteringEngine(db_client=mock_db, n_clusters=2)
    with pytest.raises(ValueError, match="No material properties found"):
        engine.fit()
