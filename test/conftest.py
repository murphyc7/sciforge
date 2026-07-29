import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sciforge.matkit.utils import db_client


@pytest.fixture(name="mock_db")
def fixture_mock_db(monkeypatch):
    """Creates an isolated in-memory SQLite database for testing code behaviors."""
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)  # noqa: N806

    # Construct schemas instantly in memory
    db_client.Base.metadata.create_all(bind=engine)

    client = db_client.DatabaseClient()
    monkeypatch.setattr(client, "engine", engine)
    monkeypatch.setattr(client, "SessionLocal", TestingSessionLocal)

    yield client
    db_client.Base.metadata.drop_all(bind=engine)
    engine.dispose()
