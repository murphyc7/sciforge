import os


def _resolve_db_url(raw_url: str) -> str:
    """Pins plain PostgreSQL URLs to the psycopg2 driver declared in pyproject.toml.

    SQLAlchemy 2.1 changed the default driver for ``postgresql://`` URLs from psycopg2
    to psycopg (v3), so an unqualified URL fails on a fresh install with a missing
    ``psycopg`` module. URLs that already name a driver or backend are left untouched.

    Args:
        raw_url (str): The database URL as read from the environment.

    Returns:
        str: The URL with an explicit psycopg2 driver where applicable.
    """
    if raw_url.startswith("postgresql://"):
        return raw_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return raw_url


class Config:
    """Manages system configuration and environment variables."""

    DB_URL: str = os.getenv(
        "DATABASE_URL", "postgresql://user:pass@localhost:5432/sciml_db"
    )
    API_KEY: str = os.getenv("MATERIALS_PROJECT_API_KEY", "mock_key_for_testing")
