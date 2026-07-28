import os


class Config:
    """Manages system configuration and environment variables."""

    DB_URL: str = os.getenv(
        "DATABASE_URL", "postgresql://user:pass@localhost:5432/sciml_db"
    )
    API_KEY: str = os.getenv("MATERIALS_PROJECT_API_KEY", "mock_key_for_testing")
