"""Tests for murphyc7/sciforge/src/sciforge/matkit/utils/config.py"""

import pytest

from sciforge.matkit.utils.config import Config, _resolve_db_url


@pytest.mark.parametrize(
    ("raw_url", "expected"),
    [
        (
            "postgresql://user:pass@localhost:5432/db",
            "postgresql+psycopg2://user:pass@localhost:5432/db",
        ),
        (
            "postgresql+psycopg://user:pass@localhost:5432/db",
            "postgresql+psycopg://user:pass@localhost:5432/db",
        ),
        ("sqlite:///:memory:", "sqlite:///:memory:"),
    ],
)
def test_resolve_db_url_pins_only_unqualified_postgres_urls(raw_url, expected):
    """Verifies plain PostgreSQL URLs gain the psycopg2 driver and others are untouched."""
    assert _resolve_db_url(raw_url) == expected


def test_config_db_url_never_uses_ambiguous_postgres_scheme():
    """Guards against Config bypassing the driver-pinning helper."""
    assert not Config.DB_URL.startswith("postgresql://")
