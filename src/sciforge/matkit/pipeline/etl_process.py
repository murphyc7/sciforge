"""Data Engineering Extract, Transform, and Load (ETL) processing components.

Ingests nested JSON parameters from physics clients, cleans layout structures,
and saves them to relational tables.
"""

import logging
from typing import Any

from sciforge.matkit.pipeline.api_client import BaseMaterialClient
from sciforge.matkit.utils.db_client import DatabaseClient, MaterialModel

logger = logging.getLogger(__name__)


class MaterialETLPipeline:
    """Manages the Extract, Transform, and Load lifecycle for materials discovery."""

    def __init__(
        self, api_client: BaseMaterialClient, db_client: DatabaseClient
    ) -> None:
        """Initialises the pipeline runner.

        Args:
            api_client (BaseMaterialClient): Data client matching the retrieval protocol.
            db_client (DatabaseClient): Target database manager for relational storage.
        """
        self.api_client = api_client
        self.db_client = db_client

    def _transform(self, raw_data: dict[str, Any]) -> MaterialModel:
        """Maps an unformatted JSON dictionary response to a structured database model.

        Fails gracefully to standard physical defaults (1.0) if optional nested parameters
        are missing.

        Args:
            raw_data (Dict[str, Any]): Raw semi-structured API output parameters.

        Returns:
            MaterialModel: A database record structured for SQLAlchemy insertion.
        """
        props = raw_data.get("electronic_properties", {})
        return MaterialModel(
            material_id=raw_data["material_id"],
            formula=raw_data["formula"],
            space_group=raw_data["space_group"],
            effective_mass_me=props.get("effective_mass_electrons", 1.0),
            permittivity=props.get("relative_permittivity", 1.0),
        )

    def run(self, material_id: str) -> None:
        """Executes a synchronous extraction, transformation, and load run for a material.

        Args:
            material_id (str): Alphanumeric reference string to ingest.

        Raises:
            Exception: Re-raises any underlying network or database exception encountered.
        """
        try:
            logger.info(f"Extracting data for target: {material_id}")
            raw_payload = self.api_client.fetch_material_data(material_id)

            logger.info("Transforming schema to relational models...")
            record = self._transform(raw_payload)

            logger.info("Loading payload into targeted SQL infrastructure...")
            with self.db_client.get_session() as session:
                session.merge(record)  # Inserts or updates if record exists
                session.commit()
            logger.info(f"Pipeline executed successfully for {material_id}")

        except Exception as e:
            logger.error(f"Pipeline failure encountered: {str(e)}")
            raise e
