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
        self.api_client = api_client
        self.db_client = db_client

    def _transform(self, raw_data: dict[str, Any]) -> MaterialModel:
        """Transforms raw API JSON into a strictly typed database ORM entity."""
        props = raw_data.get("electronic_properties", {})
        return MaterialModel(
            material_id=raw_data["material_id"],
            formula=raw_data["formula"],
            space_group=raw_data["space_group"],
            effective_mass_me=props.get("effective_mass_electrons", 1.0),
            permittivity=props.get("relative_permittivity", 1.0),
        )

    def run(self, material_id: str) -> None:
        """Executes the pipeline synchronously for a target material."""
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
