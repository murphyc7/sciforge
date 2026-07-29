#!/usr/bin/env python3
import argparse
import logging
import sys

from sciforge.matkit.pipeline.api_client import MockMaterialsProjectClient
from sciforge.matkit.pipeline.clustering import ElectronicClusteringEngine
from sciforge.matkit.pipeline.etl_process import MaterialETLPipeline
from sciforge.matkit.utils.db_client import DatabaseClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materials Discovery ETL & Clustering Runner"
    )
    # FIX: Change to accept one or more material IDs as a list
    parser.add_argument(
        "--material-ids",
        type=str,
        nargs="+",
        default=["mp-1234"],
        help="Space-separated list of Material IDs",
    )
    args = parser.parse_args()

    db_client = DatabaseClient()
    db_client.create_tables()

    api_client = MockMaterialsProjectClient(api_key="dev_mock_key")
    pipeline = MaterialETLPipeline(api_client=api_client, db_client=db_client)

    try:
        # Step A: Run ingestion loop for ALL requested target profiles first
        logger.info(f"Targeting batch ingestion for: {args.material_ids}")
        for mat_id in args.material_ids:
            pipeline.run(material_id=mat_id)

        # Step B: Now that the database is fully seeded, run clustering exactly once
        logger.info("Triggering clustering calculations across all stored records...")
        engine = ElectronicClusteringEngine(db_client=db_client, n_clusters=2)
        engine.fit()
        results = engine.assign_clusters()

        for record in results:
            logger.info(
                f"Material {record['formula']} ({record['material_id']}) -> Cluster {record['cluster_id']}"
            )

    except Exception as e:
        logger.error(f"Application workflow failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
