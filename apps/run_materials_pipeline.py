#!/usr/bin/env python3
import argparse
import logging
import sys

from sciforge.matkit.pipeline.api_client import MockMaterialsProjectClient
from sciforge.matkit.pipeline.clustering import ElectronicClusteringEngine
from sciforge.matkit.pipeline.etl_process import MaterialETLPipeline
from sciforge.matkit.utils.db_client import DatabaseClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(description="Materials Discovery ETL & Clustering Runner")
    parser.add_init_argument("--material-id", type=str, default="mp-1234", help="Target Material ID")
    args = parser.parse_args()

    # Initialize shared persistence layers
    db_client = DatabaseClient()
    db_client.create_tables()

    # Initialize data acquisition layer using strategy pattern
    api_client = MockMaterialsProjectClient(api_key="dev_mock_key")
    pipeline = MaterialETLPipeline(api_client=api_client, db_client=db_client)

    try:
        # Step A: Ingest targeted properties into SQL architecture
        pipeline.run(material_id=args.material_id)

        # Step B: Run data analytics across the updated schema
        logger.info("Triggering clustering calculations across all stored records...")
        engine = ElectronicClusteringEngine(db_client=db_client, n_clusters=2)
        engine.fit()
        results = engine.assign_clusters()

        for record in results:
            logger.info(f"Material {record['formula']} -> Assigned to Cluster {record['cluster_id']}")

    except Exception as e:
        logger.error(f"Application workflow failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
