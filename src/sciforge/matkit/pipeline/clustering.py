"""Data Analytics and unsupervised physical parameter clustering engine.

Extracts data vectors from the relational framework, standardises feature variances,
and labels records using k-means clustering.
"""

import logging
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sciforge.matkit.utils.db_client import DatabaseClient, MaterialModel

logger = logging.getLogger(__name__)


class ElectronicClusteringEngine:
    """Clusters database material structural profiles based on physical constraints."""

    def __init__(self, db_client: DatabaseClient, n_clusters: int = 3) -> None:
        """Initialises the statistical clustering model.

        Args:
            db_client (DatabaseClient): Active relational database interface client.
            n_clusters (int, optional): Total categorical splits to create. Defaults to 3.
        """
        self.db_client = db_client
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init="auto")
        self._is_fitted = False

    def _fetch_training_features(self) -> tuple[list[MaterialModel], np.ndarray]:
        """Queries properties from the database and structures them into a numerical array.

        Returns:
            Tuple[List[MaterialModel], np.ndarray]: Database objects alongside a 2D float array.

        Raises:
            ValueError: If the targeted database query returns an empty collection.
        """
        with self.db_client.get_session() as session:
            # Query all rows from the target materials table
            records: list[MaterialModel] = session.query(MaterialModel).all()

            if not records:
                raise ValueError("No material properties found in the database target.")

            # Construct a standard NumPy feature matrix
            features: np.ndarray = np.array(
                [[r.effective_mass_me, r.permittivity] for r in records],
                dtype=np.float64,
            )
            return records, features

    def fit(self) -> "ElectronicClusteringEngine":
        """Fits the data preprocessing pipeline and clustering engine.

        Returns:
            ElectronicClusteringEngine: The trained instance profile for method chaining.
        """
        try:
            logger.info("Extracting data coordinates from relational store...")
            _, features = self._fetch_training_features()

            logger.info("Scaling features and generating cluster centers...")
            scaled_features = self.scaler.fit_transform(features)
            self.model.fit(scaled_features)
            self._is_fitted = True
            return self

        except Exception as e:
            logger.error(f"Clustering fit routine failed abruptly: {str(e)}")
            raise e

    def assign_clusters(self) -> list[dict[str, Any]]:
        """Maps computed categorical groupings back to structural metadata identities.

        Returns:
            List[Dict[str, Any]]: Explicit layout summaries matching data mappings.

        Raises:
            RuntimeError: If called before running the required fit pipeline loops.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot assign clusters; fit() must be called first.")

        records, features = self._fetch_training_features()
        scaled_features = self.scaler.transform(features)
        predictions: np.ndarray = self.model.predict(scaled_features)

        results: list[dict[str, Any]] = []
        for record, cluster_idx in zip(records, predictions, strict=False):
            # Dynamic dictionary representation of data-driven outputs
            summary: dict[str, Any] = {
                "material_id": str(record.material_id),
                "formula": str(record.formula),
                "cluster_id": int(cluster_idx),
            }
            results.append(summary)

        return results
