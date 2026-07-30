"""Scientific discovery data acquisition interfaces and abstraction strategies.

Defines the structure for pulling material characteristics from third-party APIs while
decoupling structural parsing rules from the application workflow.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseMaterialClient(ABC):
    """Abstract base contract enforcing data structures for third-party physics APIs."""

    @abstractmethod
    def fetch_material_data(self, material_id: str) -> dict[str, Any]:
        """Extracts structural and electronic summaries for a target physical material.

        Args:
            material_id (str): The alphanumeric registry tracking string (e.g., 'mp-555').

        Returns:
            Dict[str, Any]: Nested dictionary representation matching physical schemas.
        """
        pass  # pragma: no cover


class MockMaterialsProjectClient(BaseMaterialClient):
    """Deterministic mock provider mimicking production web client responses.

    Insulates development workflows from remote server downtime, authorization token
    requirements, and network transmission delays.
    """

    def __init__(self, api_key: str) -> None:
        """Initializes the mock discovery interface.

        Args:
            api_key (str): Synthetic developer token string used to mimic web connection steps.
        """
        self.api_key = api_key

    def fetch_material_data(self, material_id: str) -> dict[str, Any]:
        """Generates a deterministic structural profile for a target material identifier.

        Args:
            material_id (str): Alphanumeric identification key.

        Returns:
            Dict[str, Any]: Mocked semi-structured electronic payload parameters.
        """
        # Simulating an API response from a service like the Materials Project
        return {
            "material_id": material_id,
            "formula": "GaAs",
            "space_group": 216,
            "electronic_properties": {
                "effective_mass_electrons": 0.067,
                "relative_permittivity": 12.9,
            },
        }
