from abc import ABC, abstractmethod
from typing import Any


class BaseMaterialClient(ABC):
    """Abstract interface for scientific data discovery APIs."""

    @abstractmethod
    def fetch_material_data(self, material_id: str) -> dict[str, Any]:
        pass


class MockMaterialsProjectClient(BaseMaterialClient):
    """Concrete implementation fetching material structures (Mocked for safety)."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def fetch_material_data(self, material_id: str) -> dict[str, Any]:
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
