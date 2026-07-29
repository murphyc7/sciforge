from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from sciforge.matkit.utils.config import Config

Base = declarative_base()


class MaterialModel(Base):
    """Maps to the SQL materials table."""

    __tablename__ = "materials"

    material_id = Column(String(50), primary_key=True)
    formula = Column(String(20), nullable=False)
    space_group = Column(Integer, nullable=False)
    effective_mass_me = Column(Float, nullable=False)
    permittivity = Column(Float, nullable=False)


class DatabaseClient:
    """Handles connection pooling and sessions for the application."""

    def __init__(self) -> None:
        self.engine = create_engine(Config.DB_URL, pool_size=5, max_overflow=10)
        self.SessionLocal = sessionmaker(
            bind=self.engine, autoflush=False, autocommit=False
        )

    def create_tables(self) -> None:
        """Helper to initialize tables if they do not exist."""
        Base.metadata.create_all(bind=self.engine)

    def get_session(self):
        """Returns a context-managed transactional database session."""
        return self.SessionLocal()
