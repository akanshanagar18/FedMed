"""
Module: dashboard.backend.app.database.session

Purpose:
Configures the SQLAlchemy engine and session maker for SQLite.
Provides get_db() FastAPI dependency and init_db() for table creation
and migration management on startup.
"""

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, Session
from app.config.settings import settings
from app.models.base import Base



engine = create_engine(
    settings.DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def migrate_experiments_table(connection) -> None:
    """Migrate SQLite experiments table to add v2.0 hyperparameter and checkpoint columns."""
    inspector = inspect(connection)
    if "experiments" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("experiments")}
    new_columns = [
        ("strategy_name", "VARCHAR DEFAULT 'FedAvg'"),
        ("num_clients", "INTEGER DEFAULT 2"),
        ("learning_rate", "FLOAT DEFAULT 0.0001"),
        ("batch_size", "INTEGER DEFAULT 2"),
        ("local_epochs", "INTEGER DEFAULT 1"),
        ("num_rounds", "INTEGER DEFAULT 3"),
        ("seed", "INTEGER DEFAULT 42"),
        ("dp_enabled", "BOOLEAN DEFAULT 0"),
        ("he_enabled", "BOOLEAN DEFAULT 0"),
        ("dataset_name", "VARCHAR DEFAULT 'BraTS2021'"),
        ("partition_strategy", "VARCHAR DEFAULT 'IID'"),
        ("notes", "TEXT"),
        ("checkpoint_path", "VARCHAR"),
        ("best_dice_score", "FLOAT"),
        ("best_round", "INTEGER"),
    ]

    for col_name, col_type in new_columns:
        if col_name not in existing_columns:
            connection.execute(text(f"ALTER TABLE experiments ADD COLUMN {col_name} {col_type}"))


def init_db() -> None:
    """Create all tables defined in Base.metadata and apply lightweight migrations."""
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        migrate_experiments_table(conn)


def get_db():
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
