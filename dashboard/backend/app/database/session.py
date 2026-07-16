"""
Module: dashboard.backend.app.database.session

Purpose:
Configures the SQLAlchemy engine and session maker for SQLite.
This is currently a placeholder interface for future database connectivity.

TODO:
- [ ] Ensure database tables are created on startup (Base.metadata.create_all).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config.settings import settings

# engine = create_engine(
#     settings.DATABASE_URL, connect_args={"check_same_thread": False}
# )
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency to get DB session."""
    # db = SessionLocal()
    # try:
    #     yield db
    # finally:
    #     db.close()
    yield None
