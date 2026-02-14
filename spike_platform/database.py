"""
SQLite database setup via SQLAlchemy.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from typing import Generator

from spike_platform.config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.DB_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
    echo=False,
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable WAL mode for better concurrent access."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Called on app startup."""
    # Import models so they register with Base.metadata
    import spike_platform.models.db_models  # noqa: F401
    Base.metadata.create_all(bind=engine)

    # Migrate existing DBs: add columns that may not exist yet
    with engine.connect() as conn:
        for table, column, col_type in [
            ("videos", "video_group", "TEXT"),
        ]:
            try:
                conn.execute(
                    __import__("sqlalchemy").text(
                        f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"
                    )
                )
                conn.commit()
            except Exception:
                pass  # column already exists
