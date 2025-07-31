from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from pathlib import Path

# Create base directory for database if it doesn't exist
import sys
if getattr(sys, 'frozen', False):
    # For packaged app, use a fixed location in user's home directory
    base_dir = Path(os.path.expanduser("~")) / ".pivotal"
    os.makedirs(base_dir, exist_ok=True)
else:
    base_dir = Path(__file__).resolve().parent.parent.parent

db_path = base_dir / "pivotal.db"

# Create SQLite database file in the project root directory
SQLALCHEMY_DATABASE_URL = f"sqlite:///{db_path}"

# Create SQLAlchemy engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base for models
Base = declarative_base()

# Database dependency for FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 