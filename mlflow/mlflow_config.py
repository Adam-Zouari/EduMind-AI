"""
MLflow Configuration Module

Automatically configure MLflow to use SQLite database backend.
Import this module at the start of your experiment scripts.

Usage:
    from mlflow_config import configure_mlflow
    configure_mlflow()
"""

import mlflow
import os
from pathlib import Path


# Paths - using Path objects to avoid Windows backslash issues
MLFLOW_DIR = Path(__file__).parent
DB_PATH = MLFLOW_DIR / "mlflow.db"
ARTIFACTS_DIR = MLFLOW_DIR / "mlartifacts"

# Database URI - use as_posix() to convert to forward slashes
DB_URI = f"sqlite:///{DB_PATH.as_posix()}"


def configure_mlflow(verbose=True):
    """
    Configure MLflow with database backend.
    
    Args:
        verbose: If True, print configuration info
    """
    # Ensure artifacts directory exists
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    
    # Set tracking URI
    mlflow.set_tracking_uri(DB_URI)
    
    if verbose:
        print(f"[OK] MLflow configured with database backend")
        print(f"  Database: {DB_PATH}")
        print(f"  URI: {DB_URI}")


def get_tracking_uri():
    """Get the MLflow tracking URI."""
    return DB_URI


def get_artifacts_dir():
    """Get the artifacts directory."""
    return ARTIFACTS_DIR


# Auto-configure when imported
configure_mlflow(verbose=False)
