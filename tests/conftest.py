"""shared test resources"""

import pytest
from pathlib import Path


@pytest.fixture(scope='session')
def spec_dir() -> Path:
    """Provides the absolute path to the data directory."""
    # Resolves relative to this conftest.py file location
    return Path(__file__).parent / 'sweeps'


@pytest.fixture(scope='session')
def output_dir(data_dir) -> Path:
    """path to dataset outputs"""
    return data_dir / 'outputs'
