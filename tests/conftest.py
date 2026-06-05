"""shared test resources"""

import pytest
from pathlib import Path

def get_sweep_dir() -> Path:
    """absolute path to the data directory"""
    return Path(__file__).parent / 'sweeps'


RUNS = list(get_sweep_dir().glob('*-cpu.yaml'))


@pytest.fixture(params=RUNS, ids=[p.name for p in RUNS])
def cpu_sweep_file(request):
    return str(request.param)


@pytest.fixture(scope='session')
def spec_dir() -> Path:
    return get_sweep_dir()


@pytest.fixture(scope='session')
def output_dir(data_dir) -> Path:
    """path to dataset outputs"""
    return data_dir / 'outputs'
