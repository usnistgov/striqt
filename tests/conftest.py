"""shared test resources"""

import pytest
from pathlib import Path

SWEEP_DIR = Path(__file__).parent / 'sweeps'

CPU_RUNS = (
    SWEEP_DIR/'cw-cpu.yaml',
    SWEEP_DIR/'dirac_delta-cpu.yaml',
    SWEEP_DIR/'noise-cpu.yaml',
    SWEEP_DIR/'sawtooth-cpu.yaml'
)


@pytest.fixture(params=CPU_RUNS, ids=[p.name for p in CPU_RUNS])
def cpu_sweep_file(request):
    return str(request.param)


@pytest.fixture(scope='session')
def spec_dir() -> Path:
    return SWEEP_DIR


@pytest.fixture(scope='session')
def output_dir(data_dir) -> Path:
    """path to dataset outputs"""
    return data_dir / 'outputs'
