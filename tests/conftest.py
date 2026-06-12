"""shared test resources"""

import warnings
from typing import Any

import numpy as np
import pytest
from pathlib import Path

np.seterr(divide='ignore')


# ---------------------------------------------------------------------------
# Array namespace fixtures for numerical testing (numpy, cupy, dask)
# ---------------------------------------------------------------------------

def _get_cupy():
    """Try to import cupy, return None if unavailable."""
    try:
        import cupy as cp
        # Verify CUDA is actually available
        cp.cuda.runtime.getDeviceCount()
        return cp
    except (ImportError, cp.cuda.runtime.CUDARuntimeError if 'cp' in dir() else Exception):
        return None


def _get_dask_array():
    """Try to import dask.array, return None if unavailable."""
    try:
        import dask.array as da
        return da
    except ImportError:
        return None


# Build list of available array namespaces
_ARRAY_NAMESPACES: list[tuple[str, Any]] = [('numpy', np)]

_cupy = _get_cupy()
if _cupy is not None:
    _ARRAY_NAMESPACES.append(('cupy', _cupy))
else:
    warnings.warn(
        'cupy is not available or CUDA is not configured; cupy tests will be skipped',
        UserWarning,
        stacklevel=1,
    )

_dask_array = _get_dask_array()
if _dask_array is not None:
    _ARRAY_NAMESPACES.append(('dask', _dask_array))
else:
    warnings.warn(
        'dask.array is not available; dask tests will be skipped',
        UserWarning,
        stacklevel=1,
    )


@pytest.fixture(params=_ARRAY_NAMESPACES, ids=[name for name, _ in _ARRAY_NAMESPACES])
def xp(request):
    """Parameterized fixture providing array namespace (numpy, cupy, dask).

    Use this fixture to write tests that run against multiple array backends.

    Example:
        def test_my_function(xp):
            arr = xp.array([1, 2, 3])
            result = my_function(arr)
            # assertions...
    """
    return request.param[1]


@pytest.fixture(params=_ARRAY_NAMESPACES, ids=[name for name, _ in _ARRAY_NAMESPACES])
def xp_name(request):
    """Parameterized fixture providing (name, namespace) tuple.

    Use when you need both the name and the namespace.

    Example:
        def test_my_function(xp_name):
            name, xp = xp_name
            if name == 'dask':
                pytest.skip('dask not supported for this test')
            arr = xp.array([1, 2, 3])
    """
    return request.param


@pytest.fixture
def np_array():
    """Fixture providing numpy module (always available)."""
    return np


@pytest.fixture
def cp_array():
    """Fixture providing cupy module, skips if unavailable."""
    if _cupy is None:
        pytest.skip('cupy is not available')
    return _cupy


@pytest.fixture
def da_array():
    """Fixture providing dask.array module, skips if unavailable."""
    if _dask_array is None:
        pytest.skip('dask.array is not available')
    return _dask_array


def to_numpy(arr):
    """Convert array from any namespace to numpy for comparison.

    Handles numpy, cupy, and dask arrays.
    """
    if hasattr(arr, 'get'):  # cupy
        return arr.get()
    elif hasattr(arr, 'compute'):  # dask
        return arr.compute()
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# Sweep test fixtures
# ---------------------------------------------------------------------------

SWEEP_DIR = Path(__file__).parent / 'sensor' / 'sweeps'

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
