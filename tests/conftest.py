"""shared test resources"""

from __future__ import annotations

import warnings
from typing import Any, List, Tuple

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
        import cupy as cp  # type: ignore

        # Verify CUDA is actually available
        cp.cuda.runtime.getDeviceCount()
        return cp
    except (
        ImportError,
        cp.cuda.runtime.CUDARuntimeError if 'cp' in dir() else Exception,
    ):
        return None


def _dask_available():
    """Check if dask.array is available without importing it."""
    try:
        import importlib.util
        return importlib.util.find_spec('dask.array') is not None
    except (ImportError, ModuleNotFoundError):
        return False


def _get_dask_array():
    """Import dask.array lazily."""
    import dask.array as da
    return da


# Build list of available array namespaces
_ARRAY_NAMESPACES: List[Tuple[str, Any]] = [('numpy', np)]

_cupy = _get_cupy()
if _cupy is not None:
    _ARRAY_NAMESPACES.append(('cupy', _cupy))
else:
    warnings.warn(
        'cupy is not available or CUDA is not configured; cupy tests will be skipped',
        UserWarning,
        stacklevel=1,
    )

# Check dask availability without importing (to avoid reifying scipy)
_dask_is_available = _dask_available()
if _dask_is_available:
    _ARRAY_NAMESPACES.append(('dask', None))  # placeholder, loaded lazily
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
    name, ns = request.param
    if name == 'dask' and ns is None:
        return _get_dask_array()
    return ns


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
    name, ns = request.param
    if name == 'dask' and ns is None:
        return (name, _get_dask_array())
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
    if not _dask_is_available:
        pytest.skip('dask.array is not available')
    return _get_dask_array()


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
# Hypothesis strategies for array-based property testing
# ---------------------------------------------------------------------------

def _get_hypothesis_extras():
    """Lazily import hypothesis extras to avoid import overhead."""
    from hypothesis import strategies as st
    from hypothesis.extra.numpy import arrays, array_shapes
    return st, arrays, array_shapes


def positive_power_arrays(
    min_value: float = 1e-15,
    max_value: float = 1e15,
    dtype=None,
    min_size: int = 1,
    max_size: int = 100,
    min_dims: int = 1,
    max_dims: int = 2,
):
    """Strategy for positive power values (valid for log operations).
    
    Specification:
        - All values > 0 (required for log10)
        - Range spans typical RF power measurements (-150 to +150 dBm)
        - Supports float32 and float64 dtypes
    """
    st, arrays, array_shapes = _get_hypothesis_extras()
    
    if dtype is None:
        dtype_strategy = st.sampled_from([np.float32, np.float64])
    else:
        dtype_strategy = st.just(dtype)
    
    return dtype_strategy.flatmap(
        lambda dt: arrays(
            dtype=dt,
            shape=array_shapes(min_dims=min_dims, max_dims=max_dims,
                             min_side=min_size, max_side=max_size),
            elements=st.floats(
                min_value=min_value,
                max_value=max_value,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )


def dB_arrays(
    min_value: float = -150.0,
    max_value: float = 150.0,
    dtype=None,
    min_size: int = 1,
    max_size: int = 100,
    min_dims: int = 1,
    max_dims: int = 2,
    filter_near_zero: bool = False,
):
    """Strategy for dB values in typical measurement range.
    
    Specification:
        - Range: -150 to +150 dB (covers most RF applications)
        - No NaN or infinity
        - Supports float32 and float64 dtypes
        - filter_near_zero: exclude values with |x| < 1e-6 (avoids precision issues)
    """
    st, arrays, array_shapes = _get_hypothesis_extras()
    
    if dtype is None:
        dtype_strategy = st.sampled_from([np.float32, np.float64])
    else:
        dtype_strategy = st.just(dtype)
    
    elements = st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )
    if filter_near_zero:
        elements = elements.filter(lambda x: abs(x) > 1e-6)
    
    return dtype_strategy.flatmap(
        lambda dt: arrays(
            dtype=dt,
            shape=array_shapes(min_dims=min_dims, max_dims=max_dims,
                             min_side=min_size, max_side=max_size),
            elements=elements,
        )
    )


def envelope_arrays(
    min_magnitude: float = 1e-8,
    max_magnitude: float = 1e8,
    include_complex: bool = True,
    dtype=None,
    min_size: int = 1,
    max_size: int = 50,
    min_dims: int = 1,
    max_dims: int = 2,
):
    """Strategy for envelope (amplitude) values, optionally complex.
    
    Specification:
        - Magnitude > 0 (required for log10)
        - Optionally includes complex values
        - Complex values have controlled magnitude
    """
    st, arrays, array_shapes = _get_hypothesis_extras()
    
    if dtype is None:
        dtype_strategy = st.sampled_from([np.float32, np.float64])
    else:
        dtype_strategy = st.just(dtype)
    
    @st.composite
    def _envelope(draw):
        dt = draw(dtype_strategy)
        shape = draw(array_shapes(min_dims=min_dims, max_dims=max_dims,
                                  min_side=min_size, max_side=max_size))
        
        if include_complex and draw(st.booleans()):
            # Generate complex values with controlled magnitude
            magnitudes = draw(arrays(
                dtype=dt,
                shape=shape,
                elements=st.floats(min_value=min_magnitude, max_value=max_magnitude,
                                   allow_nan=False, allow_infinity=False),
            ))
            phases = draw(arrays(
                dtype=dt,
                shape=shape,
                elements=st.floats(min_value=-np.pi, max_value=np.pi,
                                   allow_nan=False, allow_infinity=False),
            ))
            return magnitudes * np.exp(1j * phases)
        else:
            return draw(arrays(
                dtype=dt,
                shape=shape,
                elements=st.floats(min_value=min_magnitude, max_value=max_magnitude,
                                   allow_nan=False, allow_infinity=False),
            ))
    
    return _envelope()


def available_namespaces() -> List[Tuple[str, Any]]:
    """Return list of (name, module) for available array namespaces."""
    namespaces = [('numpy', np)]
    if _cupy is not None:
        namespaces.append(('cupy', _cupy))
    if _dask_is_available:
        namespaces.append(('dask', _get_dask_array()))
    return namespaces


def convert_array(arr: np.ndarray, xp, chunks: str | None = 'auto'):
    """Convert numpy array to target namespace.
    
    Args:
        arr: Source numpy array
        xp: Target array namespace (numpy, cupy, or dask.array)
        chunks: Chunk specification for dask arrays (default: 'auto')
    
    Returns:
        Array in target namespace
    """
    xp_name = getattr(xp, '__name__', str(xp))
    
    if 'cupy' in xp_name:
        return xp.asarray(arr)
    elif 'dask' in xp_name:
        return xp.from_array(arr, chunks=chunks)
    else:
        return arr


def for_each_namespace(base_strategy):
    """Strategy that generates arrays across multiple backends.
    
    Args:
        base_strategy: Strategy yielding numpy arrays
    
    Returns:
        Strategy yielding (array, namespace_name, namespace_module) tuples.
    
    Example:
        @given(data=for_each_namespace(positive_power_arrays()))
        def test_roundtrip(data):
            arr, xp_name, xp = data
            ...
    """
    st, _, _ = _get_hypothesis_extras()
    
    @st.composite
    def _multi_backend(draw):
        available = available_namespaces()
        xp_name, xp = draw(st.sampled_from(available))
        np_arr = draw(base_strategy)
        arr = convert_array(np_arr, xp)
        return arr, xp_name, xp
    
    return _multi_backend()


# ---------------------------------------------------------------------------
# Sweep test fixtures
# ---------------------------------------------------------------------------

SWEEP_DIR = Path(__file__).parent / 'sensor' / 'sweeps'

CPU_RUNS = (
    SWEEP_DIR / 'cw-cpu.yaml',
    SWEEP_DIR / 'dirac_delta-cpu.yaml',
    SWEEP_DIR / 'noise-cpu.yaml',
    SWEEP_DIR / 'sawtooth-cpu.yaml',
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
