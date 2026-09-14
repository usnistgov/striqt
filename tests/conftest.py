"""shared test resources"""

from __future__ import annotations

import warnings
from collections import UserDict
from pathlib import Path
from threading import Lock
from typing import Any, List, Tuple

import numpy as np
import pytest

np.seterr(divide='ignore')


class _MemoryShelf(UserDict):
    def sync(self):
        pass


@pytest.fixture(autouse=True, scope='session')
def isolated_persistent_cache():
    """Back util.persistent_lru_cache with an in-memory dict for the test session.

    The on-disk shelf is dbm.ndbm on macOS, which corrupts once the cache evicts
    past its size limit (the window-parameter searches alone write dozens of
    entries per call) or when two test processes write concurrently, after which
    every later get_window call on the machine fails. Tests also should not
    leave one-off entries in the developer's cache.
    """
    from striqt.waveform.lib import util

    shelf = _MemoryShelf(), Lock()
    original = util._get_cache_shelf
    util._get_cache_shelf = lambda: shelf
    try:
        yield
    finally:
        util._get_cache_shelf = original


# ---------------------------------------------------------------------------
# Array namespace fixtures for numerical testing (numpy, cupy, dask)
# ---------------------------------------------------------------------------


def _get_cupy():
    """Try to import cupy, return None if unavailable."""
    import importlib.util

    # importing pandas and scipy here would reify striqt's lazy imports even
    # when cupy is absent, which is what tests/test_imports.py guards against
    if importlib.util.find_spec('cupy') is None:
        return None

    try:
        import numba.cuda
        import cupy as cp  # type: ignore
        import pandas
        import scipy

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

        # Only check for 'dask' top-level to avoid importing dask itself
        return importlib.util.find_spec('dask') is not None
    except (ImportError, ModuleNotFoundError):
        return False


def _get_dask_array():
    """Import dask.array lazily."""
    import dask.array as da

    return da


_cupy = _get_cupy()
if _cupy is None:
    warnings.warn(
        'cupy is not available or CUDA is not configured; cupy tests will be skipped',
        UserWarning,
        stacklevel=1,
    )

# Check dask availability without importing (to avoid reifying scipy)
_dask_is_available = _dask_available()
if not _dask_is_available:
    warnings.warn(
        'dask.array is not available; dask tests will be skipped',
        UserWarning,
        stacklevel=1,
    )


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
    from hypothesis.extra.numpy import array_shapes, arrays

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

    @st.composite
    def _positive_power(draw):
        dt = draw(dtype_strategy)
        float_width = 32 if dt == np.float32 else 64
        # Clamp min/max to representable range for the dtype
        actual_min = float(dt(min_value))
        actual_max = float(dt(max_value))
        shape = draw(
            array_shapes(
                min_dims=min_dims,
                max_dims=max_dims,
                min_side=min_size,
                max_side=max_size,
            )
        )
        return draw(
            arrays(
                dtype=dt,
                shape=shape,
                elements=st.floats(
                    min_value=actual_min,
                    max_value=actual_max,
                    allow_nan=False,
                    allow_infinity=False,
                    width=float_width,
                ),
            )
        )

    return _positive_power()


def dB_arrays(
    min_value: float = -150.0,
    max_value: float = 150.0,
    dtype=None,
    min_size: int = 1,
    max_size: int = 100,
    min_dims: int = 1,
    max_dims: int = 2,
):
    """Strategy for dB values in typical measurement range.

    Specification:
        - Range: -150 to +150 dB (covers most RF applications)
        - No NaN or infinity
        - Supports float32 and float64 dtypes
    """
    st, arrays, array_shapes = _get_hypothesis_extras()

    if dtype is None:
        dtype_strategy = st.sampled_from([np.float32, np.float64])
    else:
        dtype_strategy = st.just(dtype)

    @st.composite
    def _dB(draw):
        dt = draw(dtype_strategy)
        float_width = 32 if dt == np.float32 else 64
        # Clamp min/max to representable range for the dtype
        actual_min = float(dt(min_value))
        actual_max = float(dt(max_value))
        shape = draw(
            array_shapes(
                min_dims=min_dims,
                max_dims=max_dims,
                min_side=min_size,
                max_side=max_size,
            )
        )

        elements = st.floats(
            min_value=actual_min,
            max_value=actual_max,
            allow_nan=False,
            allow_infinity=False,
            width=float_width,
        )

        return draw(
            arrays(
                dtype=dt,
                shape=shape,
                elements=elements,
            )
        )

    return _dB()


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
        float_width = 32 if dt == np.float32 else 64
        # Clamp min/max to representable range for the dtype
        actual_min_mag = float(dt(min_magnitude))
        actual_max_mag = float(dt(max_magnitude))
        actual_min_phase = float(dt(-np.pi))
        actual_max_phase = float(dt(np.pi))
        shape = draw(
            array_shapes(
                min_dims=min_dims,
                max_dims=max_dims,
                min_side=min_size,
                max_side=max_size,
            )
        )

        if include_complex and draw(st.booleans()):
            # Generate complex values with controlled magnitude
            magnitudes = draw(
                arrays(
                    dtype=dt,
                    shape=shape,
                    elements=st.floats(
                        min_value=actual_min_mag,
                        max_value=actual_max_mag,
                        allow_nan=False,
                        allow_infinity=False,
                        width=float_width,
                    ),
                )
            )
            phases = draw(
                arrays(
                    dtype=dt,
                    shape=shape,
                    elements=st.floats(
                        min_value=actual_min_phase,
                        max_value=actual_max_phase,
                        allow_nan=False,
                        allow_infinity=False,
                        width=float_width,
                    ),
                )
            )
            return magnitudes * np.exp(1j * phases)
        else:
            return draw(
                arrays(
                    dtype=dt,
                    shape=shape,
                    elements=st.floats(
                        min_value=actual_min_mag,
                        max_value=actual_max_mag,
                        allow_nan=False,
                        allow_infinity=False,
                        width=float_width,
                    ),
                )
            )

    return _envelope()


def shaped_arrays(
    dtype=np.float32,
    min_dims: int = 1,
    max_dims: int = 3,
    min_side: int = 1,
    max_side: int = 8,
    min_value: float = -10.0,
    max_value: float = 10.0,
):
    """Strategy for (array, axis) pairs, with axis drawn from [-ndim, ndim).

    Negative axes are drawn as often as positive ones so that axis-normalization
    branches are exercised.
    """
    st, arrays, array_shapes = _get_hypothesis_extras()

    @st.composite
    def _shaped(draw):
        shape = draw(
            array_shapes(
                min_dims=min_dims,
                max_dims=max_dims,
                min_side=min_side,
                max_side=max_side,
            )
        )
        elements = st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
            width=32 if np.dtype(dtype) == np.float32 else 64,
        )
        arr = draw(arrays(dtype=dtype, shape=shape, elements=elements))
        axis = draw(st.integers(min_value=-len(shape), max_value=len(shape) - 1))
        return arr, axis

    return _shaped()


def iq_waveforms(
    min_size: int = 64,
    max_size: int = 1024,
    multiple_of: int = 1,
    channels: int | None = None,
    dtype=None,
):
    """Strategy for complex gaussian IQ waveforms.

    Hypothesis draws the dtype, the sample count (a multiple of `multiple_of`),
    and a seed; the samples come from a generator with that seed so that
    examples are reproducible and shrink toward short waveforms.

    Args:
        channels: None for a 1-D waveform, or the number of rows of a 2-D
            (channel, sample) array as the analysis measurements use
    """
    st, _, _ = _get_hypothesis_extras()

    if dtype is None:
        dtype_strategy = st.sampled_from([np.complex64, np.complex128])
    else:
        dtype_strategy = st.just(dtype)

    @st.composite
    def _iq(draw):
        dt = draw(dtype_strategy)
        blocks = draw(
            st.integers(
                min_value=max(min_size // multiple_of, 1),
                max_value=max(max_size // multiple_of, 1),
            )
        )
        size = blocks * multiple_of
        shape = (size,) if channels is None else (channels, size)
        rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**16)))
        iq = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        return iq.astype(dt)

    return _iq()


def available_namespaces() -> List[Tuple[str, Any]]:
    """Return list of (name, module) for available array namespaces.

    Note: dask is returned with a lazy loader to avoid importing scipy
    during test collection.
    """
    namespaces = [('numpy', np)]
    if _cupy is not None:
        namespaces.append(('cupy', _cupy))
    if _dask_is_available:
        # Use lazy loading to avoid importing dask.array (which imports scipy)
        # during test collection
        namespaces.append(('dask', None))  # placeholder, loaded lazily
    return namespaces


def convert_array(arr: np.ndarray, xp, chunks: str | None = 'auto'):
    """Convert numpy array to target namespace.

    Args:
        arr: Source numpy array
        xp: Target array namespace (numpy, cupy, or dask.array), or None for dask
        chunks: Chunk specification for dask arrays (default: 'auto')

    Returns:
        Array in target namespace
    """
    if xp is None:
        # Lazy dask loading
        xp = _get_dask_array()
        return xp.from_array(arr, chunks=chunks)

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
        # Handle lazy dask loading
        if xp_name == 'dask' and xp is None:
            xp = _get_dask_array()
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
    SWEEP_DIR / 'site' / 'site-cpu.yaml',
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


@pytest.fixture(scope='session')
def cw_spec_path() -> Path:
    return SWEEP_DIR / 'cw-cpu.yaml'


@pytest.fixture(scope='session')
def cw_sweep():
    import striqt.sensor as ss

    return ss.read_yaml_spec(SWEEP_DIR / 'cw-cpu.yaml')


@pytest.fixture(scope='session')
def calibration_sweep():
    import striqt.sensor as ss

    return ss.read_json_spec(SWEEP_DIR / 'calibration.json')


@pytest.fixture
def fake_source_id(monkeypatch):
    """stand in for the hardware id lookup that PathFormatter performs"""
    import striqt.sensor as ss

    monkeypatch.setattr(
        ss.lib.controller.lookup, 'id', lambda spec, timeout=0.5: 'beef'
    )
    return 'beef'


# ---------------------------------------------------------------------------
# Spec construction paths
# ---------------------------------------------------------------------------


@pytest.fixture(params=('direct', 'from_dict'))
def construct(request):
    """build a spec either directly or through msgspec conversion"""
    if request.param == 'direct':
        return lambda cls, **kws: cls(**kws)
    return lambda cls, **kws: cls.from_dict(kws)


def raises_on_both_paths(cls, exc, match, **kws):
    """assert that constructing `cls` raises `exc` directly and via from_dict.

    msgspec re-raises ValueError and TypeError from __post_init__ as its own
    ValidationError on the conversion path, so either is accepted there.
    """
    import msgspec

    with pytest.raises(exc, match=match):
        cls(**kws)
    with pytest.raises((exc, msgspec.ValidationError), match=match):
        cls.from_dict(kws)


# ---------------------------------------------------------------------------
# Site-style sweeps: extension module binding + per-source overrides
# ---------------------------------------------------------------------------

SITE_DIR = SWEEP_DIR / 'site'


@pytest.fixture(scope='session')
def site_spec_path() -> Path:
    return SITE_DIR / 'site-cpu.yaml'


@pytest.fixture(scope='session')
def site_sweep():
    import striqt.sensor as ss

    return ss.read_yaml_spec(SITE_DIR / 'site-cpu.yaml')


@pytest.fixture(scope='session')
def site_survey_sweep():
    import striqt.sensor as ss

    return ss.read_yaml_spec(SITE_DIR / 'site-survey.yaml')


@pytest.fixture(scope='session')
def site_calibration_sweep():
    import striqt.sensor as ss

    return ss.read_yaml_spec(SITE_DIR / 'site-calibration.yaml')


@pytest.fixture
def fake_radio_id(monkeypatch):
    """stand in for the hardware id lookup, returning the id keyed in sites/radio02.yaml"""
    from site_strategies import RADIO_ID

    import striqt.sensor as ss

    monkeypatch.setattr(
        ss.lib.controller.lookup, 'id', lambda spec, timeout=0.5: RADIO_ID
    )
    return RADIO_ID


@pytest.fixture
def write_yaml(tmp_path):
    """write dedented YAML text to `tmp_path / relpath`, creating parent directories"""
    import textwrap

    def write(relpath: str, text: str) -> Path:
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))
        return path

    return write


@pytest.fixture
def isolated_extension_import():
    """hide the test suite's `extensions` module so a spec can import its own.

    Restores sys.path and sys.modules afterwards; leaving a temporary module
    cached under the name `extensions` would break every later site YAML read.
    """
    import sys

    src_dir = (SWEEP_DIR / 'src').resolve()
    saved_path = list(sys.path)
    saved_module = sys.modules.pop('extensions', None)
    sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != src_dir]
    try:
        yield
    finally:
        sys.path[:] = saved_path
        sys.modules.pop('extensions', None)
        if saved_module is not None:
            sys.modules['extensions'] = saved_module
