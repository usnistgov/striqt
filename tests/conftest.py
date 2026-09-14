"""shared test resources"""

from __future__ import annotations

import importlib.util
import warnings
from collections import UserDict
from pathlib import Path
from threading import Lock

import numpy as np
import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

np.seterr(divide='ignore')

# every property test in the suite runs under these settings: the array fixtures
# below are function scoped, and JIT compilation and GPU transfers make the
# per-example deadline meaningless
settings.register_profile(
    'striqt', suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
)
settings.load_profile('striqt')


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
# Array namespaces (numpy, cupy, dask)
# ---------------------------------------------------------------------------


def _get_cupy():
    """the cupy module when a CUDA device is available, otherwise None"""
    # importing pandas and scipy here would reify striqt's lazy imports even
    # when cupy is absent, which is what tests/test_imports.py guards against
    if importlib.util.find_spec('cupy') is None:
        return None

    try:
        import numba.cuda
        import cupy as cp  # type: ignore
        import pandas
        import scipy

        cp.cuda.runtime.getDeviceCount()
        return cp
    except (
        ImportError,
        cp.cuda.runtime.CUDARuntimeError if 'cp' in dir() else Exception,
    ):
        return None


_cupy = _get_cupy()
if _cupy is None:
    warnings.warn(
        'cupy is not available or CUDA is not configured; cupy tests will be skipped',
        UserWarning,
        stacklevel=1,
    )

# checked without importing, which would reify scipy
_dask_is_available = importlib.util.find_spec('dask') is not None
if not _dask_is_available:
    warnings.warn(
        'dask.array is not available; dask tests will be skipped',
        UserWarning,
        stacklevel=1,
    )

NAMESPACES = ['numpy']
if _cupy is not None:
    NAMESPACES.append('cupy')
if _dask_is_available:
    NAMESPACES.append('dask')


@pytest.fixture
def cupy_available():
    """the cupy module, skipping the test when no CUDA device is available"""
    if _cupy is None:
        pytest.skip('cupy is not available')
    return _cupy


@pytest.fixture(params=['numpy', 'cupy'])
def xp(request):
    """each array namespace in turn; the cupy case skips without a CUDA device"""
    if request.param == 'cupy':
        if _cupy is None:
            pytest.skip('cupy is not available')
        return _cupy
    return np


def to_numpy(arr):
    """a numpy copy of a numpy, cupy or dask array"""
    if hasattr(arr, 'get'):  # cupy
        return arr.get()
    elif hasattr(arr, 'compute'):  # dask
        return arr.compute()
    return np.asarray(arr)


def convert_array(arr: np.ndarray, xp_name: str, chunks='auto'):
    """`arr` converted into the array namespace named by `xp_name`"""
    if xp_name == 'cupy':
        return _cupy.asarray(arr)
    elif xp_name == 'dask':
        import dask.array as da

        return da.from_array(arr, chunks=chunks)
    return arr


def numpy_and_cupy(cp, func, *args, **kws):
    """`func` evaluated on numpy arguments and again on their cupy copies.

    Returns the two results as numpy arrays for comparison.
    """
    result_np = func(*args, **kws)
    cp_args = [cp.asarray(a) if isinstance(a, np.ndarray) else a for a in args]
    return result_np, to_numpy(func(*cp_args, **kws))


# ---------------------------------------------------------------------------
# Hypothesis strategies for array-based property testing
# ---------------------------------------------------------------------------


def _float_dtypes(dtype):
    if dtype is None:
        return st.sampled_from([np.float32, np.float64])
    return st.just(dtype)


def float_arrays(shape, dtype=np.float64, min_value=-10.0, max_value=10.0):
    """finite values of `dtype` in [min_value, max_value] (rounded to the dtype)"""
    dtype = np.dtype(dtype)
    return arrays(
        dtype=dtype,
        shape=shape,
        elements=st.floats(
            min_value=float(dtype.type(min_value)),
            max_value=float(dtype.type(max_value)),
            allow_nan=False,
            allow_infinity=False,
            width=dtype.itemsize * 8,
        ),
    )


def bounded_float_arrays(
    min_value: float,
    max_value: float,
    dtype=None,
    min_size: int = 1,
    max_size: int = 100,
    min_dims: int = 1,
    max_dims: int = 2,
):
    """float32 or float64 arrays (or `dtype`) with values in [min_value, max_value]"""

    @st.composite
    def _bounded(draw):
        dt = draw(_float_dtypes(dtype))
        shape = draw(
            array_shapes(
                min_dims=min_dims,
                max_dims=max_dims,
                min_side=min_size,
                max_side=max_size,
            )
        )
        return draw(float_arrays(shape, dt, min_value, max_value))

    return _bounded()


def positive_power_arrays(min_value: float = 1e-15, max_value: float = 1e15, **kws):
    """positive power values spanning typical RF measurements (-150 to +150 dBm)"""
    return bounded_float_arrays(min_value, max_value, **kws)


def dB_arrays(min_value: float = -150.0, max_value: float = 150.0, **kws):
    """dB values in the typical measurement range"""
    return bounded_float_arrays(min_value, max_value, **kws)


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
    """envelope (amplitude) values with magnitude in [min_magnitude, max_magnitude].

    With `include_complex`, half of the examples carry a uniformly drawn phase.
    """

    @st.composite
    def _envelope(draw):
        dt = draw(_float_dtypes(dtype))
        shape = draw(
            array_shapes(
                min_dims=min_dims,
                max_dims=max_dims,
                min_side=min_size,
                max_side=max_size,
            )
        )
        magnitudes = draw(float_arrays(shape, dt, min_magnitude, max_magnitude))
        if include_complex and draw(st.booleans()):
            phases = draw(float_arrays(shape, dt, -np.pi, np.pi))
            return magnitudes * np.exp(1j * phases)
        return magnitudes

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
    """(array, axis) pairs, with axis drawn from [-ndim, ndim).

    Negative axes are drawn as often as positive ones so that axis-normalization
    branches are exercised.
    """

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
        arr = draw(float_arrays(shape, dtype, min_value, max_value))
        axis = draw(st.integers(min_value=-len(shape), max_value=len(shape) - 1))
        return arr, axis

    return _shaped()


def gaussian_iq(shape, dtype=np.complex64, seed: int = 0):
    """unit-variance gaussian noise from a fixed seed; complex unless `dtype` is real"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    if np.dtype(dtype).kind == 'c':
        x = x + 1j * rng.standard_normal(shape)
    return x.astype(dtype)


def iq_waveforms(
    min_size: int = 64,
    max_size: int = 1024,
    multiple_of: int = 1,
    channels: int | None = None,
    dtype=None,
    log_power: tuple[int, int] = (0, 0),
):
    """gaussian noise waveforms, complex unless `dtype` is a real float type.

    Hypothesis draws the dtype, the sample count (a multiple of `multiple_of`), the
    power in decades from the `log_power` range, and a seed; the samples come from a
    generator with that seed so that examples are reproducible and shrink toward
    short waveforms.

    Args:
        channels: None for a 1-D waveform, or the number of rows of a 2-D
            (channel, sample) array as the analysis measurements use
        dtype: a dtype, a sequence of dtypes to draw from, or None for complex64
            and complex128
    """
    if dtype is None:
        dtype = [np.complex64, np.complex128]
    if isinstance(dtype, (list, tuple)):
        dtype_strategy = st.sampled_from(dtype)
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
        seed = draw(st.integers(min_value=0, max_value=2**16))
        scale = 10 ** (draw(st.integers(*log_power)) / 2)
        return (scale * gaussian_iq(shape, dt, seed)).astype(dt, copy=False)

    return _iq()


def for_each_namespace(base_strategy):
    """(array, namespace name) pairs of `base_strategy` examples, converted into each
    available array namespace in turn"""

    @st.composite
    def _multi_backend(draw):
        xp_name = draw(st.sampled_from(NAMESPACES))
        return convert_array(draw(base_strategy), xp_name), xp_name

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
