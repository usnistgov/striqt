"""shared test resources"""

from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
import warnings
from collections import UserDict
from pathlib import Path
from threading import Lock
from typing import NamedTuple

import platformdirs

_user_cache_dir = Path(platformdirs.user_cache_dir('striqt'))

# STRIQT_CACHE_DIR names a directory of this session, so the collection-time spec
# validation below and a shelf fault that deletes the cache files touch no cache of the
# developer or of a concurrent test run
_session_cache_dir = None
if not os.environ.get('STRIQT_CACHE_DIR'):
    _session_cache_dir = tempfile.mkdtemp(prefix='striqt-tests-')
    os.environ['STRIQT_CACHE_DIR'] = _session_cache_dir

# matplotlib reads the backend and config dir on first import. The figures tests plot
# headless. The font cache is written under the user cache directory because a sandbox
# without write access to ~/.matplotlib rebuilds the cache on every run
os.environ.setdefault('MPLBACKEND', 'agg')
if not os.environ.get('MPLCONFIGDIR'):
    (_user_cache_dir / 'matplotlib').mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(_user_cache_dir / 'matplotlib')

import msgspec
import numpy as np
import pytest
from hypothesis import HealthCheck, settings, strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

np.seterr(divide='ignore')

# every property test in the suite runs under these settings. function_scoped_fixture
# is suppressed because the array fixtures below are function scoped, and the deadline
# is removed because JIT compilation and GPU transfers make a per-example deadline
# meaningless. differing_executors is suppressed because the hypothesis plugin
# recognizes parametrization only through parametrize marks or fixture params, not the
# pytest_generate_tests call that drives the `xp` fixture (hypothesis issue 3733)
settings.register_profile(
    'striqt',
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.differing_executors,
    ],
    deadline=None,
)
settings.load_profile('striqt')


class _MemoryShelf(UserDict):
    def sync(self):
        pass


# `util.persistent_cache` is backed by an in-memory dict for the whole pytest process,
# installed at import rather than by a session fixture. Module-level spec reads in the
# strategy modules (`site_strategies.py`) validate sweeps during collection and design
# windows before any fixture runs. The on-disk shelf is dbm.ndbm on macOS python 3.9,
# which corrupts once the cache evicts past its size limit (the window-parameter
# searches alone write dozens of entries per call) and then fails every later
# `get_window` call in the process
from striqt.waveform.lib import util as _sw_util

_memory_shelf = _MemoryShelf(), Lock()
_disk_shelf_getter = _sw_util._get_cache_shelf
_sw_util._get_cache_shelf = lambda: _memory_shelf


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _sw_util._get_cache_shelf = _disk_shelf_getter
    if _session_cache_dir is not None:
        shutil.rmtree(_session_cache_dir, ignore_errors=True)


# %% Array namespaces (numpy, cupy, dask)


def _get_cupy():
    """return the cupy module when a CUDA device is available, otherwise None."""
    # an unconditional import of pandas and scipy here would reify the lazy imports
    # of striqt even when cupy is absent. tests/test_imports.py asserts that the lazy
    # imports stay unreified
    if importlib.util.find_spec('cupy') is None:
        return None

    try:
        import cupy as cp  # type: ignore
        import pandas  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        return None

    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        return None
    return cp


_cupy = _get_cupy()
if _cupy is None:
    warnings.warn(
        'cupy is not available or CUDA is not configured; cupy tests will be skipped',
        UserWarning,
        stacklevel=1,
    )

# an import of dask would reify scipy, so the check uses find_spec
_dask_is_available = importlib.util.find_spec('dask') is not None


def namespace_module(name: str):
    """return the array module named 'numpy', 'cupy', or 'dask', or skip the test
    when cupy or dask is unavailable."""
    if name == 'numpy':
        return np
    elif name == 'cupy':
        if _cupy is None:
            pytest.skip('cupy is not available')
        return _cupy
    elif name == 'dask':
        if not _dask_is_available:
            pytest.skip('dask is not available')
        import dask.array as da

        return da
    raise ValueError(f'unknown array namespace {name!r}')


@pytest.fixture
def cupy_available():
    """return the cupy module, or skip the test when no CUDA device is available."""
    return namespace_module('cupy')


@pytest.fixture(scope='module')
def real_soapy():
    """return the SoapySDR module that striqt imported, or skip the test when the
    bindings are absent."""
    from striqt.sensor.lib.sources import soapy

    pytest.importorskip('SoapySDR')
    if (
        soapy.SoapySDR is None
        or getattr(soapy.SoapySDR, '__name__', None) != 'SoapySDR'
    ):
        pytest.skip('striqt did not import SoapySDR at load time')
    return soapy.SoapySDR


def pytest_generate_tests(metafunc):
    """parametrize `xp` over the namespaces named by a `namespaces` marker, or numpy
    and cupy by default."""
    if 'xp' in metafunc.fixturenames:
        marker = metafunc.definition.get_closest_marker('namespaces')
        names = marker.args if marker else ('numpy', 'cupy')
        metafunc.parametrize('xp', names, indirect=True)


@pytest.fixture
def xp(request):
    """return the array namespace of the current parameter. The cupy and dask cases
    skip when the backend is unavailable."""
    return namespace_module(request.param)


def as_xp(xp, *arrays):
    """convert numpy `arrays` into the namespace `xp`. The return is one array for one
    input and a tuple for several."""

    def convert(arr):
        if xp is np:
            return arr
        elif xp.__name__.startswith('dask'):
            return xp.from_array(arr, chunks='auto')
        return xp.asarray(arr)

    converted = tuple(convert(a) for a in arrays)
    return converted[0] if len(converted) == 1 else converted


# %% Hypothesis strategies for array-based property testing

bounded_ints = st.integers(min_value=-(10**6), max_value=10**6)
finite_floats = st.floats(allow_nan=False, allow_infinity=False)
short_text = st.text(max_size=8)
scalars = st.one_of(st.none(), st.booleans(), bounded_ints, finite_floats, short_text)


def float_dtypes(dtype=None):
    """draw float32 or float64, or only `dtype` when given."""
    if dtype is None:
        return st.sampled_from([np.float32, np.float64])
    return st.just(dtype)


def _shape_strategy(min_dims, max_dims, min_side, max_side):
    return array_shapes(
        min_dims=min_dims, max_dims=max_dims, min_side=min_side, max_side=max_side
    )


def float_arrays(shape, dtype=np.float64, min_value=-10.0, max_value=10.0):
    """draw arrays of `shape` with finite `dtype` values in [`min_value`,
    `max_value`], the bounds rounded to `dtype`."""
    dtype = np.dtype(dtype)
    lo, hi = float(dtype.type(min_value)), float(dtype.type(max_value))
    elements = st.floats(
        min_value=lo,
        max_value=hi,
        allow_nan=False,
        allow_infinity=False,
        width=dtype.itemsize * 8,
    )
    return arrays(dtype=dtype, shape=shape, elements=elements)


def bounded_float_arrays(
    min_value: float,
    max_value: float,
    dtype=None,
    min_size: int = 1,
    max_size: int = 100,
    min_dims: int = 1,
    max_dims: int = 2,
):
    """draw float32 or float64 arrays (or `dtype`) with `min_dims` to `max_dims`
    dimensions, sides of `min_size` to `max_size`, and values in [`min_value`,
    `max_value`]."""

    @st.composite
    def _bounded(draw):
        dt = draw(float_dtypes(dtype))
        shape = draw(_shape_strategy(min_dims, max_dims, min_size, max_size))
        return draw(float_arrays(shape, dt, min_value, max_value))

    return _bounded()


def positive_power_arrays(min_value: float = 1e-15, max_value: float = 1e15, **kws):
    """draw arrays of positive linear power values in [`min_value`, `max_value`], by
    default 1e-15 to 1e15 mW (-150 to +150 dBm)."""
    return bounded_float_arrays(min_value, max_value, **kws)


def dB_arrays(min_value: float = -150.0, max_value: float = 150.0, **kws):
    """draw arrays of dB values in [`min_value`, `max_value`]."""
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
    """draw envelope (amplitude) arrays with magnitude in [`min_magnitude`,
    `max_magnitude`].

    With `include_complex`, half of the examples are complex with a uniformly drawn
    phase.
    """

    @st.composite
    def _envelope(draw):
        dt = draw(float_dtypes(dtype))
        shape = draw(_shape_strategy(min_dims, max_dims, min_size, max_size))
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
    """draw (array, axis) pairs with `axis` in [-ndim, ndim).

    Negative axes are drawn as often as positive ones.
    """

    @st.composite
    def _shaped(draw):
        shape = draw(_shape_strategy(min_dims, max_dims, min_side, max_side))
        arr = draw(float_arrays(shape, dtype, min_value, max_value))
        axis = draw(st.integers(min_value=-len(shape), max_value=len(shape) - 1))
        return arr, axis

    return _shaped()


def gaussian_iq(shape, dtype=np.complex64, seed: int = 0):
    """return standard normal noise of `shape` from `seed`, complex unless `dtype` is
    real."""
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
    """draw gaussian noise waveforms, complex unless `dtype` is a real float type.

    Each example draws a dtype, a sample count, a power, and the seed of the
    generator.

    Args:
        min_size: lower bound on the sample count, floored to a multiple of
            `multiple_of`
        max_size: upper bound on the sample count, floored to a multiple of
            `multiple_of`
        multiple_of: the sample count is a multiple of this value
        channels: `None` for a 1-D waveform, or the row count of a 2-D
            (channel, sample) array
        dtype: a dtype, a sequence of dtypes to draw from, or `None` for complex64
            and complex128
        log_power: (low, high) inclusive range of the waveform power in decades
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
        min_blocks = max(min_size // multiple_of, 1)
        max_blocks = max(max_size // multiple_of, 1)
        blocks = draw(st.integers(min_value=min_blocks, max_value=max_blocks))
        size = blocks * multiple_of
        shape = (size,) if channels is None else (channels, size)
        seed = draw(st.integers(min_value=0, max_value=2**16))
        scale = 10 ** (draw(st.integers(*log_power)) / 2)
        return (scale * gaussian_iq(shape, dt, seed)).astype(dt, copy=False)

    return _iq()


# %% Sweep test fixtures

SWEEP_DIR = Path(__file__).parent / 'sensor' / 'sweeps'


@pytest.fixture
def sweeps_copy(tmp_path) -> Path:
    """a copy of the fixture sweep tree under `tmp_path`, for tests that edit or write
    beside a spec file"""
    dest = tmp_path / 'sweeps'
    shutil.copytree(SWEEP_DIR, dest, ignore=shutil.ignore_patterns('__pycache__'))
    return dest.resolve()


@pytest.fixture(scope='session')
def synthetic_spec_path() -> Path:
    return SWEEP_DIR / 'synthetic.yaml'


@pytest.fixture(scope='session')
def synthetic_sweep():
    import striqt.sensor as ss

    return ss.read_yaml_spec(SWEEP_DIR / 'synthetic.yaml')


@pytest.fixture(scope='session')
def calibration_sweep():
    import striqt.sensor as ss

    return ss.read_json_spec(SWEEP_DIR / 'calibration.json')


@pytest.fixture
def fake_source_id(monkeypatch):
    """replace the hardware id lookup with a stub returning ``'beef'``."""
    import striqt.sensor as ss

    monkeypatch.setattr(
        ss.lib.controller.lookup, 'id', lambda spec, timeout=0.5: 'beef'
    )
    return 'beef'


@pytest.fixture
def array_backend(xp) -> str:
    """return the `Source.array_backend` name that selects the `xp` namespace. The
    cupy case skips with `xp`."""
    return 'cupy' if xp.__name__ == 'cupy' else 'numpy'


@pytest.fixture
def isolated_lookup(monkeypatch):
    """replace the controller registries with empty ones for the test and return the
    `lookup` module."""
    from collections import defaultdict
    from threading import Event

    import striqt.sensor as ss

    lookup = ss.lib.controller.lookup
    for name in ('_obj', '_id', '_ready'):
        monkeypatch.setattr(lookup, name, defaultdict(Event))
    return lookup


@pytest.fixture
def scripted_input(monkeypatch):
    """replace `sa.util.blocking_input` with a stub that pops ``script['answers']``
    and appends each prompt to ``script['prompts']``, and return `script`."""
    import striqt.analysis as sa

    script = {'answers': [], 'prompts': []}

    def blocking_input(prompt=None):
        script['prompts'].append(prompt)
        return script['answers'].pop(0)

    monkeypatch.setattr(sa.util, 'blocking_input', blocking_input)
    return script


@pytest.fixture
def isolated_bindings(monkeypatch):
    """restore the module globals `registry` and `tagged_sweeps` of `ss.lib.bindings`
    after the test and return the module."""
    import striqt.sensor as ss

    bindings = ss.lib.bindings
    monkeypatch.setattr(bindings, 'registry', dict(bindings.registry))
    monkeypatch.setattr(bindings, 'tagged_sweeps', bindings.tagged_sweeps)
    return bindings


@pytest.fixture
def isolated_analysis_registry():
    """restore the live `sa.registry` after a test registers a measurement.

    Teardown restores the dict items and the `names`, `caches`, `coordinates`,
    `signal_trigger`, `parameter_defaults`, and `parameter_fields` attributes, and
    clears every `sw.util.lru_cache`.
    """
    import striqt.analysis as sa
    import striqt.waveform as sw

    registry = sa.registry
    attrs = (
        'names',
        'caches',
        'coordinates',
        'signal_trigger',
        'parameter_defaults',
        'parameter_fields',
    )
    items = dict(registry)
    saved = {name: dict(getattr(registry, name)) for name in attrs if name != 'names'}
    names = set(registry.names)
    yield registry
    registry.clear()
    registry.update(items)
    registry.names.clear()
    registry.names.update(names)
    for name, value in saved.items():
        attr = getattr(registry, name)
        attr.clear()
        attr.update(value)
    sw.util.clear_caches()


@pytest.fixture
def corrections_flags(monkeypatch):
    """override the `USE_OARESAMPLE` and `IGNORE_HIGHSIDE_LO` switches of
    `striqt.sensor.lib.compute.corrections` for one test through the returned
    ``set(use_oaresample=None, ignore_highside_lo=None)``.

    `None` leaves a switch unchanged. Every `sw.util.lru_cache` is cleared on each
    call of `set` and at teardown.
    """
    import striqt.waveform as sw
    from striqt.sensor.lib.compute import corrections

    def set_flags(use_oaresample=None, ignore_highside_lo=None):
        flags = {
            'USE_OARESAMPLE': use_oaresample,
            'IGNORE_HIGHSIDE_LO': ignore_highside_lo,
        }
        sw.util.clear_caches()
        for name, value in flags.items():
            if value is not None:
                monkeypatch.setattr(corrections, name, int(value))

    yield set_flags
    sw.util.clear_caches()


@pytest.fixture(params=('resample', 'oaresample'))
def resample_path(request, corrections_flags):
    """run the test on each resampler path in turn and return the name of the path:
    ``'resample'`` for the whole-record FFT resampler and the FIR, ``'oaresample'``
    for the STFT overlap-add resampler with the low-pass on its bins."""
    corrections_flags(use_oaresample=request.param == 'oaresample')
    return request.param


@pytest.fixture
def armed_tone_controller(isolated_lookup):
    """yield a `single_tone` Controller on the synthetic function source, armed with
    the scale-only preset capture."""
    from sweep_strategies import SOURCE
    from synthetic_sources import SCALE_ONLY, preset_capture

    import striqt.sensor as ss

    with ss.bindings.single_tone.from_source_spec(SOURCE) as ctrl:
        ctrl._arm_spec(preset_capture('single_tone', **SCALE_ONLY))
        yield ctrl


def assert_source_released(lookup, sweep):
    """assert that `lookup` has no controller for the source of `sweep`."""
    import striqt.sensor as ss

    assert not isinstance(lookup._get(sweep.source), ss.lib.controller.Controller)


# %% Spec construction paths


@pytest.fixture(params=('direct', 'from_dict'))
def construct(request):
    """return ``build(cls, **kws)``, a spec constructor on the direct or the
    `from_dict` path."""
    if request.param == 'direct':
        return lambda cls, **kws: cls(**kws)
    return lambda cls, **kws: cls.from_dict(kws)


def raises_on_both_paths(cls, exc, match, **kws):
    """assert that constructing `cls` from `kws` raises `exc` matching `match`, both
    directly and through `from_dict`.

    On the `from_dict` path, `msgspec.ValidationError` is accepted in place of `exc`,
    and nested spec values in `kws` are converted to builtins.
    """
    with pytest.raises(exc, match=match):
        cls(**kws)
    with pytest.raises((exc, msgspec.ValidationError), match=match):
        cls.from_dict(_spec_kws_to_builtins(kws))


def construct_both(cls, **kws):
    """return the (direct, from_dict) instances of `cls` built from `kws`."""
    return cls(**kws), cls.from_dict(_spec_kws_to_builtins(kws))


def _spec_kws_to_builtins(kws: dict) -> dict:
    # nested capture specs have frozendict fields, and msgspec needs the spec enc_hook
    # to encode a frozendict
    import striqt.analysis as sa

    return msgspec.to_builtins(kws, enc_hook=sa.specs.helpers._enc_hook)


# %% Site-style sweeps: extension module binding + per-source overrides

SITE_DIR = SWEEP_DIR / 'site'
SITE_SPEC = SITE_DIR / 'site-cpu.yaml'
SITE_CELLULAR_SPEC = SITE_DIR / 'site-cellular.yaml'


@pytest.fixture(scope='session')
def site_sweep():
    import striqt.sensor as ss

    return ss.read_yaml_spec(SITE_SPEC)


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
    """replace the hardware id lookup with a stub returning `RADIO_ID`, the id keyed
    in sites/radio02.yaml."""
    from site_strategies import RADIO_ID

    import striqt.sensor as ss

    monkeypatch.setattr(
        ss.lib.controller.lookup, 'id', lambda spec, timeout=0.5: RADIO_ID
    )
    return RADIO_ID


@pytest.fixture
def write_yaml(tmp_path):
    """write dedented YAML text to ``tmp_path / relpath`` through the returned
    ``write(relpath, text) -> Path``. Parent directories are created."""
    import textwrap

    def write(relpath: str, text: str) -> Path:
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))
        return path

    return write


@pytest.fixture
def isolated_extension_import():
    """hide the `extensions` module of the test suite for one test.

    `sys.path` and ``sys.modules['extensions']`` are restored at teardown.
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


# %% Fake SoapySDR

FAKE_SOAPY_SPEC = SWEEP_DIR / 'fake_soapy-cpu.yaml'
FAKE_SOAPY_CALIBRATION_SPEC = SWEEP_DIR / 'fake_soapy-calibration-cpu.yaml'
FAKE_SOAPY_CELLULAR_SPEC = SWEEP_DIR / 'fake_soapy-cellular-cpu.yaml'


@pytest.fixture
def fake_soapy(monkeypatch):
    """yield a `FakeSoapySDR` module installed as
    `striqt.sensor.lib.sources.soapy.SoapySDR`."""
    from fake_soapy import install_fake_soapy

    import striqt.waveform as sw

    yield install_fake_soapy(monkeypatch)
    # read_calibration and the lookups cache by path
    sw.util.clear_caches()


@pytest.fixture
def fake_soapy_ext(fake_soapy):
    """return the `fake_soapy_bindings` module, registered by reading
    `fake_soapy-cpu.yaml`."""
    import sys

    import striqt.sensor as ss

    ss.read_yaml_spec(FAKE_SOAPY_SPEC)
    return sys.modules['fake_soapy_bindings']


@pytest.fixture
def fake_controller(fake_soapy_ext):
    """return ``open(**kws)``, a factory of `Controller` context managers for the
    `fake_soapy` binding on rx ports (0, 1).

    `reuse_iq` and `rx_ports` are passed to the controller and every other keyword to
    the source spec.
    """
    import striqt.sensor as ss

    def open_controller(*, reuse_iq=False, rx_ports=(0, 1), **spec_kws):
        ctrl_cls = ss.lib.bindings.get_controller('fake_soapy')
        spec = fake_soapy_ext.FakeSoapySourceSpec(**spec_kws)
        return ctrl_cls.from_source_spec(spec, reuse_iq=reuse_iq, rx_ports=rx_ports)

    return open_controller


@pytest.fixture
def soapy_device(fake_soapy):
    """return one fake device opened through the sequence form of
    `SoapySDR.Device`."""
    return fake_soapy.Device(({},))[0]


@pytest.fixture
def soapy_stream(soapy_device):
    """yield an `RxStream` on `soapy_device` with a default `SoapySource` spec, set up
    and enabled on port 0."""
    from soapy_factories import source_spec

    from striqt.sensor.lib.sources import soapy

    stream = soapy.RxStream(source_spec(), soapy.probe_soapy_info(soapy_device))
    stream.setup(soapy_device, ports=(0,))
    stream.enable(soapy_device, True)
    yield stream


@pytest.fixture
def clear_function_caches():
    """clear the `striqt.waveform` function caches at teardown."""
    import striqt.waveform as sw

    yield
    sw.util.clear_caches()


@pytest.fixture
def calibration_nc(tmp_path, clear_function_caches):
    """return the path of a saved Y-factor calibration with a frequency-dependent
    noise figure on port 0 and a flat noise figure on port 1."""
    from sweep_strategies import save_yfactor_calibration

    nf = {0: {1e9: 5.0, 2e9: 7.0}, 1: 8.0}
    return save_yfactor_calibration(tmp_path / 'cal.nc', nf_dB=nf)


class FakeRun(NamedTuple):
    """the outcome of one sweep against the fake SoapySDR device"""

    path: str
    prompts: list
    sweep: object
    fake: object
    results: object

    def calibration(self, **sel):
        """return the calibration saved at `path`, selected by the `sel` field
        values."""
        import striqt.sensor as ss

        return ss.read_calibration(self.path).sel(**sel)


@pytest.fixture(scope='module')
def fake_sweep_runner(tmp_path_factory):
    """run sweeps against one fake SoapySDR module installed for the whole test
    module, through the returned
    ``run(spec_path_or_spec, *, output_name, **replace) -> FakeRun``.

    The calibration prompts are answered by confirming the ENR and by switching the
    fake noise diode when a prompt asks to enable or disable the diode at a port. The
    source id lookup is stubbed to ``'beef'``.

    `replace` fields are applied to the sweep spec before the run. The sink path is
    replaced by `output_name` under a fresh temporary directory. `FakeRun.prompts`
    lists only the prompts of that run. `FakeRun.results` is the sink output
    concatenated along `capture`, or an empty list when the sink yields nothing.

    A module that uses this fixture must not also use the function-scoped
    `fake_soapy` fixture.
    """
    import re
    from pathlib import Path

    from fake_soapy import install_fake_soapy

    import striqt.analysis as sa
    import striqt.sensor as ss
    import striqt.waveform as sw

    prompts = []

    with pytest.MonkeyPatch.context() as mp:
        fake = install_fake_soapy(mp)

        def blocking_input(prompt=None):
            prompts.append(prompt)
            match = re.match(
                r'(enable|disable) noise diode at port (\d+)', prompt or ''
            )
            if match:
                fake.model.diode_on[int(match.group(2))] = match.group(1) == 'enable'
                return ''
            return 'y'

        mp.setattr(sa.util, 'blocking_input', blocking_input)
        mp.setattr(ss.lib.controller.lookup, 'id', lambda spec, timeout=0.5: 'beef')

        def run(spec, *, output_name, **replace):
            import xarray as xr

            if isinstance(spec, (str, Path)):
                spec_path = spec
                spec = ss.read_yaml_spec(spec_path)
            else:
                spec_path = None
            path = tmp_path_factory.mktemp('fake-sweep') / output_name
            spec = spec.replace(sink=spec.sink.replace(path=str(path)), **replace)
            first_prompt = len(prompts)
            with ss.open_resources(spec, spec_path) as resources:
                results = [
                    ss.lib.compute.from_delayed(dd)
                    if isinstance(dd, ss.lib.compute.DelayedDataset)
                    else dd
                    for dd in ss.iterate_sweep(resources)
                    if dd is not None
                ]
            if results:
                results = xr.concat(results, 'capture')
            return FakeRun(str(path), prompts[first_prompt:], spec, fake, results)

        yield run
    sw.util.clear_caches()


@pytest.fixture(autouse=True)
def restore_logging_state():
    """snapshot and restore the process-global logging state that `show_messages`,
    `log_to_file`, `log_capture_context`, and `StriqtLogger` mutate.

    Only the adapters that existed at setup are restored. An adapter created during
    the test is left in place.
    """
    import logging

    import striqt.analysis as sa

    def snapshot(logger, handler_owner, attr):
        return logger.level, list(logger.handlers), getattr(handler_owner, attr, None)

    adapters = dict(sa.util._logger_adapters)
    saved = {
        name: (*snapshot(adapter.logger, adapter, '_screen_handler'), adapter.extra)
        for name, adapter in adapters.items()
    }
    parent = logging.getLogger('striqt')
    parent_saved = snapshot(parent, parent, '_striqt_handler')

    def restore_handlers(logger, handlers):
        for handler in logger.handlers:
            if handler not in handlers:
                handler.close()
        logger.handlers[:] = handlers

    try:
        yield
    finally:
        for name, (level, handlers, screen, extra) in saved.items():
            adapter = sa.util._logger_adapters[name]
            adapter.logger.setLevel(level)
            restore_handlers(adapter.logger, handlers)
            adapter.extra = extra
            if screen is None:
                adapter.__dict__.pop('_screen_handler', None)
            else:
                adapter._screen_handler = screen
        level, handlers, file_handler = parent_saved
        parent.setLevel(level)
        restore_handlers(parent, handlers)
        if file_handler is None:
            parent.__dict__.pop('_striqt_handler', None)
        else:
            parent._striqt_handler = file_handler


# %% Hardware tests

HARDWARE_ENV = 'STRIQT_TEST_HARDWARE'


def pytest_configure(config):
    # registered here rather than in pyproject.toml because pytest 8 ignores
    # [tool.pytest]
    config.addinivalue_line(
        'markers', f'hardware: needs an attached SDR; opt in with {HARDWARE_ENV}=1'
    )
    config.addinivalue_line(
        'markers',
        'namespaces(*names): array namespaces to parametrize the xp fixture over '
        "(default 'numpy', 'cupy'; 'dask' may be added)",
    )


def pytest_collection_modifyitems(config, items):
    """skip `hardware` tests unless STRIQT_TEST_HARDWARE=1 opts in"""
    import os

    if os.environ.get(HARDWARE_ENV) == '1':
        return
    skip = pytest.mark.skip(reason=f'needs an attached SDR; set {HARDWARE_ENV}=1')
    for item in items:
        if 'hardware' in item.keywords:
            item.add_marker(skip)
