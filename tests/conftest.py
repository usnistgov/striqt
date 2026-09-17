"""shared test resources"""

from __future__ import annotations

import importlib.util
import warnings
from collections import UserDict
from pathlib import Path
from threading import Lock
from typing import NamedTuple

import msgspec
import numpy as np
import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

np.seterr(divide='ignore')

# every property test in the suite runs under these settings: the array fixtures
# below are function scoped, and JIT compilation and GPU transfers make the
# per-example deadline meaningless. differing_executors is suppressed because the
# hypothesis plugin recognizes parametrization only through parametrize marks or
# fixture params, not the pytest_generate_tests call that drives the `xp` fixture
# (hypothesis issue 3733)
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
        try:
            # this needs to happen first or a linking error happens on py39 jetson
            import numba.cuda  # noqa: F401
        except ImportError:
            pass
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

# checked without importing, which would reify scipy
_dask_is_available = importlib.util.find_spec('dask') is not None


def namespace_module(name: str):
    """the array module named 'numpy', 'cupy' or 'dask', skipping the test when cupy
    or dask is unavailable"""
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
    """the cupy module, skipping the test when no CUDA device is available"""
    return namespace_module('cupy')


def pytest_generate_tests(metafunc):
    """parametrize `xp` over the namespaces named by a `namespaces` marker, or numpy
    and cupy by default"""
    if 'xp' in metafunc.fixturenames:
        marker = metafunc.definition.get_closest_marker('namespaces')
        names = marker.args if marker else ('numpy', 'cupy')
        metafunc.parametrize('xp', names, indirect=True)


@pytest.fixture
def xp(request):
    """each array namespace in turn; cupy and dask cases skip when unavailable"""
    return namespace_module(request.param)


def as_xp(xp, *arrays):
    """numpy arrays converted into the namespace `xp`; one array in, one array out,
    otherwise a tuple"""

    def convert(arr):
        if xp is np:
            return arr
        elif xp.__name__.startswith('dask'):
            return xp.from_array(arr, chunks='auto')
        return xp.asarray(arr)

    converted = tuple(convert(a) for a in arrays)
    return converted[0] if len(converted) == 1 else converted


# ---------------------------------------------------------------------------
# Hypothesis strategies for array-based property testing
# ---------------------------------------------------------------------------


def float_dtypes(dtype=None):
    """float32 and float64, or just `dtype` when given"""
    if dtype is None:
        return st.sampled_from([np.float32, np.float64])
    return st.just(dtype)


def _shape_strategy(min_dims, max_dims, min_side, max_side):
    return array_shapes(
        min_dims=min_dims, max_dims=max_dims, min_side=min_side, max_side=max_side
    )


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
        dt = draw(float_dtypes(dtype))
        shape = draw(_shape_strategy(min_dims, max_dims, min_size, max_size))
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
    """(array, axis) pairs, with axis drawn from [-ndim, ndim).

    Negative axes are drawn as often as positive ones so that axis-normalization
    branches are exercised.
    """

    @st.composite
    def _shaped(draw):
        shape = draw(_shape_strategy(min_dims, max_dims, min_side, max_side))
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
        min_blocks = max(min_size // multiple_of, 1)
        max_blocks = max(max_size // multiple_of, 1)
        blocks = draw(st.integers(min_value=min_blocks, max_value=max_blocks))
        size = blocks * multiple_of
        shape = (size,) if channels is None else (channels, size)
        seed = draw(st.integers(min_value=0, max_value=2**16))
        scale = 10 ** (draw(st.integers(*log_power)) / 2)
        return (scale * gaussian_iq(shape, dt, seed)).astype(dt, copy=False)

    return _iq()


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


@pytest.fixture
def run_sweep_to_zarr(tmp_path, fake_source_id):
    """run(spec, name='out.zarr.zip') -> (spec as run, store path) after a sweep end
    to end into tmp_path, without chdir (no spec path is given to open_resources)"""
    import striqt.sensor as ss

    def run(spec, name='out.zarr.zip'):
        path = tmp_path / name
        spec = spec.replace(sink=spec.sink.replace(path=str(path)))
        with ss.open_resources(spec, None) as resources:
            list(ss.iterate_sweep(resources))
        return spec, path

    return run


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
    ValidationError on the conversion path, so either is accepted there. Nested spec
    values in `kws` are converted to builtins for the from_dict path.
    """
    with pytest.raises(exc, match=match):
        cls(**kws)
    with pytest.raises((exc, msgspec.ValidationError), match=match):
        cls.from_dict(_spec_kws_to_builtins(kws))


def construct_both(cls, **kws):
    """(direct, from_dict) instances of `cls` built from the same keywords"""
    return cls(**kws), cls.from_dict(_spec_kws_to_builtins(kws))


def _spec_kws_to_builtins(kws: dict) -> dict:
    # nested capture specs carry frozendict fields, which need the specs' enc_hook
    import striqt.analysis as sa

    return msgspec.to_builtins(kws, enc_hook=sa.specs.helpers._enc_hook)


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


# ---------------------------------------------------------------------------
# Fake SoapySDR
# ---------------------------------------------------------------------------

FAKE_SOAPY_SPEC = SWEEP_DIR / 'fake_soapy-cpu.yaml'
FAKE_SOAPY_CALIBRATION_SPEC = SWEEP_DIR / 'fake_soapy-calibration-cpu.yaml'


@pytest.fixture
def fake_soapy(monkeypatch):
    """a FakeSoapySDR module installed as striqt.sensor.lib.sources.soapy.SoapySDR"""
    from fake_soapy import install_fake_soapy

    import striqt.waveform as sw

    yield install_fake_soapy(monkeypatch)
    # read_calibration and the lookups cache by path
    sw.util.clear_caches()


@pytest.fixture
def fake_soapy_ext(fake_soapy):
    """the fake_soapy_bindings module, registered by reading fake_soapy-cpu.yaml"""
    import sys

    import striqt.sensor as ss

    ss.read_yaml_spec(FAKE_SOAPY_SPEC)
    return sys.modules['fake_soapy_bindings']


def answer_calibration_prompts(monkeypatch, fake) -> list[str]:
    """stub blocking_input to answer the calibration prompts: confirm the ENR, and
    switch the fake noise diode on 'enable|disable noise diode at port N'. Returns
    the live prompt log."""
    import re

    import striqt.analysis as sa

    prompts = []

    def blocking_input(prompt=None):
        prompts.append(prompt)
        match = re.match(r'(enable|disable) noise diode at port (\d+)', prompt or '')
        if match:
            fake.model.diode_on[int(match.group(2))] = match.group(1) == 'enable'
            return ''
        return 'y'

    monkeypatch.setattr(sa.util, 'blocking_input', blocking_input)
    return prompts


@pytest.fixture
def answer_prompts(monkeypatch, fake_soapy):
    """answer the calibration prompts against the function-scoped fake; returns the
    prompt log"""
    return answer_calibration_prompts(monkeypatch, fake_soapy)


@pytest.fixture
def soapy_device(fake_soapy):
    """one fake device opened through the sequence form of SoapySDR.Device"""
    return fake_soapy.Device(({},))[0]


@pytest.fixture
def soapy_stream(soapy_device):
    """an RxStream on `soapy_device` with a default SoapySource spec, set up and
    enabled on port 0"""
    from soapy_factories import source_spec

    from striqt.sensor.lib.sources import soapy

    stream = soapy.RxStream(source_spec(), soapy.probe_soapy_info(soapy_device))
    stream.setup(soapy_device, ports=(0,))
    stream.enable(soapy_device, True)
    yield stream


@pytest.fixture
def clear_function_caches():
    """clear the striqt.waveform function caches after the test (read_calibration and
    the lookups cache by path)"""
    import striqt.waveform as sw

    yield
    sw.util.clear_caches()


@pytest.fixture
def calibration_nc(tmp_path, clear_function_caches):
    """path of a saved Y-factor calibration with a frequency-dependent noise figure on
    port 0 and a flat one on port 1"""
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
        """the calibration saved at `path`, selected by field values"""
        import striqt.sensor as ss

        return ss.read_calibration(self.path).sel(**sel)


@pytest.fixture(scope='module')
def fake_sweep_runner(tmp_path_factory):
    """run(spec_path_or_spec, *, output_name, **replace) -> FakeRun against one fake
    SoapySDR module installed for the whole test module, with the calibration prompts
    answered and the source id lookup stubbed to 'beef'.

    `replace` fields are applied to the sweep spec before it runs; the sink path is
    replaced by `output_name` under a fresh temporary directory. `FakeRun.prompts`
    holds only the prompts of that run and `FakeRun.results` the sink output
    concatenated along `capture` (an empty list when the sink yields nothing).

    A module that uses this fixture must not also use the function-scoped
    `fake_soapy` fixture: the second install_fake_soapy would shadow the first.
    """
    from pathlib import Path

    from fake_soapy import install_fake_soapy

    import striqt.sensor as ss
    import striqt.waveform as sw

    with pytest.MonkeyPatch.context() as mp:
        fake = install_fake_soapy(mp)
        prompts = answer_calibration_prompts(mp, fake)
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
                results = [ds for ds in ss.iterate_sweep(resources) if ds is not None]
            if results:
                results = xr.concat(results, 'capture')
            return FakeRun(str(path), prompts[first_prompt:], spec, fake, results)

        yield run
    sw.util.clear_caches()


@pytest.fixture(autouse=True)
def restore_logging_state():
    """snapshot and restore the process-global logging state that show_messages,
    log_to_file, log_capture_context and StriqtLogger mutate.

    Only the adapters that existed at setup are restored. Adapters created during
    the test are left as they are: they come from the lazy import of a striqt
    package, and the modules that import them at module level would otherwise
    find their adapter deleted.
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


# ---------------------------------------------------------------------------
# Hardware tests
# ---------------------------------------------------------------------------

HARDWARE_ENV = 'STRIQT_TEST_HARDWARE'


def pytest_configure(config):
    # registered here rather than in pyproject.toml: pytest 8 ignores [tool.pytest]
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
