"""Shared fixtures and timing helpers of the profiling scripts.

The fixtures reproduce the aggregate-directivity-acquisition fragments ``denver-3750``
(capture), ``7201b`` (source), and ``cellular`` (analysis): 107.52 MS/s resampled from
the 125 MHz master clock, 2 ports, 100 MHz analysis bandwidth, 0.1 s, 12 measurements,
and the ``cellular_5g_pss_sync`` trigger. Nothing under ``src/`` is modified.

The scripts share these conventions. A timed row is the median of ``--repeat`` calls
after ``--warmup`` untimed calls. ``host`` is the wall time until the call returns.
``device`` is the span between CUDA events recorded around the call, equal to ``host``
on numpy. ``pool peak`` is the high-water mark of the cupy memory pool during one
further call, above the bytes in use before that call, sampled by a memory hook at
every allocation; the arrays that exist before the call are not counted. The pool is
not freed between the calls of one row.

On the Tegra X2 the cupy import fails with a ``GLIBCXX`` version error unless the
system libstdc++ is preloaded (``LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libstdc++.so.6``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import math
import platform
import resource
import statistics
import sys
import time
from collections.abc import Callable, Generator
from fractions import Fraction
from types import ModuleType
from typing import Any, cast

import numpy as np

import striqt.analysis as sa
import striqt.sensor as ss
import striqt.waveform as sw

cp: Any = None
"""the cupy module after `select_backend('cupy')`, None on numpy"""

# fragments/captures/denver-3750.yaml
CAPTURE = {
    'center_frequency': 3750e6,
    'analysis_bandwidth': 100e6,
    'sample_rate': 107.52e6,
    'port': (0, 1),
    'gain': -10,
    'duration': 0.1,
    'adjust_analysis': {
        'frequency_offset': -40.08e6,
        'guard_bandwidths': (875e3, 575e3),
        'frame_slots': 'dddsuudddddddsuudddd',
        'special_symbols': 'ddddddfffffffu',
    },
}

# fragments/analysis/cellular.yaml, with the YAML anchors expanded
_SPECTROGRAM = {
    'frequency_resolution': 15000,
    'fractional_overlap': Fraction(13, 28),
    'window_fill': Fraction(15, 28),
    'integration_bandwidth': 360e3,
    'window': ('kaiser', 11.884),
    'lo_bandstop': 120e3,
}
_SCS = 30000
_POWER_DETECTORS = ('rms', 'peak')
_SYMBOL_PERIOD = Fraction(1, 28000)
ANALYSIS = {
    'spectrogram': _SPECTROGRAM,
    'cellular_resource_power_histogram': {
        'subcarrier_spacing': _SCS,
        'window': _SPECTROGRAM['window'],
        'power_low': -150,
        'power_high': 0,
        'power_resolution': 0.1,
        'average_rbs': True,
        'average_slots': False,
        'cyclic_prefix': 'normal',
        'lo_bandstop': _SPECTROGRAM['lo_bandstop'],
        'frame_slots': 'dddsuudddddddsuudddd',
        'special_symbols': 'ddddddfffffffu',
        'guard_bandwidths': (875e3, 875e3),
    },
    'cellular_5g_pss_correlation': {
        'subcarrier_spacing': _SCS,
        'delay': 0,
        'discovery_periodicity': 20e-3,
        'frequency_offset': 0,
        'symbol_indexes': 'c',
    },
    'cellular_5g_pss_sync': {
        'subcarrier_spacing': _SCS,
        'delay': 0,
        'symbol_indexes': 'c',
        'discovery_periodicity': 20e-3,
        'per_port': True,
        'max_beams': 4,
    },
    'cellular_5g_ssb_spectrogram': {
        'frequency_offset': 0,
        'subcarrier_spacing': _SCS,
        'lo_bandstop': _SPECTROGRAM['lo_bandstop'],
        'window': _SPECTROGRAM['window'],
    },
    'power_spectral_density': dict(
        _SPECTROGRAM, time_statistic=('mean', 0.5, 0.9, 0.95, 0.99, 'max')
    ),
    'spectrogram_ratio_histogram': dict(
        _SPECTROGRAM, power_low=-50, power_high=50, power_resolution=0.1
    ),
    'spectrogram_histogram': dict(
        _SPECTROGRAM, power_low=-150, power_high=0, power_resolution=0.1
    ),
    'cellular_cyclic_autocorrelation': {
        'subcarrier_spacings': (_SCS,),
        'frame_slots': 'dddsuudddddddsuudddd',
        'frame_range': (0, 1),
    },
    'cyclic_channel_power': {
        'cyclic_period': 0.01,
        'cyclic_statistics': ('min', 'mean', 'max'),
        'detector_period': _SYMBOL_PERIOD,
        'power_detectors': _POWER_DETECTORS,
    },
    'channel_power_histogram': {
        'detector_period': _SYMBOL_PERIOD,
        'power_detectors': _POWER_DETECTORS,
        'power_low': -100,
        'power_high': 5,
        'power_resolution': 0.1,
    },
    'channel_power_time_series': {
        'detector_period': 0.01,
        'power_detectors': _POWER_DETECTORS,
    },
}


# %% Backend selection
def select_backend(name: str) -> tuple[ModuleType, str]:
    """return the array namespace and the device name selected by `name`.

    The banner line of `print_banner` is printed.

    Args:
        name: ``'cupy'`` imports cupy, configures it through
            `striqt.waveform.arrays.configure_cupy`, and sets the module `cp`;
            ``'numpy'`` leaves `cp` None

    Returns:
        the array namespace (`cupy` or `numpy`) and the CUDA device name, ``'none'``
        on numpy
    """
    global cp
    if name == 'cupy':
        import cupy  # ty: ignore[unresolved-import]

        cp = cupy
        sw.arrays.configure_cupy()
        xp: ModuleType = cupy
        device = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    elif name == 'numpy':
        xp = np
        device = 'none'
    else:
        raise ValueError(f'unknown backend {name!r}')
    print_banner(xp, device)
    return xp, device


def print_banner(xp: ModuleType, device: str) -> None:
    """print the machine, the numpy and cupy versions, and the device name."""
    cupy_version = '-' if xp is np else xp.__version__
    print(
        f'{platform.machine()} numpy {np.__version__} cupy {cupy_version} '
        f'device: {device}'
    )


# %% Timing and memory
def device_synchronize() -> None:
    """block until the device is idle; nothing on numpy."""
    if cp is not None:
        cp.cuda.Device().synchronize()


def time_call(func: Callable[[], Any]) -> tuple[float, float, Any]:
    """return the (host, device) duration (in s) of one call and the value it returns.

    The host time ends when the call returns. The device time is the span between
    CUDA events recorded before and after the call, equal to the host time on numpy.
    """
    if cp is None:
        t0 = time.perf_counter()
        ret = func()
        dt = time.perf_counter() - t0
        return dt, dt, ret
    start, end = cp.cuda.Event(), cp.cuda.Event()
    device_synchronize()
    start.record()
    t0 = time.perf_counter()
    ret = func()
    host = time.perf_counter() - t0
    end.record()
    end.synchronize()
    return host, float(cp.cuda.get_elapsed_time(start, end)) / 1e3, ret


def repeat_call(
    func: Callable[[], Any], repeat: int, warmup: int
) -> tuple[float, float, Any]:
    """return the median (host, device) durations (in s) over `repeat` timed calls
    and the value of the last call.

    `warmup` untimed calls run first.
    """
    for _ in range(warmup):
        func()
    hosts, devices = [], []
    ret = None
    for _ in range(repeat):
        host, device, ret = time_call(func)
        hosts.append(host)
        devices.append(device)
    return statistics.median(hosts), statistics.median(devices), ret


def peak_pool_bytes(func: Callable[[], Any]) -> int:
    """return the high-water mark (in bytes) of the cupy pool during one call of
    `func`, above the bytes in use before the call; 0 on numpy.

    The mark is sampled at every allocation by a `cupy.cuda.MemoryHook`.
    """
    if cp is None:
        func()
        return 0
    pool = cp.get_default_memory_pool()
    base = pool.used_bytes()
    peak = [base]

    class Hook(cp.cuda.MemoryHook):
        name = 'profiling-peak'

        def malloc_postprocess(self, **_: Any) -> None:
            peak[0] = max(peak[0], pool.used_bytes())

    with Hook():
        func()
    device_synchronize()
    return peak[0] - base


def pool_used_bytes() -> int:
    """return the bytes of the cupy pool in use by live arrays; 0 on numpy."""
    if cp is None:
        return 0
    return int(cp.get_default_memory_pool().used_bytes())


def pool_total_bytes() -> int:
    """return the bytes the cupy pool holds from the driver, in use or free; 0 on
    numpy."""
    if cp is None:
        return 0
    return int(cp.get_default_memory_pool().total_bytes())


def free_pool() -> None:
    """release the free blocks of the cupy pool to the driver; nothing on numpy."""
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()


def rss_bytes() -> int:
    """return the resident set of the process (in bytes).

    Linux reads the current value from ``/proc/self/statm``; elsewhere the value is
    the high-water mark that `resource.getrusage` reports.
    """
    try:
        with open('/proc/self/statm') as fd:
            pages = int(fd.read().split()[1])
        return pages * resource.getpagesize()
    except (OSError, IndexError, ValueError):
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is in kilobytes on Linux and in bytes on macOS
        return usage if sys.platform == 'darwin' else usage * 1024


@contextlib.contextmanager
def nvtx_range(name: str) -> Generator[None, None, None]:
    """mark the enclosed block as an NVTX range named `name` for ``nsys``; nothing
    without cupy or without NVTX."""
    nvtx: Any = getattr(cp.cuda, 'nvtx', None) if cp is not None else None
    pushed = False
    if nvtx is not None:
        try:
            nvtx.RangePush(name)
            pushed = True
        except (RuntimeError, AttributeError):
            pushed = False
    try:
        yield
    finally:
        if pushed:
            nvtx.RangePop()


class StopwatchCollector(logging.Handler):
    """a logging handler that sums the `stopwatch` time of each named block while
    `enabled` is True"""

    def __init__(self) -> None:
        super().__init__(level=sa.util.PERFORMANCE_DETAIL)
        self.times: dict[str, float] = {}
        """the summed time (in s) of each block, by `stopwatch` name"""
        self.enabled = False
        """False drops every record"""

    def emit(self, record: logging.LogRecord) -> None:
        # `stopwatch` passes the timing dict as the message argument of the record
        fields: dict[str, Any] = (
            record.args if isinstance(record.args, dict) else vars(record)
        )
        name = fields.get('stopwatch_name')
        if not self.enabled or not isinstance(name, str):
            return
        self.times[name] = self.times.get(name, 0.0) + float(fields['stopwatch_time'])


def attach_stopwatch_collector() -> StopwatchCollector:
    """return a `StopwatchCollector` added to the ``analysis`` logger, with the logger
    level lowered to `PERFORMANCE_DETAIL`."""
    collector = StopwatchCollector()
    logger = sa.util.get_logger('analysis').logger
    logger.setLevel(min(logger.level, sa.util.PERFORMANCE_DETAIL))
    logger.addHandler(collector)
    return collector


# %% ADA specs and the input buffer
@dataclasses.dataclass(frozen=True)
class Setup:
    capture: ss.specs.SoapyCapture
    """the ADA 3750 MHz capture, with `duration` from the command line"""
    source: ss.specs.Source
    """the air7201b source spec, with `signal_trigger` and `array_backend` set"""
    analysis: sa.specs.AnalysisGroup
    """the cellular analysis after `adjust_analysis` of the capture"""
    trigger: sa.Trigger | None
    """the trigger that `attach_corrections` installs, or None"""


def build_setup(duration: float, backend: str, trigger: bool) -> Setup:
    """return the ADA capture, source, and analysis specs.

    Args:
        duration: the capture duration (in s)
        backend: the `array_backend` of the source spec, ``'numpy'`` or ``'cupy'``
        trigger: True sets the ``cellular_5g_pss_sync`` signal trigger with a 20 ms
            strobe on the source
    """
    binding = ss.bindings.air7201b
    capture = binding.schema.capture(**dict(CAPTURE, duration=duration))
    source = binding.schema.source(
        signal_trigger='cellular_5g_pss_sync' if trigger else None,
        trigger_strobe=20e-3,
        array_backend=backend,
    )
    analysis = ss.specs.BundledAnalysis.from_dict(ANALYSIS)
    analysis = ss.specs.sequencing.adjust_analysis(analysis, capture.adjust_analysis)
    trig = ss.lib.compute.get_trigger_from_spec(source, analysis)
    return Setup(capture, source, analysis, trig)


def source_sample_count(setup: Setup) -> int:
    """return the length of the acquisition buffer at the source rate.

    The count is the lead overlap, the record of `duration` through the exact
    resampler ratio, and the tail overlap, as the controller acquires the buffer.
    """
    capture, source = setup.capture, setup.source
    design = ss.lib.compute.design_resampler(capture, source.master_clock_rate)
    samples_out = round(capture.duration * capture.sample_rate)
    exact = Fraction(samples_out * design['nfft'], design['nfft_out'])
    record = math.floor(exact + Fraction(1, 2))
    lead, tail = ss.lib.compute.get_correction_overlaps(capture, source, setup.analysis)
    return lead + record + tail


def build_raw(setup: Setup, xp: ModuleType) -> ss.specs.AcquiredIQ:
    """return an acquisition of ones at the source rate, with the production overlaps."""
    count = source_sample_count(setup)
    port = setup.capture.port
    ports = len(port) if isinstance(port, tuple) else 1
    x = xp.ones((ports, count), dtype=xp.complex64)
    return ss.specs.AcquiredIQ(
        x=x,
        capture=setup.capture,
        info=ss.specs.AcquisitionInfo(),
        extra_data={},
        source_spec=setup.source,
        # the air7201b source without a calibration file
        power_units='FS',
        resampler=ss.lib.compute.design_resampler(
            setup.capture, setup.source.master_clock_rate
        ),
    )


def oaresample_kws(raw: ss.specs.AcquiredIQ) -> dict[str, Any]:
    """return the keyword arguments of the `sw.oaresample` call in the resampling
    stage of `raw`, other than `x`, `overwrite_x`, `out`, and `scale`."""
    design = cast(Any, raw.resampler)
    capture = cast(Any, raw.capture)
    return {
        'up': design['nfft_out'],
        'down': design['nfft'],
        'fs': design['fs_sdr'],
        'window': design['window'],
        'axis': 1,
        'frequency_shift': design['lo_offset'],
        'filter_bandwidth': capture.analysis_bandwidth,
        'transition_bandwidth': ss.lib.compute.corrections.FIR_TRANSITION_BW,
    }


def measurement_names(analysis: sa.specs.AnalysisGroup) -> list[str]:
    """return the names of the enabled measurements of `analysis`, in field order."""
    return [name for name, kws in analysis.to_dict().items() if kws]


# %% Reporting
def to_numpy(a: Any) -> np.ndarray:
    """return `a` as a numpy array, copied from the device under cupy."""
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def fmt_mb(nbytes: int) -> str:
    """return `nbytes` as whole MiB with the unit."""
    return f'{nbytes / 2**20:.0f} MB'


def print_row(name: str, host: float, device: float, nbytes: int, extra: str) -> None:
    """print one row of a stage table: `name`, the host and device times (in ms), the
    size of the returned array (in MiB), and `extra`."""
    print(
        f'{name:<26} {host * 1e3:9.2f} {device * 1e3:9.2f} {nbytes / 2**20:9.1f}  {extra}'
    )


def print_table_header(name_width: int = 26) -> None:
    """print the column labels of a stage table printed by `print_row`."""
    print(f'{"row":<{name_width}} {"host ms":>9} {"device ms":>9} {"MB":>9}')
