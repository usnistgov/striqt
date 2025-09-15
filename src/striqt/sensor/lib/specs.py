"""data structures that specify operation of radio hardware, captures, and sweeps"""

from __future__ import annotations
import itertools
import functools
import numbers
import typing
from typing import Annotated, Optional, Literal, Any, Union

import msgspec

from . import util

from striqt import analysis
from striqt.analysis.lib.specs import meta, SpecBase, _SlowHashSpecBase

if typing.TYPE_CHECKING:
    import pandas as pd
else:
    # to resolve the 'pd.Timestamp' stub at runtime
    pd = util.lazy_import('pandas')

SingleChannelType = Annotated[int, meta('Input port index', ge=0)]
SingleGainType = Annotated[float, meta('Gain setting', 'dB')]
CenterFrequencyType = Annotated[float, meta('RF center frequency', 'Hz', gt=0)]
PortType = Annotated[
    Union[SingleChannelType, tuple[SingleChannelType, ...]],
    meta('Input port indices'),
]
GainType = Annotated[
    Union[SingleGainType, tuple[SingleGainType, ...]],
    meta('Gain setting for each channel', 'dB'),
]
LOShiftType = Annotated[Literal['left', 'right', 'none'], meta('LO shift direction')]
DelayType = Annotated[float, meta('Delay in acquisition start time', 's', gt=0)]
StartTimeType = Annotated['pd.Timestamp', meta('Acquisition start time')]
BackendSampleRateType = Annotated[
    float, meta('Force the specified sample rate in the source', 'Hz', gt=0)
]


def _dict_hash(d):
    key_hash = frozenset(d.keys())
    value_hash = tuple(
        [_dict_hash(v) if isinstance(v, dict) else v for v in d.values()]
    )
    return hash(key_hash) ^ hash(value_hash)


AnalysisBandwidthType = Annotated[
    float, meta('Bandwidth of the analysis filter (or inf to disable)', 'Hz', gt=0)
]


@util.lru_cache()
def _validate_multichannel(port, gain):
    """guarantee that self.gain is a number or matches the length of self.port"""
    if isinstance(port, numbers.Number):
        if isinstance(gain, tuple):
            raise ValueError(
                'gain must be a single number unless multiple ports are specified'
            )
    else:
        if isinstance(gain, tuple) and len(gain) != len(port):
            raise ValueError('gain, when specified as a tuple, must match port count')


class WaveformCapture(
    analysis.Capture, forbid_unknown_fields=True, frozen=True, cache_hash=True
):
    """Capture specification structure for a generic waveform.

    This subset of RadioCapture is broken out here to simplify the evaluation of
    sampling parameters that are independent from other radio parameters.
    """

    # acquisition
    duration: Annotated[float, meta('Acquisition duration', 's', gt=0)] = 0.1
    sample_rate: Annotated[float, meta('Sample rate', 'S/s', gt=0)] = 15.36e6

    # filtering and resampling
    analysis_bandwidth: AnalysisBandwidthType = float('inf')
    lo_shift: LOShiftType = 'none'
    host_resample: bool = True
    backend_sample_rate: Optional[BackendSampleRateType] = None


class _WaveformCaptureKeywords(typing.TypedDict, total=False):
    # this needs to be kept in sync with WaveformCapture in order to
    # properly provide type hints for IDEs in the arm and acquire
    # call signatures of source.Base objects
    duration: Annotated[float, meta('Acquisition duration', 's', gt=0)]
    sample_rate: Annotated[float, meta('Sample rate', 'S/s', gt=0)]

    # filtering and resampling
    analysis_bandwidth: AnalysisBandwidthType
    lo_shift: LOShiftType
    host_resample: bool
    backend_sample_rate: Optional[BackendSampleRateType]


class RadioCapture(
    WaveformCapture, forbid_unknown_fields=True, frozen=True, cache_hash=True
):
    """Capture specification for a single radio waveform"""

    # RF and leveling
    center_frequency: CenterFrequencyType = 3710e6
    port: PortType = 0
    gain: GainType = -10

    delay: Optional[DelayType] = None
    start_time: Optional[StartTimeType] = None

    # a counter used to reset the sweep timestamp on Repeat(None)
    _sweep_index: int = 0

    def __post_init__(self):
        _validate_multichannel(self.port, self.gain)


class _RadioCaptureKeywords(_WaveformCaptureKeywords, total=False):
    # this needs to be kept in sync with WaveformCapture in order to
    # properly provide type hints for IDEs in the arm and acquire
    # call signatures of source.Base objects
    center_frequency: CenterFrequencyType
    port: PortType
    gain: GainType
    delay: Optional[DelayType]
    start_time: Optional[StartTimeType]


class FileSourceCapture(RadioCapture, forbid_unknown_fields=True, cache_hash=True):
    """Capture specification read from a file, with support for None sentinels"""

    # RF and leveling
    center_frequency: Optional[CenterFrequencyType] = float('nan')
    port: Optional[PortType] = 0
    gain: Optional[GainType] = float('nan')
    backend_sample_rate: Optional[float] = float('nan')


class LoopBase(
    SpecBase,
    tag=str.lower,
    tag_field='kind',
    forbid_unknown_fields=False,
    frozen=True,
    kw_only=True,
):
    field: str

    def get_points(self):
        raise NotImplementedError


class Range(LoopBase, forbid_unknown_fields=True, frozen=True, kw_only=True):
    start: float
    stop: float
    step: float

    def get_points(self):
        import numpy as np

        if self.start == self.stop:
            return np.array([self.start])

        a = np.arange(self.start, self.stop + self.step / 2, self.step)
        return list(a)


class Repeat(LoopBase, forbid_unknown_fields=True, frozen=True, kw_only=True):
    field: Optional[str] = '_sweep_index'
    count: int = 1

    def get_points(self):
        return list(range(self.count))


class List(LoopBase, forbid_unknown_fields=True, frozen=True, kw_only=True):
    values: tuple[typing.Any, ...]

    def get_points(self):
        return self.values


class FrequencyBinRange(
    LoopBase, forbid_unknown_fields=True, frozen=True, kw_only=True
):
    start: float
    stop: float
    step: float

    def get_points(self):
        from math import ceil
        import numpy as np

        span = self.stop - self.start
        count = ceil(span / self.step)
        expanded_span = count * self.step
        points = np.linspace(-expanded_span / 2, expanded_span / 2, count + 1)
        if points[0] < self.start:
            points = points[1:]
        if points[-1] > self.stop:
            points = points[:-1]
        return list(points)


LoopSpecifier = typing.Union[Repeat, List, Range, FrequencyBinRange]


TimeSourceType = Annotated[
    Literal['host', 'internal', 'external', 'gps'],
    meta('Hardware source for timestamps'),
]

ClockSourceType = Annotated[
    Literal['internal', 'external', 'gps'],
    meta('Hardware source for the frequency reference'),
]

ContinuousTriggerType = Annotated[
    bool,
    meta('Whether to trigger immediately after each call to acquire() when armed'),
]

GaplessRepeatType = Annotated[
    bool,
    meta('whether to raise an exception on overflows between identical captures'),
]

TimeSyncEveryCaptureType = Annotated[
    bool, meta('whether to sync to PPS before each capture in a sweep')
]

WarmupSweepType = Annotated[
    bool,
    meta(
        'whether to run the GPU compute on empty buffers before sweeping for more even run time'
    ),
]

ArrayBackendType = Annotated[
    Union[Literal['numpy'], Literal['cupy']],
    meta(
        'array module to use, which sets the type of compute device (numpy = cpu, cupy = gpu)'
    ),
]

FastLOType = Annotated[
    bool,
    meta(
        'if False, permit the radio to use slower frequency changes/channel enables to improve LO spurs'
    ),
]

SyncSourceType = Annotated[
    str,
    meta(
        'name of a registered waveform sync function for analysis-based IQ synchronization'
    ),
]

SyncSourceType = Annotated[
    str,
    meta(
        'name of a registered waveform sync function for analysis-based IQ synchronization'
    ),
]


ReceiveRetriesType = Annotated[
    int,
    meta(
        'number of attempts to retry acquisition on a stream error',
        ge=0,
    ),
]


class RadioSetup(SpecBase, forbid_unknown_fields=True, frozen=True, cache_hash=True):
    """run-time characteristics of the radio that are left invariant during a sweep"""

    driver: Optional[str] = None
    resource: dict = {}
    calibration: Optional[str] = None

    # sequencing
    gapless_repeats: GaplessRepeatType = False
    warmup_sweep: WarmupSweepType = True
    receive_retries: ReceiveRetriesType = 0

    # synchronization and triggering
    time_source: TimeSourceType = 'host'
    time_sync_every_capture: TimeSyncEveryCaptureType = False
    channel_sync_source: Optional[str] = None
    clock_source: ClockSourceType = 'internal'
    periodic_trigger: Optional[float] = None

    # in the future, these should probably move to an analysis config
    array_backend: ArrayBackendType = 'cupy'
    cupy_max_fft_chunk_size: Optional[int] = None

    # validation data
    uncalibrated_peak_detect: Union[bool, typing.Literal['auto']] = 'auto'

    # calibration subclasses set this True to skip unecessary
    # re-acquisitions
    reuse_iq = False

    _transient_holdoff_time: Optional[float] = None
    _rx_port_count: Optional[int] = None

    def __post_init__(self):
        if not self.gapless_repeats:
            pass
        elif self.time_sync_every_capture:
            raise ValueError(
                'time_sync_every_capture and gapless_repeats are mutually exclusive'
            )
        elif self.receive_retries > 0:
            raise ValueError(
                'receive_retries must be 0 when gapless_repeats is enabled'
            )

        if self.channel_sync_source is None:
            pass
        elif self.channel_sync_source not in analysis.register.channel_sync_source:
            registered = set(analysis.register.channel_sync_source)
            raise ValueError(
                f'channel_sync_source "{self.channel_sync_source!r}" is not one of the registered functions {registered!r}'
            )


class _RadioSetupKeywords(typing.TypedDict, total=False):
    # this needs to be kept in sync with WaveformCapture in order to
    # properly provide type hints for IDEs in the setup
    # call signature of source.Base objects

    driver: typing.NotRequired[Optional[str]]
    resource: typing.NotRequired[dict]
    calibration: typing.NotRequired[Optional[str]]

    gapless_repeats: typing.NotRequired[GaplessRepeatType]
    warmup_sweep: typing.NotRequired[WarmupSweepType]
    receive_retries: typing.NotRequired[ReceiveRetriesType]

    time_source: TimeSourceType
    clock_source: ClockSourceType
    periodic_trigger: Optional[float]
    time_sync_every_capture: TimeSyncEveryCaptureType
    channel_sync_source: Optional[str] = None

    array_backend: ArrayBackendType
    cupy_max_fft_chunk_size: int


class Description(SpecBase, forbid_unknown_fields=True, frozen=True, cache_hash=True):
    summary: Optional[str] = None
    location: Optional[tuple[float, float, float]] = None
    signal_chain: Optional[tuple[str, ...]] = tuple()
    version: str = 'unversioned'


AliasMatchType = Annotated[
    typing.Union[dict[str, Any], tuple[dict[str, Any], ...]],
    meta('one or more dictionaries of valid match sets to "or"'),
]


class Output(
    _SlowHashSpecBase, forbid_unknown_fields=True, frozen=True, cache_hash=True
):
    path: Optional[str] = '{yaml_name}-{start_time}'
    log_path: Optional[str] = None
    log_level: str = 'info'
    store: typing.Union[Literal['zip'], Literal['directory']] = 'directory'
    coord_aliases: dict[str, dict[str, AliasMatchType]] = {}
    max_threads: Optional[int] = None


SweepStructType = Annotated[
    typing.Union[str, Literal['striqt.sensor.Sweep']],
    meta(
        'striqt.sensor.Sweep subclass to import to decode the sweep specification structure'
    ),
]
SinkClassType = Annotated[
    typing.Union[str, Literal['striqt.sensor.writers.CaptureAppender']],
    meta('data sink class to import and use'),
]
PeripheralClassType = Annotated[
    typing.Union[str, Literal['striqt.sensor.peripherals.NoPeripherals']],
    meta('peripheral manager class for import'),
]
ExtensionPathType = Annotated[
    Optional[str],
    meta(
        'optional import path to add (if relative, specified with regard to this yaml)'
    ),
]


class Extensions(SpecBase, forbid_unknown_fields=True, frozen=True, cache_hash=True):
    peripherals: PeripheralClassType = 'striqt.sensor.peripherals.NoPeripherals'
    sink: SinkClassType = 'striqt.sensor.sinks.CaptureAppender'
    sweep_struct: SweepStructType = 'striqt.sensor.Sweep'
    import_path: ExtensionPathType = None


# dynamically generate Analysis type for "built-in" measurements in to striqt.analysis
BundledAnalysis = analysis.register.to_analysis_spec_type(analysis.register.measurement)
BundledAlignmentAnalysis = analysis.register.to_analysis_spec_type(
    analysis.register.channel_sync_source
)

WindowFillType = Annotated[
    float,
    meta(
        'size of the averaging window as a fraction of the analysis interval',
        ge=0,
        le=1,
    ),
]


@functools.lru_cache(2)
def _expand_loops(
    explicit_captures: tuple[RadioCapture, ...], loops: tuple[LoopSpecifier, ...]
) -> tuple[RadioCapture, ...]:
    """evaluate the loop specification, and flatten into one list of loops"""
    fields = tuple([loop.field for loop in loops])

    for f in fields:
        if f is None:
            continue
        if f not in explicit_captures[0].__struct_fields__:
            raise TypeError(f'loop specifies capture field {f!r} that is not defined')

    loop_points = [loop.get_points() for loop in loops]
    combinations = itertools.product(*loop_points)

    result = []
    for values in combinations:
        updates = dict(zip(fields, values))

        extras = [c.replace(**updates) for c in explicit_captures]

        result += extras

    if len(result) == 0:
        return explicit_captures
    else:
        return tuple(result)


class Sweep(SpecBase, forbid_unknown_fields=True, frozen=True, cache_hash=True):
    captures: tuple[RadioCapture, ...] = tuple()
    radio_setup: RadioSetup = RadioSetup()
    defaults: RadioCapture = RadioCapture()
    loops: tuple[LoopSpecifier, ...] = ()

    analysis: BundledAnalysis = BundledAnalysis()  # type: ignore
    description: typing.Union[Description, str] = ''
    extensions: Extensions = Extensions()
    output: Output = Output()

    def get_captures(self, looped=True):
        """subclasses may use this to autogenerate capture sequences"""
        explicit_captures = object.__getattribute__(self, 'captures')
        if looped:
            return _expand_loops(tuple(explicit_captures), self.loops)
        else:
            return explicit_captures

    def __getattribute__(self, name):
        if name == 'captures':
            return self.get_captures()
        else:
            return super().__getattribute__(name)

    def __post_init__(self):
        looped_fields = []
        for loop in self.loops:
            if loop.field in looped_fields:
                raise TypeError(
                    f'multiple loops specified for capture field {repr(loop.field)}'
                )
            else:
                looped_fields.append(loop.field)

    @classmethod
    def _from_registry(cls: type[Sweep]) -> type[Sweep]:
        bases = typing.get_type_hints(cls, include_extras=True)

        Analysis = analysis.register.to_analysis_spec_type(
            analysis.register.measurement,
            base=bases[cls.analysis.__name__],
        )

        fields = ((cls.analysis.__name__, Analysis, Analysis()),)

        return msgspec.defstruct(
            cls.__name__,
            fields,
            bases=(cls,),
            frozen=True,
            forbid_unknown_fields=True,
            cache_hash=True,
        )
