"""data structures that specify operation of radio hardware, captures, and sweeps"""

from __future__ import annotations

import functools
import itertools
import numbers
import typing
from typing import (
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    Optional,
    TypedDict,
    Union,
)

import msgspec

from striqt import analysis
from striqt.analysis.lib.specs import SpecBase, meta

from . import util

if typing.TYPE_CHECKING:
    import pandas as pd

else:
    # to resolve the 'pd.Timestamp' stub at runtime
    pd = util.lazy_import('pandas')

_TC = typing.TypeVar('_TC', bound='CaptureSpec')
_TS = typing.TypeVar('_TS', bound='SourceSpec')


kws = dict(
    forbid_unknown_fields=True,
    cache_hash=True,
)


def _dict_hash(d):
    key_hash = frozenset(d.keys())
    value_hash = tuple(
        [_dict_hash(v) if isinstance(v, dict) else v for v in d.values()]
    )
    return hash(key_hash) ^ hash(value_hash)


@util.lru_cache()
def _validate_multichannel(port, gain):
    """guarantee that self.gain is a number, or matches the length of self.port"""
    if isinstance(port, numbers.Number):
        if isinstance(gain, tuple):
            raise ValueError(
                'gain must be a single number unless multiple ports are specified'
            )
    else:
        if isinstance(gain, tuple) and len(gain) != len(port):
            raise ValueError('gain, when specified as a tuple, must match port count')


@functools.lru_cache(4)
def _expand_loops(
    explicit: tuple[_TC, ...], loops: tuple[LoopSpec, ...]
) -> tuple[_TC, ...]:
    """evaluate the loop specification, and flatten into one list of loops"""
    fields = tuple([loop.field for loop in loops])

    for f in fields:
        if f is None:
            continue
        if f not in explicit[0].__struct_fields__:
            raise TypeError(f'loop specifies capture field {f!r} that is not defined')

    loop_points = [loop.get_points() for loop in loops]
    combinations = itertools.product(*loop_points)

    result = []
    for values in combinations:
        updates = dict(zip(fields, values))

        extras = [c.replace(**updates) for c in explicit]

        result += extras

    if len(result) == 0:
        return explicit
    else:
        return tuple(result)


AnalysisBandwidthType = Annotated[
    float, meta('Bandwidth of the analysis filter (or inf to disable)', 'Hz', gt=0)
]
BackendSampleRateType = Annotated[float, meta('Source sample rate', 'Hz', gt=0)]
BaseClockRateType = Annotated[
    float, meta('Base sample rate used inside the source', 'Hz', gt=0)
]
CenterFrequencyType = Annotated[float, meta('RF center frequency', 'Hz', gt=0)]
DelayType = Annotated[float, meta('Delay in acquisition start time', 's', gt=0)]
GainScalarType = Annotated[float, meta('Gain setting', 'dB')]
GainType = Annotated[
    Union[GainScalarType, tuple[GainScalarType, ...]],
    meta('Gain setting for each channel', 'dB'),
]
LOShiftType = Annotated[Literal['left', 'right', 'none'], meta('LO shift direction')]
PortScalarType = Annotated[int, meta('Input port index', ge=0)]
PortType = Annotated[
    Union[PortScalarType, tuple[PortScalarType, ...]],
    meta('Input port indices'),
]


class WaveformCapture(analysis.CaptureBase, frozen=True, kw_only=True, **kws):
    """Capture specification structure for a generic waveform.

    This subset of RadioCapture is broken out here to simplify the evaluation of
    sampling parameters that are independent from other radio parameters.
    """

    # acquisition
    port: PortType
    duration: Annotated[float, meta('Acquisition duration', 's', gt=0)] = 0.1
    sample_rate: Annotated[float, meta('Sample rate', 'S/s', gt=0)] = 15.36e6

    # filtering and resampling
    analysis_bandwidth: AnalysisBandwidthType = float('inf')

    lo_shift: LOShiftType = 'none'
    host_resample: bool = True
    backend_sample_rate: Optional[BackendSampleRateType] = None


class _WaveformCaptureKeywords(TypedDict, total=False):
    # this needs to be kept in sync with WaveformCapture in order to
    # properly provide type hints for IDEs in the arm and acquire
    # call signatures of source.Base objects
    port: PortType
    duration: Annotated[float, meta('Acquisition duration', 's', gt=0)]
    sample_rate: Annotated[float, meta('Sample rate', 'S/s', gt=0)]

    # filtering and resampling
    analysis_bandwidth: AnalysisBandwidthType
    lo_shift: LOShiftType
    host_resample: bool
    backend_sample_rate: Optional[BackendSampleRateType]


class CaptureSpec(WaveformCapture, frozen=True, kw_only=True, **kws):
    """Capture specification for a single radio waveform"""

    delay: Optional[DelayType] = None

    # a counter used to reset the sweep timestamp on Repeat(None)
    sweep_index: typing.ClassVar[int] = 0


class _CaptureSpecKeywords(_WaveformCaptureKeywords, total=False):
    # this needs to be kept in sync with WaveformCapture in order to
    # properly provide type hints for IDEs in the arm and acquire
    # call signatures of source.Base objects
    delay: DelayType


SourceIDType = Annotated[str, meta('Source UUID string')]
StartTimeType = Annotated[
    'pd.Timestamp', meta('Acquisition start time of the first capture')
]
SweepStartTimeType = Annotated['pd.Timestamp', meta('Capture acquisition start time')]


class AcquisitionInfo(analysis.specs.AcquisitionInfo, frozen=True, kw_only=True, **kws):
    """extra coordinate information returned from an acquisition"""

    sweep_time: SweepStartTimeType | None
    start_time: StartTimeType | None
    backend_sample_rate: BackendSampleRateType
    source_id: SourceIDType


class SoapyCaptureSpec(CaptureSpec, frozen=True, kw_only=True, **kws):
    center_frequency: CenterFrequencyType
    gain: GainType

    def __post_init__(self):
        super().__post_init__()
        _validate_multichannel(self.port, self.gain)


class _SoapyCaptureSpecKeywords(_CaptureSpecKeywords, total=False):
    # this needs to be kept in sync with WaveformCapture in order to
    # properly provide type hints for IDEs in the arm and acquire
    # call signatures of source.Base objects
    center_frequency: CenterFrequencyType
    gain: GainType


class FileCaptureSpec(CaptureSpec, frozen=True, kw_only=True, **kws):
    """Capture specification read from a file, with support for None sentinels"""

    # RF and leveling
    backend_sample_rate: BackendSampleRateType | None = None

    def __post_init__(self):
        if self.backend_sample_rate is not None:
            raise TypeError('backend_sample_rate is fixed by the file source')


ClockSourceType = Annotated[
    Literal['internal', 'external', 'gps'],
    meta('Hardware source for the frequency reference'),
]
ContinuousTriggerType = Annotated[
    bool,
    meta('Whether to trigger immediately after each call to acquire() when armed'),
]
ReceiveRetriesType = Annotated[
    int,
    meta(
        'number of attempts to retry acquisition on a stream error',
        ge=0,
    ),
]
TimeSourceType = Annotated[
    Literal['host', 'internal', 'external', 'gps'],
    meta('Hardware source for timestamps'),
]

TimeSyncEveryCaptureType = Annotated[
    bool, meta('whether to sync to PPS before each capture in a sweep')
]

GaplessRepeatType = Annotated[
    bool,
    meta('whether to raise an exception on overflows between identical captures'),
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


class _SourceSpecKeywords(TypedDict, total=False):
    # this needs to be kept in sync with WaveformCapture in order to
    # properly provide type hints for IDEs in the setup
    # call signature of source.Base objects

    driver: str
    base_clock_rate: BaseClockRateType

    resource: dict
    calibration: str

    gapless_repeats: GaplessRepeatType
    warmup_sweep: WarmupSweepType

    periodic_trigger: float
    channel_sync_source: str

    array_backend: ArrayBackendType
    cupy_max_fft_chunk_size: int
    uncalibrated_peak_detect: Union[bool, Literal['auto']]


class SourceSpec(SpecBase, frozen=True, kw_only=True, **kws):
    """run-time characteristics of the radio that are left invariant during a sweep"""

    # driver: Optional[str]

    # acquisition
    base_clock_rate: BaseClockRateType
    calibration: Optional[str] = None

    # sequencing
    warmup_sweep: WarmupSweepType = True
    gapless_retrigger: GaplessRepeatType = False

    # synchronization and triggering
    periodic_trigger: Optional[float] = None
    channel_sync_source: Optional[str] = None

    # in the future, these should probably move to an analysis config
    array_backend: ArrayBackendType = 'cupy'
    cupy_max_fft_chunk_size: Optional[int] = None

    # validation data
    uncalibrated_peak_detect: Union[bool, Literal['auto']] = 'auto'

    # calibration subclasses set this True to skip unecessary
    # re-acquisitions
    reuse_iq: ClassVar[bool] = False

    transient_holdoff_time = None
    stream_all_rx_ports = False
    transport_dtype: Literal['int16', 'complex64'] = 'complex64'


class SoapySourceSpec(SourceSpec, frozen=True, kw_only=True, **kws):
    device_kwargs: ClassVar[dict[str, Any]] = {}

    time_source: TimeSourceType = 'host'
    time_sync_every_capture: TimeSyncEveryCaptureType = False
    clock_source: ClockSourceType = 'internal'
    receive_retries: ReceiveRetriesType = 0

    # True if the same clock drives acquisition on all RX ports
    shared_rx_sample_clock = True
    rx_enable_delay = 0.0

    def __post_init__(self):
        from striqt.analysis import registry

        if not self.gapless_retrigger:
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
        elif self.channel_sync_source not in registry.channel_sync_source:
            registered = set(registry.channel_sync_source)
            raise ValueError(
                f'channel_sync_source "{self.channel_sync_source!r}" is not one of the registered functions {registered!r}'
            )


class _SoapySourceSpecKeywords(_SourceSpecKeywords, total=False):
    receive_retries: ReceiveRetriesType
    time_source: TimeSourceType
    clock_source: ClockSourceType
    time_sync_every_capture: TimeSyncEveryCaptureType


class NullSourceSpec(SourceSpec, frozen=True, kw_only=True, **kws):
    # make these configurable, to support matching hardware for warmup sweeps
    num_rx_ports: int
    stream_all_rx_ports: bool = False


class Description(SpecBase, frozen=True, kw_only=True, **kws):
    summary: Optional[str] = None
    location: Optional[tuple[float, float, float]] = None
    signal_chain: Optional[tuple[str, ...]] = tuple()
    version: str = 'unversioned'


class LoopBase(
    SpecBase, tag=str.lower, tag_field='kind', frozen=True, kw_only=True, **kws
):
    field: str

    def get_points(self):
        raise NotImplementedError


class Range(LoopBase, frozen=True, kw_only=True, **kws):
    start: float
    stop: float
    step: float

    def get_points(self):
        import numpy as np

        if self.start == self.stop:
            return np.array([self.start])

        a = np.arange(self.start, self.stop + self.step / 2, self.step)
        return list(a)


class Repeat(LoopBase, frozen=True, kw_only=True, **kws):
    field: str = '_sweep_index'
    count: int = 1

    def get_points(self):
        return list(range(self.count))


class List(LoopBase, frozen=True, kw_only=True, **kws):
    values: tuple[Any, ...]

    def get_points(self):
        return self.values


class FrequencyBinRange(LoopBase, frozen=True, kw_only=True, **kws):
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


LoopSpec = Union[Repeat, List, Range, FrequencyBinRange]


AliasMatchType = Annotated[
    Union[dict[str, dict[str, Any]], tuple[dict[str, dict[str, Any]], ...]],
    meta('one or more dictionaries of valid match sets to "or"'),
]


class SinkSpec(analysis.specs._SlowHashSpecBase, frozen=True, kw_only=True, **kws):
    path: str = '{yaml_name}-{start_time}'
    log_path: Optional[str] = None
    log_level: str = 'info'
    store: Literal['zip', 'directory'] = 'directory'
    coord_aliases: dict[str, dict[str, AliasMatchType]] = {}
    max_threads: Optional[int] = None


SinkClassType = Annotated[
    Union[str, Literal['striqt.sensor.writers.CaptureAppender']],
    meta('data sink class to import and use'),
]
ModuleNameType = Annotated[
    Union[str, None],
    meta('name of the extension module that calls bind_sensor'),
]
ExtensionPathType = Annotated[
    str,
    meta('path to append to sys.path before extension imports'),
]


class ExtensionSpec(SpecBase, frozen=True, kw_only=True, **kws):
    sink: SinkClassType = 'striqt.sensor.sinks.CaptureAppender'
    import_path: typing.Optional[ExtensionPathType] = None
    import_name: ModuleNameType = None


# registered striqt.analysis.measurements -> Analysis spec
BundledAnalysis = analysis.registry.tospec()
BundledAlignmentAnalysis = analysis.registry.channel_sync_source.to_spec()


class SweepSpec(SpecBase, Generic[_TS, _TC], frozen=True, kw_only=True, **kws):
    source: _TS
    captures: tuple[_TC, ...] = tuple()
    loops: tuple[LoopSpec, ...] = ()

    analysis: BundledAnalysis = BundledAnalysis()  # type: ignore
    description: Union[Description, str] = ''
    extensions: ExtensionSpec = ExtensionSpec()
    sink: SinkSpec = SinkSpec()

    @property
    def looped_captures(self) -> tuple[_TC, ...]:
        """subclasses may use this to autogenerate capture sequences"""
        return _expand_loops(self.captures, self.loops)

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
    def _from_registry(
        cls: type[SweepSpec], registry: analysis.MeasurementRegistry
    ) -> type[SweepSpec]:
        bases = typing.get_type_hints(cls, include_extras=True)

        AnalysisCls = registry.tospec(bases[cls.analysis.__name__])

        fields = ((cls.analysis.__name__, AnalysisCls, AnalysisCls()),)

        subcls = msgspec.defstruct(
            cls.__name__,
            fields,
            bases=(cls,),
            frozen=True,
            forbid_unknown_fields=True,
            cache_hash=True,
        )

        return typing.cast(type[SweepSpec], subcls)
