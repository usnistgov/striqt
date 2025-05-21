"""data structures that specify operation of radio hardware, captures, and sweeps"""

from __future__ import annotations
import msgspec
import numbers
import typing
from typing import Annotated, Optional, Literal, Any, Union

from . import util

from striqt import analysis
from striqt.analysis.lib.specs import (
    meta,
    Measurement,  # noqa: F401
    StructBase,
)

if typing.TYPE_CHECKING:
    import pandas as pd
else:
    # this is needed to resolve the 'pd.Timestamp' stub at runtime
    pd = util.lazy_import('pandas')

SingleChannelType = Annotated[int, meta('Input port index', ge=0)]
SingleGainType = Annotated[float, meta('Gain setting', 'dB')]
CenterFrequencyType = Annotated[float, meta('RF center frequency', 'Hz', gt=0)]
ChannelType = Annotated[
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


def _make_default_analysis():
    return analysis.lib.registry.measurement.container_spec()()


AnalysisBandwidthType = Annotated[
    float, meta('Bandwidth of the analysis filter (or inf to disable)', 'Hz', gt=0)
]


@util.lru_cache()
def _validate_multichannel(channel, gain):
    """guarantee that self.gain is a number or matches the length of self.channel"""
    if isinstance(channel, numbers.Number):
        if isinstance(gain, tuple):
            raise ValueError(
                'gain must be a single number unless multiple channels are specified'
            )
    else:
        if isinstance(gain, tuple) and len(gain) != len(channel):
            raise ValueError(
                'gain, when specified as a tuple, must match channel count'
            )


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
    channel: ChannelType = 0
    gain: GainType = -10

    delay: Optional[DelayType] = None
    start_time: Optional[StartTimeType] = None

    def __post_init__(self):
        _validate_multichannel(self.channel, self.gain)


class _RadioCaptureKeywords(_WaveformCaptureKeywords, total=False):
    # this needs to be kept in sync with WaveformCapture in order to
    # properly provide type hints for IDEs in the arm and acquire
    # call signatures of source.Base objects
    center_frequency: CenterFrequencyType
    channel: ChannelType
    gain: GainType
    delay: Optional[DelayType]
    start_time: Optional[StartTimeType]


class FileSourceCapture(RadioCapture, forbid_unknown_fields=True, cache_hash=True):
    """Capture specification read from a file, with support for None sentinels"""

    # RF and leveling
    center_frequency: Optional[CenterFrequencyType] = float('nan')
    channel: Optional[ChannelType] = 0
    gain: Optional[GainType] = float('nan')
    backend_sample_rate: Optional[float] = float('nan')


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


class RadioSetup(StructBase, forbid_unknown_fields=True, frozen=True, cache_hash=True):
    """run-time characteristics of the radio that are left invariant during a sweep"""

    driver: Optional[str] = None
    resource: dict = {}
    time_source: TimeSourceType = 'host'
    clock_source: ClockSourceType = 'internal'
    continuous_trigger: ContinuousTriggerType = True
    periodic_trigger: Optional[float] = None
    calibration: Optional[str] = None
    gapless_repeats: GaplessRepeatType = False
    time_sync_every_capture: TimeSyncEveryCaptureType = False
    warmup_sweep: WarmupSweepType = True
    array_backend: ArrayBackendType = 'cupy'
    fast_lo: FastLOType = True

    # this is enabled by a calibration subclass to skip unecessary
    # re-acquisitions
    reuse_iq = False

    _transient_holdoff_time: Optional[float] = None
    _rx_channel_count: Optional[int] = None

    def __post_init__(self):
        if self.gapless_repeats and self.time_sync_every_capture:
            raise ValueError(
                'time_sync_every_capture and gapless_repeats are mutually exclusive'
            )


class _RadioSetupKeywords(typing.TypedDict, total=False):
    # this needs to be kept in sync with WaveformCapture in order to
    # properly provide type hints for IDEs in the setup
    # call signature of source.Base objects

    driver: Optional[str]
    resource: dict = {}
    time_source: TimeSourceType
    clock_source: ClockSourceType
    continuous_trigger: ContinuousTriggerType
    periodic_trigger: Optional[float]
    calibration: Optional[str]
    gapless_repeats: GaplessRepeatType
    time_sync_every_capture: TimeSyncEveryCaptureType
    warmup_sweep: WarmupSweepType
    array_backend: ArrayBackendType
    fast_lo: FastLOType


class Description(StructBase, forbid_unknown_fields=True, frozen=True, cache_hash=True):
    summary: Optional[str] = None
    location: Optional[tuple[float, float, float]] = None
    signal_chain: Optional[tuple[str, ...]] = tuple()
    version: str = 'unversioned'


class Output(StructBase, forbid_unknown_fields=True, frozen=True, cache_hash=True):
    path: Optional[str] = '{yaml_name}-{start_time}'
    store: typing.Union[Literal['zip'], Literal['directory']] = 'directory'
    coord_aliases: dict[str, dict[str, dict[str, Any]]] = {}

    def __hash__(self):
        # hashing coordinate aliases greatly speeds up xarray coordinate generation
        return hash(self.path) ^ hash(self.store) ^ _dict_hash(self.coord_aliases)


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


class Extensions(StructBase, forbid_unknown_fields=True, frozen=True, cache_hash=True):
    peripherals: PeripheralClassType = 'striqt.sensor.peripherals.NoPeripherals'
    sink: SinkClassType = 'striqt.sensor.sinks.CaptureAppender'
    sweep_struct: SweepStructType = 'striqt.sensor.Sweep'
    import_path: ExtensionPathType = None


class Sweep(StructBase, forbid_unknown_fields=True, frozen=True, cache_hash=True):
    captures: tuple[RadioCapture, ...] = tuple()
    radio_setup: RadioSetup = RadioSetup()
    defaults: RadioCapture = RadioCapture()
    analysis: dict = msgspec.field(default_factory=_make_default_analysis)
    description: Description = Description()
    extensions: Extensions = Extensions()
    output: Output = Output()

    def get_captures(self):
        """allow autogeneration of capture sequences"""
        return object.__getattribute__(self, 'captures')

    def __getattribute__(self, name):
        if name == 'captures':
            return self.get_captures()
        else:
            return super().__getattribute__(name)
