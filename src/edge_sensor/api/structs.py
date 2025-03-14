"""data structures for configuration of radio hardware, captures, and sweeps"""

from __future__ import annotations
import functools
import msgspec
import numbers
import typing
from typing import Annotated, Optional, Literal, Any, Union

from . import util

import channel_analysis
import channel_analysis.api.filters
from channel_analysis.api.structs import (
    meta,
    ChannelAnalysis,  # noqa: F401
    struct_to_builtins,  # noqa: F401
    builtins_to_struct,  # noqa: F401
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
LOShiftType = Literal['left', 'right', 'none']
DelayType = Annotated[float, meta('Delay in acquisition start time', 's', gt=0)]
StartTimeType = Annotated['pd.Timestamp', meta('Acquisition start time')]


def _dict_hash(d):
    key_hash = frozenset(d.keys())
    value_hash = tuple(
        [_dict_hash(v) if isinstance(v, dict) else v for v in d.values()]
    )
    return hash(key_hash) ^ hash(value_hash)


def _make_default_analysis():
    return channel_analysis.as_registered_channel_analysis.spec_type()()


class WaveformCapture(channel_analysis.Capture, forbid_unknown_fields=True):
    """Capture specification structure for a generic waveform.

    This subset of RadioCapture is broken out here to simplify the evaluation of
    sampling parameters that are independent from other radio parameters.
    """

    # acquisition
    duration: Annotated[float, meta('Acquisition duration', 's', gt=0)] = 0.1
    sample_rate: Annotated[float, meta('Sample rate', 'S/s', gt=0)] = 15.36e6

    # filtering and resampling
    analysis_bandwidth: Annotated[
        float, meta('Bandwidth of the analysis filter (or inf to disable)', 'Hz', gt=0)
    ] = float('inf')
    lo_shift: Annotated[LOShiftType, meta('LO shift direction')] = 'none'
    host_resample: bool = True


@functools.lru_cache
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


class RadioCapture(WaveformCapture, forbid_unknown_fields=True):
    """Capture specification for a single radio waveform"""

    # RF and leveling
    center_frequency: CenterFrequencyType = 3710e6
    channel: ChannelType = 0
    gain: GainType = -10

    delay: Optional[DelayType] = None
    start_time: Optional[StartTimeType] = None

    def __post_init__(self):
        _validate_multichannel(self.channel, self.gain)


class FileSourceCapture(RadioCapture, forbid_unknown_fields=True):
    """Capture specification read from a file, with support for None sentinels"""

    # RF and leveling
    center_frequency: Optional[CenterFrequencyType] = float('nan')
    channel: Optional[ChannelType] = 0
    gain: Optional[GainType] = float('nan')
    backend_sample_rate: Optional[float] = float('nan')


TimeSourceType = Literal['host', 'internal', 'external', 'gps']
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


class RadioSetup(msgspec.Struct, forbid_unknown_fields=True):
    """run-time characteristics of the radio that are left invariant during a sweep"""

    driver: Optional[str] = None
    device_args: dict = {}
    resource: Any = None
    time_source: TimeSourceType = 'host'
    continuous_trigger: ContinuousTriggerType = True
    periodic_trigger: Optional[float] = None
    calibration: Optional[str] = None
    gapless_repeats: GaplessRepeatType = False
    time_sync_every_capture: TimeSyncEveryCaptureType = False
    warmup_sweep: WarmupSweepType = True
    array_backend: ArrayBackendType = 'cupy'

    _transient_holdoff_time: Optional[float] = None
    _rx_channel_count: Optional[int] = None

    def __post_init__(self):
        if self.gapless_repeats and self.time_sync_every_capture:
            raise ValueError(
                'time_sync_every_capture and gapless_repeats are mutually exclusive'
            )


class Description(msgspec.Struct, forbid_unknown_fields=True):
    summary: Optional[str] = None
    location: Optional[tuple[float, float, float]] = None
    signal_chain: Optional[tuple[str, ...]] = tuple()
    version: str = 'unversioned'


class Output(msgspec.Struct, forbid_unknown_fields=True, frozen=True, cache_hash=True):
    path: Optional[str] = '{yaml_name}-{start_time}'
    store: typing.Union[Literal['zip'], Literal['directory']] = 'zip'
    coord_aliases: dict[str, dict[str, dict[str, Any]]] = {}

    def __hash__(self):
        # hashing coordinate aliases greatly speeds up xarray coordinate generation
        return hash(self.path) ^ hash(self.store) ^ _dict_hash(self.coord_aliases)


class Sweep(msgspec.Struct, forbid_unknown_fields=True):
    captures: tuple[RadioCapture, ...]
    radio_setup: RadioSetup = msgspec.field(default_factory=RadioSetup)
    defaults: RadioCapture = msgspec.field(default_factory=RadioCapture)
    channel_analysis: dict = msgspec.field(default_factory=_make_default_analysis)
    description: Description = msgspec.field(default_factory=Description)
    output: Output = msgspec.field(default_factory=Output)
