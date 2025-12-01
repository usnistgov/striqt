"""schema for the specification of calibration and sweeps"""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path
import typing
from typing import Annotated, Any, Generic, Literal, Optional, Union

import msgspec

from striqt import analysis
from striqt.analysis.lib.specs import SpecBase, meta

from . import util

if typing.TYPE_CHECKING:
    _T = typing.TypeVar('_T')
    import pandas as pd
    from typing_extensions import Self


def _dict_hash(d):
    key_hash = frozenset(d.keys())
    value_hash = tuple(
        [_dict_hash(v) if isinstance(v, dict) else v for v in d.values()]
    )
    return hash(key_hash) ^ hash(value_hash)


@util.lru_cache()
def _validate_multichannel(port, gain):
    """guarantee that self.gain is a number, or matches the length of self.port"""
    if not isinstance(port, tuple):
        if isinstance(gain, tuple):
            raise ValueError(
                'gain must be a single number unless multiple ports are specified'
            )
    else:
        if isinstance(gain, tuple) and len(gain) != len(port):
            raise ValueError('gain, when specified as a tuple, must match port count')


@util.lru_cache()
def _check_fields(cls: type[SpecBase], names: tuple[str, ...], new_instance=False):
    fields = msgspec.structs.fields(cls)
    available = set(names)

    if new_instance:
        required = {f.name for f in fields if f.required}
        missing = required - available
        if len(missing) > 0:
            raise TypeError(f'missing required loop fields {missing!r}')

    extra = available - {f.name for f in fields}
    if len(extra) > 0:
        raise TypeError(f'invalid capture fields {extra!r} specified in loops')


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


class ResampledCapture(analysis.Capture, frozen=True, kw_only=True):
    """Capture specification for a generic waveform with resampling support"""

    # acquisition
    port: PortType
    lo_shift: LOShiftType = 'none'
    host_resample: bool = True
    backend_sample_rate: Optional[BackendSampleRateType] = None

    # a counter used to track the loop index for Repeat(None)
    _sweep_index: int = 0


class _ResampledCaptureKeywords(analysis.specs._CaptureKeywords, total=False):
    # this needs to be kept in sync with CaptureSpec in order to
    # properly provide type hints for IDEs for .replace()-ish methods
    port: PortType
    lo_shift: LOShiftType
    host_resample: bool
    backend_sample_rate: Optional[BackendSampleRateType]


class SoapyCapture(ResampledCapture, frozen=True, kw_only=True):
    delay: Optional[DelayType] = None
    center_frequency: CenterFrequencyType
    gain: GainType

    def __post_init__(self):
        super().__post_init__()
        _validate_multichannel(self.port, self.gain)


class _SoapyCaptureKeywords(_ResampledCaptureKeywords):
    # this needs to be kept in sync with WaveformCapture in order to
    # properly provide type hints for IDEs in the arm and acquire
    # call signatures of source.Base objects
    delay: DelayType
    center_frequency: CenterFrequencyType
    gain: GainType


FrequencyOffsetType = Annotated[float, meta('Baseband frequency offset', 'Hz')]
SNRType = Annotated[float, meta('SNR with added noise ', 'dB')]
PSDType = Annotated[float, meta('noise total channel power', 'mW/Hz', ge=0)]
PowerType = Annotated[float, meta('peak power level', 'dB', gt=0)]
TimeType = Annotated[float, meta('start time offset', 's')]
PeriodType = Annotated[float, meta('waveform period', 's', ge=0)]


class SingleToneCaptureSpec(ResampledCapture, frozen=True, kw_only=True):
    frequency_offset: FrequencyOffsetType = 0
    snr: typing.Optional[SNRType] = None


class DiracDeltaCaptureSpec(ResampledCapture, frozen=True, kw_only=True):
    time: TimeType = 0
    power: PowerType = 0


class SawtoothCaptureSpec(ResampledCapture, kw_only=True, frozen=True, dict=True):
    period: PeriodType = 0.01
    power: PowerType = 1


class NoiseCaptureSpec(ResampledCapture, kw_only=True, frozen=True, dict=True):
    power_spectral_density: PSDType = 1e-17


NoiseDiodeEnabledType = Annotated[bool, meta(standard_name='Noise diode enabled')]


class FileCapture(ResampledCapture, frozen=True, kw_only=True):
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
    meta('number of acquisition retry attempts on stream error', ge=0),
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
    meta('if True, run empty buffers through the GPU before sweeping'),
]
ArrayBackendType = Annotated[
    Literal['numpy', 'cupy'],
    meta('array module to use to set compute device: numpy = cpu, cupy = gpu'),
]
SyncSourceType = Annotated[
    str,
    meta('name of a registered waveform alignment function'),
]


class Source(SpecBase, frozen=True, kw_only=True):
    """run-time characteristics of the radio that are left invariant during a sweep"""

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
    uncalibrated_peak_detect: bool | Literal['auto'] = False

    transient_holdoff_time: typing.ClassVar[float] = 0
    stream_all_rx_ports: typing.ClassVar[bool | None] = False
    transport_dtype: typing.ClassVar[Literal['int16', 'complex64']] = 'complex64'


class _SourceKeywords(typing.TypedDict, total=False):
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
    uncalibrated_peak_detect: bool | Literal['auto']


class SoapySource(Source, frozen=True, kw_only=True):
    time_source: TimeSourceType = 'host'
    time_sync_every_capture: TimeSyncEveryCaptureType = False
    clock_source: ClockSourceType = 'internal'
    receive_retries: ReceiveRetriesType = 0

    uncalibrated_peak_detect: bool | Literal['auto'] = True

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


class _SoapySourceKeywords(_SourceKeywords, total=False):
    receive_retries: ReceiveRetriesType
    time_source: TimeSourceType
    clock_source: ClockSourceType
    time_sync_every_capture: TimeSyncEveryCaptureType


class FunctionSourceSpec(Source, kw_only=True, frozen=True):
    # make these configurable, to support matching hardware for warmup sweeps
    num_rx_ports: int
    stream_all_rx_ports: typing.ClassVar[bool] = False


class NoSource(Source, frozen=True, kw_only=True):
    # make these configurable, to support matching hardware for warmup sweeps
    num_rx_ports: int
    stream_all_rx_ports: typing.ClassVar[bool] = False


FormatType = Annotated[
    typing.Literal['auto', 'mat', 'tdms'],
    meta('data format or auto to guess by extension'),
]
WaveformInputPath = Annotated[Path, meta('path to the waveform data file')]
FileMetadataType = Annotated[dict, meta('any capture fields not included in the file')]
FileLoopType = Annotated[
    bool, meta('whether to loop the file to create longer IQ waveforms')
]


class FileSourceSpec(NoSource, kw_only=True, frozen=True, dict=True):
    path: WaveformInputPath
    file_format: FormatType = 'auto'
    file_metadata: FileMetadataType = {}
    loop: FileLoopType = False


class TDMSFileSourceSpec(NoSource, frozen=True, kw_only=True):
    path: WaveformInputPath


class ZarrIQSourceSpec(NoSource, frozen=True, kw_only=True):
    path: WaveformInputPath
    select: Annotated[
        dict, meta('dictionary to select in the data as .sel(**select)')
    ] = {}


class Description(SpecBase, frozen=True, kw_only=True):
    summary: Optional[str] = None
    location: Optional[tuple[float, float, float]] = None
    signal_chain: Optional[tuple[str, ...]] = tuple()
    version: str = 'unversioned'


class LoopBase(SpecBase, tag=str.lower, tag_field='kind', frozen=True, kw_only=True):
    field: str

    def get_points(self):
        raise NotImplementedError


class Range(LoopBase, frozen=True, kw_only=True):
    start: float
    stop: float
    step: float

    def get_points(self):
        import numpy as np

        if self.start == self.stop:
            return np.array([self.start])

        a = np.arange(self.start, self.stop + self.step / 2, self.step)
        return list(a)


class Repeat(LoopBase, frozen=True, kw_only=True):
    field: str = '_sweep_index'
    count: int = 1

    def get_points(self):
        return list(range(self.count))


class List(LoopBase, frozen=True, kw_only=True):
    values: tuple[Any, ...]

    def get_points(self):
        return self.values


class FrequencyBinRange(LoopBase, frozen=True, kw_only=True):
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


class Sink(analysis.specs._SlowHashSpecBase, frozen=True, kw_only=True):
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


class Extension(SpecBase, frozen=True, kw_only=True):
    sink: SinkClassType | None = None
    import_path: typing.Optional[ExtensionPathType] = None
    import_name: ModuleNameType = None


class Peripherals(SpecBase, frozen=True, kw_only=True):
    pass


class NoPeripherals(Peripherals, frozen=True, kw_only=True):
    pass


ENRType = Annotated[float, meta(standard_name='Excess noise ratio', units='dB')]
AmbientTemperatureType = Annotated[
    float, meta(standard_name='Ambient temperature', units='K')
]


class ManualYFactorPeripheral(Peripherals, frozen=True, kw_only=True):
    enr: ENRType
    ambient_temperature: AmbientTemperatureType


# registered striqt.analysis.measurements -> Analysis spec
BundledAnalysis = analysis.registry.tospec()
BundledAlignmentAnalysis = analysis.registry.channel_sync_source.to_spec()


class SweepInfo(typing.NamedTuple):
    reuse_iq: bool


# forward references break msgspec when used with bindings, so this
# needs to be here after the bound classes have been defined
_TC = typing.TypeVar('_TC', bound=ResampledCapture)
_TP = typing.TypeVar('_TP', bound=Peripherals)
_TPC = typing.TypeVar('_TPC', bound=Peripherals)
_TS = typing.TypeVar('_TS', bound=Source)


@util.lru_cache(4)
def _expand_loops(sweep: Sweep[_TS, _TP, _TC], nyquist_only=False) -> tuple[_TC, ...]:
    """evaluate the loop specification, and flatten into one list of loops"""

    loop_fields = tuple([loop.field for loop in sweep.loops])

    if len(sweep.captures) > 0:
        cls = type(sweep.captures[0])
        _check_fields(cls, loop_fields, False)
    elif sweep.__bindings__ is None:
        raise TypeError(
            'loops may apply only to explicit capture lists unless the sweep '
            'is bound to a sensor with striqt.sensor.bind_sensor'
        )
    else:
        from . import bindings

        assert isinstance(sweep.__bindings__, bindings.SensorBinding)
        cls = sweep.__bindings__.schema.capture
        _check_fields(cls, loop_fields, True)

    assert issubclass(cls, analysis.Capture)

    loop_points = [loop.get_points() for loop in sweep.loops]
    combinations = itertools.product(*loop_points)

    result = []
    for values in combinations:
        updates = dict(zip(loop_fields, values))
        if len(sweep.captures) > 0:
            # iterate specified captures if avialable
            new = (c.replace(**updates) for c in sweep.captures)
        else:
            # otherwise, instances are new captures
            new = (cls.fromdict(updates) for _ in range(1))

        if nyquist_only:
            new = (c for c in new if c.sample_rate >= c.analysis_bandwidth)

        result += list(new)

    if len(result) == 0:
        return sweep.captures
    else:
        return tuple(result)


class Sweep(SpecBase, Generic[_TS, _TP, _TC], frozen=True, kw_only=True):
    source: _TS
    captures: tuple[_TC, ...] = tuple()
    loops: tuple[LoopSpec, ...] = ()
    analysis: BundledAnalysis = BundledAnalysis()  # type: ignore
    description: Description | str = ''
    extensions: Extension = Extension()
    sink: Sink = Sink()
    peripherals: _TP = typing.cast(_TP, Peripherals())

    info: typing.ClassVar[SweepInfo] = SweepInfo(reuse_iq=False)
    __bindings__: typing.ClassVar[typing.Any] = None

    def loop_captures(self) -> tuple[_TC, ...]:
        """apply loops to self.captures"""
        return _expand_loops(self)

    def __post_init__(self):
        if len(self.loops) == 0:
            return

        from collections import Counter

        (which, howmany), *_ = Counter(l.field for l in self.loops).most_common(1)
        if howmany > 1:
            raise TypeError(f'more than one loop of capture field {which!r}')

    # @classmethod
    # def _from_registry(
    #     cls: type[Sweep], registry: analysis.MeasurementRegistry
    # ) -> type[Sweep]:
    #     bases = typing.get_type_hints(cls, include_extras=True)

    #     AnalysisCls = registry.tospec(bases[cls.analysis.__name__])

    #     fields = ((cls.analysis.__name__, AnalysisCls, AnalysisCls()),)

    #     subcls = msgspec.defstruct(
    #         cls.__name__,
    #         fields,
    #         bases=(cls,),
    #         frozen=True,
    #         forbid_unknown_fields=True,
    #         cache_hash=True,
    #     )

    #     return typing.cast(type[Sweep], subcls)


class CalibrationSweep(
    Sweep[_TS, _TP, _TC],
    typing.Generic[_TS, _TP, _TC, _TPC],
    frozen=True,
    kw_only=True,
):
    """This specialized sweep is fed to the YAML file loader
    to specify the change in expected capture structure."""

    info: typing.ClassVar[SweepInfo] = SweepInfo(reuse_iq=True)
    calibration: _TPC | None = None

    def __post_init__(self):
        if len(self.captures) > 0:
            raise TypeError(
                'calibration sweeps may specify loops but not captures, only loops'
            )
        if self.source.calibration is not None:
            raise ValueError('source.calibration must be None for a calibration sweep')

        super().__post_init__()


SourceIDType = Annotated[str, meta('Source UUID string')]

StartTimeType = Annotated[
    'pd.Timestamp', meta('Acquisition start time of the first capture')
]
SweepStartTimeType = Annotated['pd.Timestamp', meta('Capture acquisition start time')]


# we really only need a dataclass for internal message-passing,
# but using msgspec.Struct here to support kw_only=True for python < 3.10
class AcquisitionInfo(msgspec.Struct, kw_only=True, frozen=True):
    """information about an acquired acquisition"""

    # duck-type methods and structure of SpecBase

    source_id: SourceIDType = ''

    def replace(self, **attrs) -> 'Self':
        """returns a copy of self with changed attributes.

        See also:
            Python standard library `copy.replace`
        """
        return dataclasses.replace(self, **attrs)

    def todict(self) -> dict:
        """return a dictinary representation of `self`"""
        return dataclasses.asdict(self)

    @classmethod
    def fromdict(cls: type[_T], d: dict) -> _T:
        return cls(**d)


class SoapyAcquisitionInfo(AcquisitionInfo, kw_only=True, frozen=True):
    """extra coordinate information returned from an acquisition"""

    delay: typing.Optional[DelayType] = None
    sweep_time: SweepStartTimeType | None
    start_time: StartTimeType | None
    backend_sample_rate: BackendSampleRateType
    source_id: SourceIDType = ''


class FileAcquisitionInfo(AcquisitionInfo, kw_only=True, frozen=True):
    center_frequency: CenterFrequencyType = float('nan')
    backend_sample_rate: BackendSampleRateType
    port: PortType = 0
    gain: GainType = float('nan')
    source_id: SourceIDType = ''


@util.lru_cache()
def dataclass_fields(
    cls: type[AcquisitionInfo],
) -> tuple[msgspec.structs.FieldInfo, ...]:
    import msgspec

    hints = typing.get_type_hints(cls)
    return tuple(
        [
            msgspec.structs.FieldInfo(
                name=f.name,
                encode_name=f.name,
                type=hints[f.name],
            )
            for f in dataclasses.fields(cls)
        ]
    )
