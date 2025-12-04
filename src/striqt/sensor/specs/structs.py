"""schema for the specification of calibration and sweeps"""

from __future__ import annotations as __

import typing

import msgspec

from striqt import analysis as _analysis
from striqt.analysis.lib.specs import SpecBase, Capture, _CaptureKeywords, _SlowHashSpecBase

from ..lib import util
from . import types


if typing.TYPE_CHECKING:
    _T = typing.TypeVar('_T')
    from typing_extensions import Self as _Self

    # pd imports need to be here for msgspec to resolve timestamp types
    import pandas as pd
else:
    pd = util.lazy_import('pandas')


@util.lru_cache()
def _validate_multichannel(port, gain):
    """ensure self.gain is a number or matches len(self.port)"""
    if not isinstance(port, tuple):
        if isinstance(gain, tuple):
            raise ValueError(
                'gain must be a single number unless multiple ports are specified'
            )
    else:
        if isinstance(gain, tuple) and len(gain) != len(port):
            raise ValueError('gain, when specified as a tuple, must match port count')


class ResampledCapture(Capture, frozen=True, kw_only=True):
    """Capture specification for a generic waveform with resampling support"""

    # acquisition
    port: types.Port
    lo_shift: types.LOShift = 'none'
    host_resample: bool = True
    backend_sample_rate: typing.Optional[types.BackendSampleRate] = None

    # a counter used to track the loop index for Repeat(None)
    _sweep_index: int = 0


class _ResampledCaptureKeywords(_CaptureKeywords, total=False):
    # this needs to be kept in sync with CaptureSpec in order to
    # properly provide type hints for IDEs for .replace()-ish methods
    port: types.Port
    lo_shift: types.LOShift
    host_resample: bool
    backend_sample_rate: typing.Optional[types.BackendSampleRate]


class SoapyCapture(ResampledCapture, frozen=True, kw_only=True):
    delay: typing.Optional[types.StartDelay] = None
    center_frequency: types.CenterFrequency
    gain: types.Gain

    def __post_init__(self):
        super().__post_init__()
        _validate_multichannel(self.port, self.gain)


class _SoapyCaptureKeywords(_ResampledCaptureKeywords):
    # this needs to be kept in sync with WaveformCapture in order to
    # properly provide type hints for IDEs in the arm and acquire
    # call signatures of source.Base objects
    delay: typing.Optional[types.StartDelay]
    center_frequency: types.CenterFrequency
    gain: types.Gain


class SingleToneCaptureSpec(ResampledCapture, frozen=True, kw_only=True):
    frequency_offset: types.FrequencyOffset = 0
    snr: typing.Optional[types.SNR] = None


class DiracDeltaCaptureSpec(ResampledCapture, frozen=True, kw_only=True):
    time: types.TimeOffset = 0
    power: types.Power = 0


class SawtoothCaptureSpec(ResampledCapture, kw_only=True, frozen=True, dict=True):
    period: types.Period = 0.01
    power: types.Power = 1


class NoiseCaptureSpec(ResampledCapture, kw_only=True, frozen=True, dict=True):
    power_spectral_density: types.PSD = 1e-17


class FileCapture(ResampledCapture, frozen=True, kw_only=True):
    """Capture specification read from a file, with support for None sentinels"""

    # RF and leveling
    backend_sample_rate: typing.Optional[types.BackendSampleRate] = None

    def __post_init__(self):
        if self.backend_sample_rate is not None:
            raise TypeError('backend_sample_rate is fixed by the file source')


class Source(SpecBase, frozen=True, kw_only=True):
    """run-time characteristics of the radio that are left invariant during a sweep"""

    # acquisition
    base_clock_rate: types.BaseClockRate
    calibration: typing.Optional[str] = None

    # sequencing
    warmup_sweep: types.WarmupSweep = True
    gapless_rearm: types.GaplessRepeat = False

    # synchronization and triggering
    periodic_trigger: typing.Optional[float] = None
    channel_sync_source: typing.Optional[str] = None

    # in the future, these should probably move to an analysis config
    array_backend: types.ArrayBackend = 'cupy'
    cupy_max_fft_chunk_size: typing.Optional[int] = None

    # validation data
    uncalibrated_peak_detect: types.OverloadDetectFlag = False

    transient_holdoff_time: typing.ClassVar[float] = 0
    stream_all_rx_ports: typing.ClassVar[bool | None] = False
    transport_dtype: typing.ClassVar[types.TransportDType] = 'float32'


class _SourceKeywords(typing.TypedDict, total=False):
    # this needs to be kept in sync with WaveformCapture in order to
    # properly provide type hints for IDEs in the setup
    # call signature of source.Base objects

    base_clock_rate: types.BaseClockRate

    resource: dict
    calibration: str

    gapless_rearm: types.GaplessRepeat
    warmup_sweep: types.WarmupSweep

    periodic_trigger: float
    channel_sync_source: str

    array_backend: types.ArrayBackend
    cupy_max_fft_chunk_size: int
    uncalibrated_peak_detect: types.OverloadDetectFlag


class SoapySource(Source, frozen=True, kw_only=True):
    time_source: types.TimeSource = 'host'
    time_sync_every_capture: types.TimeSyncEveryCapture = False
    clock_source: types.ClockSource = 'internal'
    receive_retries: types.ReceiveRetries = 0

    uncalibrated_peak_detect: types.OverloadDetectFlag = True

    # True if the same clock drives acquisition on all RX ports
    shared_rx_sample_clock = True
    rx_enable_delay = 0.0

    def __post_init__(self):
        from striqt.analysis import registry

        if not self.gapless_rearm:
            pass
        elif self.time_sync_every_capture:
            raise ValueError(
                'time_sync_every_capture and gapless_rearm are mutually exclusive'
            )
        elif self.receive_retries > 0:
            raise ValueError('receive_retries must be 0 when gapless_rearm is enabled')
        if self.channel_sync_source is None:
            pass
        elif self.channel_sync_source not in registry.channel_sync_source:
            registered = set(registry.channel_sync_source)
            raise ValueError(
                f'channel_sync_source "{self.channel_sync_source!r}" is not one of the registered functions {registered!r}'
            )


class _SoapySourceKeywords(_SourceKeywords, total=False):
    receive_retries: types.ReceiveRetries
    time_source: types.TimeSource
    clock_source: types.ClockSource
    time_sync_every_capture: types.TimeSyncEveryCapture


class FunctionSourceSpec(Source, kw_only=True, frozen=True):
    # make these configurable, to support matching hardware for warmup sweeps
    num_rx_ports: int
    stream_all_rx_ports: typing.ClassVar[bool] = False


class NoSource(Source, frozen=True, kw_only=True):
    # make these configurable, to support matching hardware for warmup sweeps
    num_rx_ports: int
    stream_all_rx_ports: typing.ClassVar[bool] = False

class FileSourceSpec(NoSource, kw_only=True, frozen=True, dict=True):
    path: types.WaveformInputPath
    file_format: types.Format = 'auto'
    file_metadata: types.FileMetadata = {}
    loop: types.FileLoop = False


class TDMSFileSourceSpec(NoSource, frozen=True, kw_only=True):
    path: types.WaveformInputPath


class ZarrIQSourceSpec(NoSource, frozen=True, kw_only=True):
    path: types.WaveformInputPath
    center_frequency: types.CenterFrequency
    select: types.ZarrSelect = {}


class Description(SpecBase, frozen=True, kw_only=True):
    summary: typing.Optional[str] = None
    location: typing.Optional[tuple[float, float, float]] = None
    signal_chain: typing.Optional[tuple[str, ...]] = tuple()
    version: str = 'unversioned'


class LoopBase(SpecBase, tag=str.lower, tag_field='kind', frozen=True, kw_only=True):
    field: str

    def get_points(self):
        raise NotImplementedError


class Range(LoopBase, frozen=True, kw_only=True):
    start: float
    stop: float
    step: float

    def get_points(self) -> list:
        import numpy as np

        if self.start == self.stop:
            return [self.start]

        a = np.arange(self.start, self.stop + self.step / 2, self.step)
        return list(a)


class Repeat(LoopBase, frozen=True, kw_only=True):
    field: str = '_sweep_index'
    count: int = 1

    def get_points(self) -> list[int]:
        return list(range(self.count))


class List(LoopBase, frozen=True, kw_only=True):
    values: tuple[typing.Any, ...]

    def get_points(self) -> list:
        return list(self.values)


class FrequencyBinRange(LoopBase, frozen=True, kw_only=True):
    start: float
    stop: float
    step: float

    def get_points(self) -> list[float]:
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


LoopSpec = typing.Union[Repeat, List, Range, FrequencyBinRange]


class Sink(_SlowHashSpecBase, frozen=True, kw_only=True):
    path: str = '{yaml_name}-{start_time}'
    log_path: typing.Optional[str] = None
    log_level: str = 'info'
    store: types.StoreFormat = 'directory'
    coord_aliases: dict[str, types.AliasMatch] = {}
    max_threads: typing.Optional[int] = None


class Extension(SpecBase, frozen=True, kw_only=True):
    sink: types.SinkClass | None = None
    import_path: typing.Optional[types.ExtensionPath] = None
    import_name: types.ModuleName = None


class Peripherals(SpecBase, frozen=True, kw_only=True):
    pass


class NoPeripherals(Peripherals, frozen=True, kw_only=True):
    pass


class ManualYFactorPeripheral(Peripherals, frozen=True, kw_only=True):
    enr: types.ENR
    ambient_temperature: types.AmbientTemperature


BundledAnalysis = _analysis.registry.tospec()
BundledAlignmentAnalysis = _analysis.registry.channel_sync_source.to_spec()


class SweepInfo(typing.NamedTuple):
    reuse_iq: bool
    loop_only_nyquist: bool


# forward references break msgspec when used with bindings, so this
# needs to be here after the bound classes have been defined
_TC = typing.TypeVar('_TC', bound=ResampledCapture)
_TP = typing.TypeVar('_TP', bound=Peripherals)
_TPC = typing.TypeVar('_TPC', bound=Peripherals)
_TS = typing.TypeVar('_TS', bound=Source)


class Sweep(SpecBase, typing.Generic[_TS, _TP, _TC], frozen=True, kw_only=True):
    source: _TS
    captures: tuple[_TC, ...] = tuple()
    loops: tuple[LoopSpec, ...] = ()
    analysis: BundledAnalysis = BundledAnalysis()  # type: ignore
    description: Description | str = ''
    extensions: Extension = Extension()
    sink: Sink = Sink()
    peripherals: _TP = typing.cast(_TP, Peripherals())

    info: typing.ClassVar[SweepInfo] = SweepInfo(reuse_iq=False, loop_only_nyquist=False)
    __bindings__: typing.ClassVar[typing.Any] = None

    def __post_init__(self):
        if len(self.loops) == 0:
            return

        from collections import Counter

        (which, howmany), *_ = Counter(l.field for l in self.loops).most_common(1)
        if howmany > 1:
            raise TypeError(f'more than one loop of capture field {which!r}')


class CalibrationSweep(
    Sweep[_TS, _TP, _TC],
    typing.Generic[_TS, _TP, _TC, _TPC],
    frozen=True,
    kw_only=True,
):
    """This specialized sweep is fed to the YAML file loader
    to specify the change in expected capture structure."""

    info: typing.ClassVar[SweepInfo] = SweepInfo(reuse_iq=True, loop_only_nyquist=True)
    calibration: _TPC | None = None

    def __post_init__(self):
        if len(self.captures) > 0:
            raise TypeError(
                'calibration sweeps may specify loops but not captures, only loops'
            )
        if self.source.calibration is not None:
            raise ValueError('source.calibration must be None for a calibration sweep')

        super().__post_init__()


# we really only need a dataclass for internal message-passing,
# but using msgspec.Struct here to support kw_only=True for python < 3.10.
#
# this does not perform validation, which is left to type-checking for this
# internal message passing
class AcquisitionInfo(msgspec.Struct, kw_only=True, frozen=True):
    """information about an acquired acquisition"""

    # duck-type methods and structure of SpecBase

    source_id: types.SourceID = ''

    def replace(self, **attrs) -> _Self:
        """returns a copy of self with changed attributes.

        See also:
            Python standard library `copy.replace`
        """
        return msgspec.structs.replace(self, **attrs)

    def todict(self) -> dict:
        """return a dictinary representation of `self`"""
        return msgspec.structs.asdict(self)

    @classmethod
    def fromdict(cls: type[_T], d: dict) -> _T:
        return cls(**d)


class SoapyAcquisitionInfo(AcquisitionInfo, kw_only=True, frozen=True):
    """extra coordinate information returned from an acquisition"""

    delay: typing.Optional[types.StartDelay] = None
    sweep_start_time: types.SweepStartTime | None
    start_time: types.StartTime | None
    backend_sample_rate: typing.Optional[types.BackendSampleRate]
    source_id: types.SourceID = ''


class FileAcquisitionInfo(AcquisitionInfo, kw_only=True, frozen=True):
    center_frequency: types.CenterFrequency = float('nan')
    backend_sample_rate: typing.Optional[types.BackendSampleRate]
    port: types.Port = 0
    gain: types.Gain = float('nan')
    source_id: types.SourceID = ''