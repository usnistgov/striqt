"""spec structs for sensor actions, their blocks, sources, peripherals, and captures.

msgspec resolves the field annotations at runtime. Field types are written as
`Union`/`Optional`, with every name they reference bound in this module.
"""

from __future__ import annotations as __

import dataclasses
import warnings
from typing import Any, cast, ClassVar, Generic, Literal, Optional, TYPE_CHECKING, Union

import msgspec

import striqt.analysis as sa
from striqt.analysis import Trigger

# pd must be bound here for msgspec to resolve the 'pd.Timestamp' types in `types`
from striqt.analysis.lib.util import pd  # noqa: F401
from striqt.analysis.specs import AnyAnalysis, AnySignalTrigger, Capture, SpecBase
from striqt.analysis.specs import frozendict

# re-exported through striqt.sensor.specs as the annotation of any analysis bundle
from striqt.analysis.specs import AnalysisGroup  # noqa: F401
from striqt.analysis.specs.helpers import SpecValidationError

from ..lib.typing import TypeVar, SS, SP, SC
from . import types


if TYPE_CHECKING:
    _T = TypeVar('_T')
    from typing_extensions import Self as _Self

    import SoapySDR  # type: ignore[import-not-found, unused-ignore]


@sa.util.lru_cache()
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


# %% Capture specs


class SensorCapture(Capture, frozen=True, kw_only=True):
    """Capture specification for a generic waveform with resampling support"""

    # acquisition
    port: types.Port
    lo_shift: types.LOShift = 'none'
    host_resample: bool = True
    backend_sample_rate: Optional[types.BackendSampleRate] = None
    adjust_analysis: types.AnalysisAdjustments = msgspec.field(default_factory=dict)
    external_lo_frequency: types.LOFrequency = None


class SoapyCapture(SensorCapture, frozen=True, kw_only=True):
    center_frequency: types.CenterFrequency
    gain: types.Gain

    def __post_init__(self):
        super().__post_init__()
        _validate_multichannel(self.port, self.gain)


class SingleToneCapture(SensorCapture, frozen=True, kw_only=True):
    frequency_offset: types.FrequencyOffset = 0
    snr: Optional[types.SNR] = None


class DiracDeltaCapture(SensorCapture, frozen=True, kw_only=True):
    time: types.TimeOffset = 0
    power: types.Power = 0


class SawtoothCapture(SensorCapture, kw_only=True, frozen=True, dict=True):
    period: types.Period = 0.01
    power: types.Power = 0


class NoiseCapture(SensorCapture, kw_only=True, frozen=True, dict=True):
    noise_psd: types.PSD = 1e-17


class FileCapture(SensorCapture, frozen=True, kw_only=True):
    """Capture specification read from a file, with support for None sentinels"""

    def __post_init__(self):
        super().__post_init__()
        if self.backend_sample_rate is not None:
            raise TypeError('backend_sample_rate is fixed by the file source')


# %% Source specs


class Source(SpecBase, frozen=True, kw_only=True):
    """run-time characteristics of the radio that are left invariant during a sweep"""

    # acquisition
    master_clock_rate: types.MasterClockRate

    # synchronization and triggering
    trigger_strobe: Union[float, None] = None
    signal_trigger: Union[str, AnySignalTrigger, None] = None

    array_backend: types.ArrayBackend = 'numpy'

    # unannotated so that subclasses can declare them as schema fields; source
    # implementations read them
    gapless = False
    calibration = None
    if TYPE_CHECKING:
        gapless: bool
        calibration: str | None

    # validation data
    transient_holdoff_time: ClassVar[float] = 0
    stream_all_rx_ports: ClassVar[Union[bool, None]] = False
    transport_dtype: ClassVar[types.TransportDType] = 'float32'


class SoapySource(Source, frozen=True, kw_only=True):
    calibration: Optional[str] = None
    time_source: types.TimeSource = 'host'
    time_sync_at: types.TimeSyncOn = 'acquire'
    clock_source: types.ClockSource = 'internal'
    receive_retries: types.ReceiveRetries = 0
    adc_overload_limit: types.ADCOverloadLimit = -1
    if_overload_limit: types.IFOverloadLimit = None

    # True when one clock drives acquisition on all RX ports
    shared_rx_sample_clock = True
    rx_enable_delay = 0.0
    gapless: types.GaplessRepeat = False

    def __post_init__(self):
        from striqt.analysis import registry

        super().__post_init__()

        if not self.gapless:
            pass
        elif self.time_sync_at == 'acquire':
            raise ValueError('time_sync_at must be "open" when gapless')
        elif self.receive_retries > 0:
            raise ValueError('receive_retries must be 0 when gapless is enabled')
        trigger = self.signal_trigger
        if trigger is None:
            pass
        elif isinstance(trigger, AnySignalTrigger):
            if len(trigger.to_dict()) != 1:
                raise ValueError('signal_trigger must enable exactly one trigger')
        elif trigger not in registry.signal_trigger:
            registered = {k for k in registry.signal_trigger if isinstance(k, str)}
            raise ValueError(f'signal_trigger must be one of {registered!r}')


class FunctionSource(Source, kw_only=True, frozen=True):
    # make these configurable, to support matching hardware for warmup sweeps
    num_rx_ports: int
    stream_all_rx_ports: ClassVar[bool] = False
    transport_dtype: ClassVar[types.TransportDType] = 'complex64'


class NoSource(Source, frozen=True, kw_only=True):
    # make these configurable, to support matching hardware for warmup sweeps
    num_rx_ports: int
    stream_all_rx_ports: ClassVar[bool] = False


class MATSource(Source, kw_only=True, frozen=True):
    path: types.WaveformInputPath
    file_format: types.Format = 'auto'
    file_metadata: Union[types.FileMetadata, None] = None
    loop: types.FileLoop = False
    transport_dtype: ClassVar[types.TransportDType] = 'complex64'
    key: Union[str, None] = None


class TDMSSource(Source, frozen=True, kw_only=True):
    path: types.WaveformInputPath
    transport_dtype: ClassVar[types.TransportDType] = 'complex64'


class ZarrIQSource(Source, frozen=True, kw_only=True):
    path: types.WaveformInputPath
    center_frequency: types.CenterFrequency
    select: types.ZarrSelect = dict()
    transport_dtype: ClassVar[types.TransportDType] = 'complex64'


# %% Sweep metadata - not used in sweep control flow


class Description(SpecBase, frozen=True, kw_only=True):
    summary: Optional[str] = None
    version: str = 'unversioned'


class SharedPlotOptions(
    SpecBase, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    # further validation in striqt.figures.specs subclass
    style: Union[str, None] = None
    col: Union[str, None] = 'port'
    row: Union[str, None] = None
    col_label_format: str = 'Port {port}'
    row_label_format: Union[str, None] = None
    col_wrap: Union[int, None] = None
    suptitle_fmt: str = '{start_time}'
    filename_fmt: str = '{name} {start_time}.svg'


class DataOptions(SpecBase, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # further validation in striqt.figures.specs subclass
    groupby_dims: tuple[str, ...] = ()
    sweep_index: int
    query: Union[str, None] = None
    select: dict[str, Any] = msgspec.field(default_factory=dict)


class PlotOptions(SpecBase, frozen=True, kw_only=True, forbid_unknown_fields=True):
    # further validation in striqt.figures.specs subclass
    data: DataOptions
    plotter: SharedPlotOptions
    variables: dict[str, dict[str, Any]] = msgspec.field(default_factory=dict)


# %% Sequencing specs


class LoopBase(SpecBase, tag=str.lower, tag_field='kind', frozen=True, kw_only=True):
    field: Union[str, None]
    isin: types.IsIn = 'capture'

    def get_points(self) -> list:
        raise NotImplementedError


class Range(LoopBase, frozen=True, kw_only=True):
    field: str
    start: float
    stop: float
    step: float

    def get_points(self) -> list:
        import numpy as np

        if self.start == self.stop:
            return [self.start]

        a = np.arange(self.start, self.stop + self.step / 2, self.step)
        return a.tolist()


class Repeat(LoopBase, frozen=True, kw_only=True):
    field: None = None
    count: int = 1

    def get_points(self) -> list[int]:
        return list(range(self.count))


class List(LoopBase, frozen=True, kw_only=True):
    field: str
    values: tuple[Union[AdjustCapturesType, str, float, int, bool, None], ...]

    def get_points(self) -> list:
        return list(self.values)


class FrequencyBinRange(LoopBase, frozen=True, kw_only=True):
    field: str
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
        return points.tolist()


LoopSpec = Union[Repeat, List, Range, FrequencyBinRange]


# %% Peripheral specs
class Peripherals(SpecBase, frozen=True, kw_only=True):
    pass


class NoPeripherals(Peripherals, frozen=True, kw_only=True):
    pass


class ManualYFactorSpec(Peripherals, frozen=True, kw_only=True):
    """the `calibration:` block of a `ManualYFactorCalibration`.

    `implied_loops` names capture fields that the operator varies by hand between
    captures. The saved corrections are indexed by these fields ahead of the loop
    fields. `detector_period` is the bin of the generated
    `channel_power_time_series` analysis. The rms detector is averaged over the bins
    before the Y-factor is formed, and only the ``papr_on``/``papr_off`` diagnostics
    depend on the bin. When None, the bin is the one `duration` of the captures and
    loops, and a bound spec with more than one duration is rejected.
    """

    enr: types.ENR
    ambient_temperature: types.AmbientTemperature
    implied_loops: tuple[str, ...] = ()
    detector_period: Optional[types.DetectorPeriod] = None


ManualYFactorPeripheral = ManualYFactorSpec


# %% Top-level Sweep specifications
# compatibility names for the bundles, which striqt.analysis.specs defines
BundledAnalysis = AnyAnalysis
BundledTriggers = AnySignalTrigger


class Sink(SpecBase, frozen=True, kw_only=True):
    """the output block of every action spec.

    `path` is where the action writes, after the ``{field}`` templates of
    `specs.paths.PathFormatter` are expanded. The action runner selects the file
    format. A `path` of None writes nothing. `log_path` and `log_level` configure
    the log file and do not depend on `path`.
    """

    path: Optional[str] = '{spec_name}-{start_time}'
    log_path: Optional[str] = None
    log_level: str = 'info'


class ZarrSink(Sink, frozen=True, kw_only=True):
    """the output block of a `Sweep`: a zarr store appended along `capture`.

    `store` selects a directory or zip store. The remaining fields tune the write
    batching and chunking of the store.
    """

    store: types.StoreFormat = 'directory'
    max_threads: Optional[int] = None
    batched_write_count: int = 1
    max_chunk_bytes: int = 50000000


class Extension(SpecBase, frozen=True, kw_only=True):
    """the `extensions:` block of an action spec.

    `import_path` is inserted into ``sys.path`` and `import_name` is imported before
    the rest of the spec is decoded. The module may bind the ``sensor_binding`` that
    the spec names.
    """

    import_path: Optional[types.ExtensionPath] = None
    import_name: types.ModuleName = None


class SweepOptions(SpecBase, frozen=True, kw_only=True):
    reuse_iq: bool = False
    loop_only_nyquist: bool = False
    skip_warmup: types.SkipWarmup = True


SWEEP_TAG_FIELD = 'sensor_binding'


class CaptureRemap(SpecBase, frozen=True, kw_only=True):
    key: Union[str, tuple[str, ...]]
    lookup: dict[Union[tuple[Any, ...], Any], Any]
    required: bool = True
    default: Any = msgspec.UNSET

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.key, tuple) or len(self.key) == 1:
            super().__post_init__()
            return

        # convert any stringified json array keys into tuples
        lookup = dict(self.lookup)
        for k in self.lookup.keys():
            if not isinstance(k, tuple):
                new_key = msgspec.json.decode(k, type=tuple)
                lookup[new_key] = lookup.pop(k)
        lookup = sa.specs.helpers.freeze(lookup)
        msgspec.structs.force_setattr(self, 'lookup', lookup)
        super().__post_init__()


_CaptureMapScalarType = Union[float, int, str, bool, None]
_AdjustSourceCapturesMap = dict[str, Union[CaptureRemap, _CaptureMapScalarType]]
AdjustCapturesType = dict[
    Union[types.SourceID, Literal['defaults']], _AdjustSourceCapturesMap
]


class ActionSpec(SpecBase, Generic[SS, SP], frozen=True, kw_only=True):
    """the top-level spec that an action runs.

    An instance is decoded from a file against the tagged class of a sensor binding
    or constructed from the API. The `source`, `peripherals`, and `sink` blocks name
    the resources that every action opens. `source` and `peripherals` are the spec
    types of the binding named by the ``sensor_binding`` tag. `extensions` names an
    out-of-tree module that the readers in `striqt.sensor.lib.io` import before
    decoding the rest of the spec.
    """

    # tagged unions used in bindings introduce the tag below as well.
    # msgspec uses this to choose the subclass to decode from.
    #
    # See: ./bindings.py
    #
    # sensor_binding: str

    description: Description = Description()
    source: SS
    peripherals: SP = cast(SP, Peripherals())
    extensions: Extension = Extension()
    sink: Sink = Sink()

    # set only on the subclasses that bind_sensor creates
    schema: ClassVar[Any] = None
    sensor: ClassVar[Any] = None
    controller: ClassVar[Any] = None
    sweep_cls: ClassVar[Any] = None

    @classmethod
    def check_schema(cls, schema: Any) -> None:
        """raise if the spec types of `schema` cannot carry this action.

        `bind_sensor` calls this once per binding. The base accepts every schema.

        Args:
            schema: the `Schema` being bound

        Raises:
            TypeError: if a spec type of `schema` lacks a field the action requires
        """


class CapturePlan(ActionSpec[SS, SP], Generic[SS, SP, SC], frozen=True, kw_only=True):
    """an `ActionSpec` with the `captures`, `loops`, and `adjust_captures` blocks.

    `loops` expands over `captures` into the sequence of captures that the action
    runs (see `sequencing.loop_captures`), and `adjust_captures` applies per-source
    overrides to each expanded capture. Construction rejects a `repeat` loop that is
    not the outermost entry, two loops over one field, and a loop over a field that
    the listed captures set to different values. Lookup keys in `adjust_captures`
    are converted to the type of the capture field they key on.
    """

    captures: tuple[SC, ...] = ()
    loops: tuple[LoopSpec, ...] = ()
    adjust_captures: AdjustCapturesType = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        from . import action, sequencing

        # convert the label keys before super().__post_init__ freezes the struct
        fixed_labels = sequencing._convert_label_lookup_keys(self)
        msgspec.structs.force_setattr(self, 'adjust_captures', fixed_labels)

        super().__post_init__()
        action._validate_loops(self.loops)
        action._validate_loop_capture_collisions(self.loops, self.captures)


class Sweep(CapturePlan[SS, SP, SC], frozen=True, kw_only=True):
    """the spec of a sweep, and the form every action opens its resources for.

    `analysis` is evaluated on each expanded capture, with the capture's
    `adjust_analysis` applied on top. `options` selects the warmup, IQ reuse, and
    the `loop_only_nyquist` filter of the expansion. `plot_hint` is read only by
    `striqt.figures`. `sink` is a `ZarrSink`, which narrows the `Sink` of
    `ActionSpec`. Construction validates each unique capture against the
    registered measurements and the `correct_iq` signal path (see
    `action.validate_sweep`).
    """

    sink: ZarrSink = ZarrSink()
    plot_hint: Union[PlotOptions, None] = None
    analysis: AnyAnalysis = AnyAnalysis()
    options: SweepOptions = SweepOptions(reuse_iq=False, loop_only_nyquist=False)

    def __post_init__(self) -> None:
        from . import action

        super().__post_init__()
        action.validate_sweep(self)
        action.validate_action(self)


class ManualYFactorCalibration(CapturePlan[SS, SP, SC], frozen=True, kw_only=True):
    """the spec of a Y-factor calibration with a hand-switched noise diode.

    The `calibration:` block is a `ManualYFactorSpec`. The runner acquires the
    expanded captures at each port with the diode off and then on, and generates
    the analysis itself. The spec has no `options` block. An `analysis` block is
    accepted, discarded, and reported with a `DeprecationWarning`. More than one
    capture is accepted only when `calibration.implied_loops` names the fields the
    captures vary, and `source.calibration` must be null. A bound spec is also
    rejected when `ManualYFactorRunner.validate` cannot build its sweep.
    """

    calibration: ManualYFactorSpec
    analysis: Any = None

    def __post_init__(self) -> None:
        from . import action

        super().__post_init__()

        if self.analysis is not None:
            warnings.warn(
                'the `analysis` block of a calibration is ignored: the calibration '
                'generates its own analysis. The block is accepted until the next '
                'release.',
                DeprecationWarning,
                stacklevel=2,
            )
            # the block is a dict, which the frozen struct's hash cannot include
            msgspec.structs.force_setattr(self, 'analysis', None)

        if len(self.captures) > 1 and not self.calibration.implied_loops:
            raise SpecValidationError(
                'Expected at most one capture in a calibration sweep, unless the '
                'calibration peripheral declares `implied_loops`',
                ('.captures',),
            )
        if self.source.calibration is not None:
            raise SpecValidationError(
                'Expected `null`, since a calibration sweep cannot itself be calibrated',
                ('.source', '.calibration'),
            )
        action.validate_action(self)


# the cyclic prefix model is validated at 30 kHz only (strict xfails in
# tests/waveform/test_waveform_comms_phy_3gpp.py cover 15 and 60 kHz)
SUPPORTED_SUBCARRIER_SPACINGS = frozenset({30e3})


class CellChannel(SpecBase, frozen=True, kw_only=True):
    """one 5G NR channel of the `cell_channels` table of a `Cellular5GNRSweep`.

    A capture whose `center_frequency` equals the channel's is acquired at the
    sample rate `striqt.waveform.comms_5gnr.nr_sample_rate` derives from
    `channel_bandwidth` and `subcarrier_spacing`, at or above `min_sample_rate`, with
    `analysis_bandwidth` equal to `channel_bandwidth`. The remaining fields reach the
    cellular measurements through the capture's `adjust_analysis`.
    `ssb_case` None selects the TS 38.213 §4.1 case from the spacing
    (`striqt.sensor.lib.actions.cellular.ssb_symbol_indexes`): ``'a'`` at 15 kHz and
    ``'c'`` at 30 kHz. A `tdd_frame_slots` string is one character per
    slot of a 10 ms frame, or a single character for every slot. Only 30 kHz spacing
    is accepted.
    """

    center_frequency: types.CenterFrequencyScalar
    channel_bandwidth: types.ChannelBandwidth
    subcarrier_spacing: types.CellularSubcarrierSpacing = 30e3
    ssb_frequency_offset: types.SSBFrequencyOffset = 0.0
    ssb_case: Optional[types.CellSSBSymbolIndexes] = None
    channel_guard_bandwidths: types.ChannelGuardBandwidths = (0.0, 0.0)
    tdd_frame_slots: types.CellularFrame = None
    tdd_special_symbols: types.CellularSpecialSymbols = None
    min_sample_rate: Optional[types.MinSampleRate] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.subcarrier_spacing not in SUPPORTED_SUBCARRIER_SPACINGS:
            raise SpecValidationError(
                f'Expected one of {sorted(SUPPORTED_SUBCARRIER_SPACINGS)!r}',
                ('.subcarrier_spacing',),
            )
        slots = round(10 * self.subcarrier_spacing / 15e3)
        if self.tdd_frame_slots is not None and len(self.tdd_frame_slots) not in (
            1,
            slots,
        ):
            raise SpecValidationError(
                f'Expected 1 or {slots} slot characters for a 10 ms frame at this '
                'subcarrier spacing',
                ('.tdd_frame_slots',),
            )


class CellularSpectrogramSpec(SpecBase, frozen=True, kw_only=True):
    window: types.WindowType
    lo_bandstop: Optional[types.LOBandstop] = None
    time_aperture: Optional[float] = None
    trim_stopband: bool = True


class CellularPowerSpectralDensitySpec(SpecBase, frozen=True, kw_only=True):
    window: types.WindowType
    time_statistic: tuple[Union[str, float], ...] = ('mean',)
    lo_bandstop: Optional[types.LOBandstop] = None
    trim_stopband: bool = True


class CellularResourcePowerHistogramSpec(SpecBase, frozen=True, kw_only=True):
    window: types.WindowType
    power_low: types.PowerBinMin
    power_high: types.PowerBinMax
    power_resolution: types.PowerBinStep
    average_rbs: types.CellularAverageRBs = False
    average_slots: types.CellularAverageSlots = False
    cyclic_prefix: types.CellularCyclicPrefix = 'normal'
    lo_bandstop: Optional[types.LOBandstop] = None


class CellularPSSCorrelationSpec(SpecBase, frozen=True, kw_only=True):
    discovery_periodicity: float = 20e-3
    delay: float = 0.0
    shared_spectrum: bool = False
    max_block_count: Optional[int] = 1


class CellularPSSSyncSpec(SpecBase, frozen=True, kw_only=True):
    discovery_periodicity: float = 20e-3
    delay: float = 0.0
    shared_spectrum: bool = False
    per_port: types.PerPort = False
    max_beams: Optional[types.MaxBeams] = None


class CellularSSBSpectrogramSpec(SpecBase, frozen=True, kw_only=True):
    window: types.WindowType = 'blackmanharris'
    discovery_periodicity: float = 20e-3
    max_block_count: Optional[int] = None
    lo_bandstop: Optional[types.LOBandstop] = None


class CellularCyclicChannelPowerSpec(SpecBase, frozen=True, kw_only=True):
    power_detectors: types.PowerDetectors = ('rms', 'peak')
    cyclic_statistics: types.CyclicStatistics = ('min', 'mean', 'max')


class CellularChannelPowerTimeSeriesSpec(SpecBase, frozen=True, kw_only=True):
    detector_period: Optional[types.DetectorPeriod] = None
    power_detectors: types.PowerDetectors = ('rms', 'peak')


class CellularChannelPowerHistogramSpec(SpecBase, frozen=True, kw_only=True):
    power_low: types.PowerBinMin
    power_high: types.PowerBinMax
    power_resolution: types.PowerBinStep
    detector_period: Optional[types.DetectorPeriod] = None
    power_detectors: types.PowerDetectors = ('rms', 'peak')


class CellularCyclicAutocorrelationSpec(SpecBase, frozen=True, kw_only=True):
    frame_range: Union[int, tuple[int, int]] = (0, 1)
    symbol_range: Union[int, tuple[int, Union[int, None]]] = (0, None)


class CellularIQWaveformSpec(SpecBase, frozen=True, kw_only=True):
    start_time_sec: Optional[float] = None
    stop_time_sec: Optional[float] = None


class Cellular5GNRAnalysis(SpecBase, frozen=True, kw_only=True):
    """the `analysis:` block of a `Cellular5GNRSweep`.

    Each field enables one measurement and names only its free parameters. The
    parameters that follow from the subcarrier spacing of the channels are filled
    when the sweep is built: the SCS/2 frequency resolution with the 13/28 overlap
    and 15/28 window fill of the OFDM symbol grid, 1 resource block (spectrogram) or
    1 subcarrier (power spectral density) of integration bandwidth, the 10 ms cyclic
    period, and 1 OFDM symbol as the default detector period. A None field disables
    its measurement.
    """

    spectrogram: Optional[CellularSpectrogramSpec] = None
    power_spectral_density: Optional[CellularPowerSpectralDensitySpec] = None
    resource_power_histogram: Optional[CellularResourcePowerHistogramSpec] = None
    pss_correlation: Optional[CellularPSSCorrelationSpec] = None
    pss_sync: Optional[CellularPSSSyncSpec] = None
    ssb_spectrogram: Optional[CellularSSBSpectrogramSpec] = None
    cyclic_channel_power: Optional[CellularCyclicChannelPowerSpec] = None
    channel_power_time_series: Optional[CellularChannelPowerTimeSeriesSpec] = None
    channel_power_histogram: Optional[CellularChannelPowerHistogramSpec] = None
    cyclic_autocorrelation: Optional[CellularCyclicAutocorrelationSpec] = None
    iq_waveform: Optional[CellularIQWaveformSpec] = None


class Cellular5GNRSweep(CapturePlan[SS, SP, SC], frozen=True, kw_only=True):
    """the spec of a sweep over 5G NR channels.

    `captures` are the binding's own capture type and name a channel by
    `center_frequency`; `cell_channels` maps a label to the `CellChannel` with that
    center frequency. The runner derives each capture's `sample_rate` and
    `analysis_bandwidth` from its channel and generates the `Sweep` analysis from
    `analysis`. A capture leaves its own `sample_rate` and `analysis_bandwidth` at
    their defaults. Every channel a capture uses must share one subcarrier spacing.
    A loop over `center_frequency` is rejected; the channels are listed as captures
    instead. The binding's capture type must have a `center_frequency` field.
    """

    sink: ZarrSink = ZarrSink()
    plot_hint: Union[PlotOptions, None] = None
    options: SweepOptions = SweepOptions()
    cell_channels: dict[str, CellChannel]
    analysis: Cellular5GNRAnalysis = Cellular5GNRAnalysis()

    def __post_init__(self) -> None:
        from . import action

        # before the generic loop checks of CapturePlan, whose collision error would
        # otherwise name the same loop with a less specific message
        for i, loop in enumerate(self.loops):
            if getattr(loop, 'field', None) == 'center_frequency':
                raise SpecValidationError(
                    'Expected no loop over center_frequency: list the channels as '
                    'captures',
                    (f'.loops[{i}]', '.field'),
                )
        super().__post_init__()
        seen: dict[float, str] = {}
        for label, channel in self.cell_channels.items():
            other = seen.setdefault(channel.center_frequency, label)
            if other != label:
                raise SpecValidationError(
                    f'Expected unique center frequencies, but {other!r} and '
                    f'{label!r} share {channel.center_frequency!r}',
                    ('.cell_channels', f'[{label!r}]', '.center_frequency'),
                )
        action.validate_action(self)

    @classmethod
    def check_schema(cls, schema: Any) -> None:
        if 'center_frequency' not in schema.capture.__struct_fields__:
            raise TypeError(
                f'{cls.__name__} needs a capture type with a center_frequency field, '
                f'but {schema.capture.__name__} has none'
            )


# %% Non-Sweep specs
# a msgspec.Struct stands in for a dataclass here to support kw_only=True on
# python < 3.10. These structs are internal message passing and are not
# validated; type checking covers them.
class AcquisitionInfo(msgspec.Struct, kw_only=True, frozen=True):
    """information about an acquisition"""

    # duck-type methods and structure of SpecBase

    source_id: types.SourceID = ''
    sweep_index: int = 0
    capture_index: int = 0

    signal_trigger: Union[Trigger, None] = None

    def replace(self, **attrs) -> _Self:
        """returns a copy of self with changed attributes.

        See also:
            Python standard library `copy.replace`
        """
        return msgspec.structs.replace(self, **attrs)

    def to_dict(self) -> dict:
        """return a dictionary representation of `self`"""
        return msgspec.structs.asdict(self)

    @classmethod
    def from_dict(cls: type[_T], d: dict) -> _T:
        try:
            return cls(**d)
        except:
            print(d)
            raise


class SoapyAcquisitionInfo(AcquisitionInfo, kw_only=True, frozen=True):
    """extra coordinate information returned from an acquisition"""

    sweep_start_time: Union[types.SweepStartTime, None] = None
    start_time: Union[types.StartTime, None]
    backend_sample_rate: Optional[types.BackendSampleRate]
    source_id: types.SourceID = ''


class FileAcquisitionInfo(AcquisitionInfo, kw_only=True, frozen=True):
    center_frequency: types.CenterFrequency = float('nan')
    backend_sample_rate: types.BackendSampleRate
    port: types.Port = 0
    gain: types.Gain = float('nan')
    source_id: types.SourceID = ''


class SourceInfo(SpecBase, kw_only=True, frozen=True, cache_hash=True):
    num_rx_ports: Union[int, None]
    retries: Union[int, None] = None

    def min_port_count(self, tuple_size: int):
        if self.num_rx_ports is None:
            return tuple_size
        else:
            return self.num_rx_ports
