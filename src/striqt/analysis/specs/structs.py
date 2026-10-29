"""data structures that specify waveform characteristics"""

from __future__ import annotations as __

import fractions
import typing
from math import inf
from typing import Optional, Union

import msgspec
from typing_extensions import Self

import striqt.waveform as sw

from . import helpers, types

_TS = typing.TypeVar('_TS', bound='SpecBase')

# msgspec.structs.replace only calls __post_init__ from 0.21, and 0.21
# dropped Python 3.9
_REPLACE_RUNS_POST_INIT = tuple(int(s) for s in msgspec.__version__.split('.')[:2]) >= (
    0,
    21,
)


class SpecBase(
    msgspec.Struct,
    kw_only=True,
    frozen=True,
    forbid_unknown_fields=True,
    cache_hash=True,
):
    """the base of every spec struct.

    An instance is a frozen, keyword-only `msgspec.Struct` that rejects unknown
    fields. Construction converts list and dict field values to tuple and
    `frozendict`. An instance round-trips through `to_dict` and `from_dict`
    unchanged.
    """

    def replace(self, **attrs) -> Self:
        """returns a copy of self with changed attributes.

        See also:
            Python standard library `copy.replace`
        """
        if len(attrs) == 0:
            return self
        new = msgspec.structs.replace(self, **attrs)
        if not _REPLACE_RUNS_POST_INIT:
            new.__post_init__()
        return new.validate()

    def to_dict(self, unfreeze: bool = False, allow_tuple_keys: bool = True) -> dict:
        """return the field values of `self` as a dict of builtin values.

        Args:
            unfreeze: True converts nested tuple and `frozendict` values to list and
                dict
            allow_tuple_keys: False renders tuple dict keys as JSON array strings

        Returns:
            a dict keyed by field name; `fractions.Fraction` values are strings and
            `frozendict` values are dicts
        """
        if allow_tuple_keys:
            enc_hook = helpers._enc_hook
        else:
            enc_hook = helpers._enc_hook_no_tuple_keys

        map = msgspec.to_builtins(self, enc_hook=enc_hook)

        if unfreeze:
            depths = helpers.inspect_freeze_depths(type(self))
            for name, depth in depths.items():
                map[name] = helpers.unfreeze(map[name], depth)

        return map

    @classmethod
    def from_dict(cls, d: dict | helpers.frozendict) -> Self:
        """return an instance of `cls` converted from the field values in `d`.

        Args:
            d: field values keyed by name; a float for a `fractions.Fraction` field
                converts to the fraction of its decimal representation

        Raises:
            msgspec.ValidationError: when a required field is missing, a key is not a
                field, or a value does not convert to the annotated type
        """
        return helpers.convert_dict(d, type=cls)

    @classmethod
    def from_spec(cls, other: SpecBase) -> Self:
        return helpers.convert_spec(other, type=cls)

    def validate(self) -> Self:
        """return an equal instance rebuilt through `to_dict` and `from_dict`.

        Raises:
            msgspec.ValidationError: when a field value does not satisfy the annotated
                type or constraints
        """
        return _validate(self)

    def __post_init__(self):
        for name, depth in helpers.inspect_freeze_depths(type(self)).items():
            v = getattr(self, name)
            if isinstance(v, (tuple, dict, list)):
                frozen = helpers.freeze(v, depth + 1)
                msgspec.structs.force_setattr(self, name, frozen)


@sw.util.lru_cache(1024)
def _validate(spec: _TS) -> _TS:
    return spec.from_dict(spec.to_dict())


class Capture(SpecBase, kw_only=True, frozen=True):
    """bare minimum information about an IQ acquisition"""

    # acquisition
    duration: types.Duration = 0.1
    sample_rate: types.SampleRate = 15.36e6
    analysis_bandwidth: types.AnalysisBandwidth = inf

    def __post_init__(self):
        from ..lib import util

        super().__post_init__()

        if not util.isroundmod(self.duration * self.sample_rate, 1):
            raise ValueError(
                f'duration {self.duration!r} is not an integer multiple of sample period'
            )


class AnalysisCapture(Capture, kw_only=True, frozen=True):
    """the projection of a capture onto the fields that a measurement may read.

    Two captures with equal values in these fields project to equal instances.
    """

    center_frequency: Union[types.CenterFrequency, None] = None


class AnalysisFilter(SpecBase, kw_only=True, frozen=True):
    nfft: int = 8192
    window: types.WindowType = 'hamming'
    nfft_out: int | None = None


class FilteredCapture(Capture, kw_only=True, frozen=True):
    # filtering and resampling
    analysis_filter: AnalysisFilter = msgspec.field(default_factory=AnalysisFilter)


class Analysis(SpecBase, kw_only=True, frozen=True):
    """the base of every measurement spec.

    The fields of a subclass are the keyword arguments of one measurement.
    """


class ErrorBound(SpecBase, kw_only=True, frozen=True):
    """an error bound on the elements of one output.

    `rms` bounds the error over all elements and `peak` the error at any one
    element, both in the units of the `Tolerance` field.
    """

    rms: float
    peak: float


class Tolerance(SpecBase, kw_only=True, frozen=True):
    """the roundoff error budget of one measurement output.

    The budget depends on the capture and the spec alone, not on the output
    values. `on_peak` bounds the error of the elements at the peak of the output.
    `off_peak_dBc` bounds the error of the other elements as a function of the
    level of each below the peak. Elements more than `off_peak_dBc.peak` below the
    peak are not bounded. The peak is the matched-filter bin of the matched input,
    e.g., the FFT bin at the frequency of a tone or the detector sample at the time
    of an impulse.
    """

    units: typing.Annotated[
        str, helpers.Meta("units of `on_peak`: 'dB', or the linear units of the output")
    ]
    rtol: typing.Annotated[
        float, helpers.Meta('relative error bound on each output value, as a ratio')
    ]
    on_peak: typing.Annotated[
        ErrorBound,
        helpers.Meta(
            'absolute error bound, in `units`, on the elements at the peak of the '
            'output'
        ),
    ]
    off_peak_dBc: typing.Annotated[
        Union[ErrorBound, None],
        helpers.Meta(
            'levels below the output peak, as negative values, at which the rms and '
            'the worst-case amplitude errors equal the amplitude of the element, or '
            'None to apply `on_peak` to every element',
            'dBc',
        ),
    ] = None


class AnalysisGroup(SpecBase, kw_only=True, frozen=True):
    """base class for a defining set of Analysis specs"""


# %% Spectral analysis
class FrequencyAnalysisSpecBase(
    Analysis,
    kw_only=True,
    frozen=True,
    dict=True,
):
    """the STFT settings shared by the spectrum measurements.

    The FFT size is the capture `sample_rate` divided by `frequency_resolution`,
    and the quotient must be a whole number. `fractional_overlap` and `window_fill`
    are the hop and the zero padding of each window as fractions of that size.
    `integration_bandwidth` is the width of the adjacent bins summed after the FFT,
    a multiple of `frequency_resolution`, or None for no summing. `trim_stopband`
    True limits the frequency axis to the `analysis_bandwidth` of the capture when
    that bandwidth is finite.
    """

    window: types.WindowType
    frequency_resolution: float
    fractional_overlap: fractions.Fraction = 0  # ty: ignore[invalid-assignment]
    window_fill: fractions.Fraction = 1  # ty: ignore[invalid-assignment]
    integration_bandwidth: typing.Optional[float] = None
    trim_stopband: bool = True
    lo_bandstop: Union[types.LOBandstop, None] = None


class Cellular5GNRSSBSpectrogram(Analysis, kw_only=True, frozen=True):
    """
    sample_rate (float): output sample rate for the resampled synchronization waveform (samples/s)
    discovery_periodicity (float): time period between synchronization blocks (s)
    frequency_offset (float or dict[float, float]):
        center frequency offset (see notes)
    max_block_count: number of synchronization blocks to evaluate
    """

    subcarrier_spacing: types.CellularSubcarrierSpacing

    # ssb parameters
    sample_rate: float = 15.36e6 / 2
    discovery_periodicity: float = 20e-3
    frequency_offset: float = 0
    max_block_count: typing.Optional[int] = None

    # spectrogram info
    window: types.WindowType = 'blackmanharris'
    lo_bandstop: Union[types.LOBandstop, None] = None


# %% Cellular 5G NR synchronizatino
class _Cellular5GNRSSBCorrelator(Analysis, kw_only=True, frozen=True):
    """
    sample_rate (float): output sample rate for the resampled synchronization waveform (samples/s)
    discovery_periodicity (float): time period between synchronization blocks (s)
    frequency_offset (float or dict[float, float]):
        center frequency offset (see notes)
    shared_spectrum:
        whether to follow the 3GPP "shared spectrum" synchronizatio block layout
    delay: minimum time delay (in s) to start of SSB set
    """

    subcarrier_spacing: types.CellularSubcarrierSpacing
    sample_rate: float = 15.36e6 / 2
    discovery_periodicity: float = 20e-3
    frequency_offset: float = 0
    shared_spectrum: bool = False
    delay: float = 0
    symbol_indexes: types.CellSSBSymbolIndexes = 'auto'
    max_lag_symbols: Union[types.MaxLagSymbols, None] = None

    def __post_init__(self):
        super().__post_init__()
        if not sw.arrays.isroundmod(self.delay * self.sample_rate, 1):
            raise msgspec.ValidationError('delay * sample_rate must be an integer')


class Cellular5GNRPSSCorrelator(_Cellular5GNRSSBCorrelator, kw_only=True, frozen=True):
    # same as SSS, but the registry requires unique spec types
    max_block_count: typing.Optional[int] = 1


class Cellular5GNRSSSCorrelator(_Cellular5GNRSSBCorrelator, kw_only=True, frozen=True):
    max_block_count: typing.Optional[int] = 1


class _Cellular5GNRSSBSync(_Cellular5GNRSSBCorrelator, frozen=True, kw_only=True):
    window_fill: types.WindowFill = 1
    per_port: types.PerPort = False
    max_beams: Union[types.MaxBeams, None] = None


class Cellular5GNPSSSync(_Cellular5GNRSSBSync, kw_only=True, frozen=True):
    # same as SSS, but the registry requires unique spec types
    pass


class Cellular5GNSSSSync(_Cellular5GNRSSBSync, kw_only=True, frozen=True):
    pass


class Spectrogram(FrequencyAnalysisSpecBase, kw_only=True, frozen=True):
    """
    time_aperture (float):
        if specified, binned RMS averaging is applied along time axis in the
        spectrogram to yield this coarser resolution (s)
    """

    time_aperture: typing.Optional[float] = None


def _validate_range(
    name: str, value: int | tuple[int, int | None], path: tuple[str, ...] = ()
) -> None:
    if not isinstance(value, tuple):
        return
    start, stop = value
    if stop is None:
        if start > 0:
            raise helpers.SpecValidationError(
                'Expected an end value, since `start` is greater than 0',
                path + (f'.{name}',),
            )
        return
    if stop < start:
        raise helpers.SpecValidationError(
            'Expected an end value at or after `start`', path + (f'.{name}',)
        )


class CellularCyclicAutocorrelator(Analysis, kw_only=True, frozen=True):
    subcarrier_spacings: types.CellularSubcarrierSpacingTuple = (15e3, 30e3, 60e3)
    frame_range: Union[int, tuple[int, int]] = (0, 1)
    frame_slots: types.CellularFrame = None
    symbol_range: Union[int, tuple[int, Union[int, None]]] = (0, None)
    generation: typing.Literal['4G', '5G'] = '5G'

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_range('frame_range', self.frame_range)
        _validate_range('symbol_range', self.symbol_range)


class CellularResourcePowerHistogram(
    Analysis,
    kw_only=True,
    frozen=True,
    dict=True,
):
    window: types.WindowType
    subcarrier_spacing: types.CellularSubcarrierSpacing
    power_low: types.PowerBinMin
    power_high: types.PowerBinMax
    power_resolution: types.PowerBinStep
    average_rbs: types.CellularAverageRBs = False
    average_slots: types.CellularAverageSlots = False
    guard_bandwidths: types.GuardBandwidths = (0, 0)
    frame_slots: types.CellularFrame = None
    special_symbols: types.CellularSpecialSymbols = None
    cyclic_prefix: types.CellularCyclicPrefix = 'normal'
    lo_bandstop: Union[types.LOBandstop, None] = None


class ChannelPowerTimeSeries(
    Analysis,
    kw_only=True,
    frozen=True,
):
    detector_period: types.DetectorPeriod
    power_detectors: types.PowerDetectors = ('rms', 'peak')


class ChannelPowerHistogram(
    ChannelPowerTimeSeries,
    kw_only=True,
    frozen=True,
):
    power_low: types.PowerBinMin
    power_high: types.PowerBinMax
    power_resolution: types.PowerBinStep


class CyclicChannelPower(Analysis, kw_only=True, frozen=True):
    cyclic_period: types.CyclicPeriod
    detector_period: types.DetectorPeriod
    power_detectors: types.PowerDetectors = ('rms', 'peak')
    cyclic_statistics: types.CyclicStatistics = ('min', 'mean', 'max')


class IQWaveform(
    Analysis,
    kw_only=True,
    frozen=True,
):
    start_time_sec: typing.Optional[float] = None
    stop_time_sec: typing.Optional[float] = None


class PowerSpectralDensity(FrequencyAnalysisSpecBase, kw_only=True, frozen=True):
    time_statistic: tuple[Union[str, float], ...] = ('mean',)


class SpectrogramHistogram(
    Spectrogram,
    kw_only=True,
    frozen=True,
):
    power_low: types.PowerBinMin
    power_high: types.PowerBinMax
    power_resolution: types.PowerBinStep


class SpectrogramHistogramRatio(
    SpectrogramHistogram,
    kw_only=True,
    frozen=True,
):
    pass


class BuiltinAnalyses(AnalysisGroup, kw_only=True, frozen=True, omit_defaults=True):
    """the ``analysis:`` block of a sweep: one optional field per registered measurement.

    A field set to a spec enables the measurement with those parameters, and
    ``None`` leaves the measurement out. `to_dict` omits the ``None`` fields.
    """

    channel_power_time_series: Optional[ChannelPowerTimeSeries] = None
    channel_power_histogram: Optional[ChannelPowerHistogram] = None
    cyclic_channel_power: Optional[CyclicChannelPower] = None
    spectrogram: Optional[Spectrogram] = None
    power_spectral_density: Optional[PowerSpectralDensity] = None
    spectrogram_histogram: Optional[SpectrogramHistogram] = None
    spectrogram_ratio_histogram: Optional[SpectrogramHistogramRatio] = None
    cellular_5g_pss_sync: Optional[Cellular5GNPSSSync] = None
    cellular_5g_pss_correlation: Optional[Cellular5GNRPSSCorrelator] = None
    cellular_5g_sss_sync: Optional[Cellular5GNSSSSync] = None
    cellular_5g_sss_correlation: Optional[Cellular5GNRSSSCorrelator] = None
    cellular_5g_ssb_spectrogram: Optional[Cellular5GNRSSBSpectrogram] = None
    cellular_cyclic_autocorrelation: Optional[CellularCyclicAutocorrelator] = None
    cellular_resource_power_histogram: Optional[CellularResourcePowerHistogram] = None
    iq_waveform: Optional[IQWaveform] = None


class BuiltinTriggers(AnalysisGroup, kw_only=True, frozen=True, omit_defaults=True):
    """the struct form of the ``signal_trigger:`` field of a source.

    One optional field per trigger, and exactly one is set. The set field names the
    trigger, and the value is the spec of the trigger measurement. With this form,
    the ``analysis:`` block need not enable the trigger measurement. The string form
    of ``signal_trigger`` names a registered trigger and takes the spec from the
    ``analysis:`` block.
    """

    cellular_5g_pss_sync: Optional[Cellular5GNPSSSync] = None
    cellular_5g_sss_sync: Optional[Cellular5GNSSSSync] = None
