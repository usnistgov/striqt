"""data structures that specify waveform characteristics"""

from __future__ import annotations as __

import fractions
import functools
from math import inf
import typing

import msgspec

from . import helpers, types

_T = typing.TypeVar('_T')
_TS = typing.TypeVar('_TS', bound='SpecBase')


class SpecBase(
    msgspec.Struct,
    kw_only=True,
    frozen=True,
    forbid_unknown_fields=True,
    cache_hash=True,
):
    """Base type for structures that support validated
    (de)serialization.

    It is a `msgspec.Struct` class with some often-used utility
    methods that fix encoding and decoding hooks for extra types.
    """

    def replace(self, **attrs) -> typing.Self:
        """returns a copy of self with changed attributes.

        See also:
            Python standard library `copy.replace`
        """
        if len(attrs) == 0:
            return self
        return msgspec.structs.replace(self, **attrs).validate()

    def to_dict(self, skip_private=False, unfreeze=False) -> dict:
        """return a dictinary representation of `self`"""
        map = helpers.to_builtins(self)

        if skip_private:
            for name in helpers._private_fields(type(self)):
                del map[name]

        if unfreeze:
            return helpers._unfreeze(map)
        else:
            return map

    @classmethod
    def from_dict(cls: type[_T], d: dict) -> _T:
        return helpers.convert_dict(d, type=cls)

    @classmethod
    def from_spec(cls: type[_T], other: SpecBase) -> _T:
        return helpers.convert_spec(other, type=cls)

    def validate(self) -> typing.Self:
        return _validate(self)

    def __post_init__(self):
        for name in helpers.freezable_fields(type(self)):
            v = getattr(self, name)
            if isinstance(v, (tuple, dict, list)):
                msgspec.structs.force_setattr(self, name, helpers._deep_freeze(v))


@functools.lru_cache(1024)
def _validate(spec: _TS) -> _TS:
    return spec.from_dict(spec.to_dict())


class Capture(SpecBase, kw_only=True, frozen=True):
    """bare minimum information about an IQ acquisition"""

    # acquisition
    duration: types.DurationType = 0.1
    sample_rate: types.SampleRateType = 15.36e6
    analysis_bandwidth: types.AnalysisBandwidthType = inf

    def __post_init__(self):
        from ..lib import util

        super().__post_init__()

        if not util.isroundmod(self.duration * self.sample_rate, 1):
            raise ValueError(
                f'duration {self.duration!r} is not an integer multiple of sample period'
            )


class AnalysisFilter(SpecBase, kw_only=True, frozen=True):
    nfft: int = 8192
    window: typing.Union[tuple[str, ...], str] = 'hamming'
    nfft_out: int | None = None


class FilteredCapture(Capture, kw_only=True, frozen=True):
    # filtering and resampling
    analysis_filter: AnalysisFilter = msgspec.field(default_factory=AnalysisFilter)
    # analysis_filter: dict = msgspec.field(
    #     default_factory=lambda: {'nfft': 8192, 'window': 'hamming'}
    # )


class AnalysisKeywords(typing.TypedDict):
    as_xarray: typing.NotRequired[bool | typing.Literal['delayed']]


class Analysis(SpecBase, kw_only=True, frozen=True):
    """
    Returns:
        Analysis result of type `(xarray.DataArray if as_xarray else type(iq))`

    Args:
        iq (numpy.ndarray or cupy.ndarray): the M-channel input waveform of shape (M,N)
        capture:
        as_xarray (bool): True to return xarray.DataArray or False to match type(iq)
    """


class AnalysisGroup(SpecBase, kw_only=True, frozen=True):
    """base class for a defining set of Analysis specs"""

    pass


# %% Spectral analysis
class FrequencyAnalysisSpecBase(
    Analysis,
    kw_only=True,
    frozen=True,
    dict=True,
):
    """
    window (specs.types.WindowType): a window specification, following `scipy.signal.get_window`
    frequency_resolution (float): the STFT resolution (in Hz)
    fractional_overlap (float):
        fraction of each FFT window that overlaps with its neighbor
    window_fill (float):
        fraction of each FFT window that is filled with the window function
        (leaving the rest zeroed)
    integration_bandwidth (float): bin bandwidth for RMS averaging in the frequency domain
    trim_stopband (bool):
        whether to trim the frequency axis to capture.analysis_bandwidth
    """

    window: types.WindowType
    frequency_resolution: float
    fractional_overlap: fractions.Fraction = fractions.Fraction(0)
    window_fill: fractions.Fraction = fractions.Fraction(1)
    integration_bandwidth: typing.Optional[float] = None
    trim_stopband: bool = True
    lo_bandstop: typing.Optional[float] = None


class Cellular5GNRSSBSpectrogram(Analysis, kw_only=True, frozen=True):
    """
    subcarrier_spacing (float): 3GPP channel subcarrier spacing (Hz)
    sample_rate (float): output sample rate for the resampled synchronization waveform (samples/s)
    discovery_periodicity (float): time period between synchronization blocks (s)
    frequency_offset (float or dict[float, float]):
        center frequency offset (see notes)
    shared_spectrum:
        whether to follow the 3GPP "shared spectrum" synchronizatio block layout
    max_block_count: number of synchronization blocks to evaluate
    trim_cp: whether to trim the cyclic prefix duration from the output
    """

    subcarrier_spacing: float

    # ssb parameters
    sample_rate: float = 15.36e6 / 2
    discovery_periodicity: float = 20e-3
    frequency_offset: float = 0
    max_block_count: typing.Optional[int] = None

    # spectrogram info
    window: types.WindowType = ('kaiser_by_enbw', 2)
    lo_bandstop: typing.Optional[float] = None

    # hard-coded for re-use by PSS/SSS functions
    shared_spectrum = False
    trim_cp = False


# %% Cellular 5G NR synchronizatino
class _Cellular5GNRSSBCorrelator(Analysis, kw_only=True, frozen=True):
    """
    subcarrier_spacing (float): 3GPP channel subcarrier spacing (Hz)
    sample_rate (float): output sample rate for the resampled synchronization waveform (samples/s)
    discovery_periodicity (float): time period between synchronization blocks (s)
    frequency_offset (float or dict[float, float]):
        center frequency offset (see notes)
    shared_spectrum:
        whether to follow the 3GPP "shared spectrum" synchronizatio block layout
    max_block_count: number of synchronization blocks to evaluate
    trim_cp: whether to trim the cyclic prefix duration from the output
    """

    subcarrier_spacing: float
    sample_rate: float = 15.36e6 / 2
    discovery_periodicity: float = 20e-3
    frequency_offset: float = 0
    shared_spectrum: bool = False
    max_block_count: typing.Optional[int] = 1
    trim_cp: bool = True


class Cellular5GNRPSSCorrelator(_Cellular5GNRSSBCorrelator, kw_only=True, frozen=True):
    # same as SSS, but the registry requires unique spec types
    pass


class Cellular5GNRSSSCorrelator(_Cellular5GNRSSBCorrelator, kw_only=True, frozen=True):
    pass


class _Cellular5GNRSSBSync(_Cellular5GNRSSBCorrelator, frozen=True, kw_only=True):
    window_fill: float = 0.5
    snr_window_fill: float = 0.08
    per_port: bool = False


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
    dB (bool): if True, returned power is transformed into dB units
    """

    time_aperture: typing.Optional[float] = None
    dB = True


class CellularCyclicAutocorrelator(
    Analysis,
    kw_only=True,
    frozen=True,
    dict=True,
):
    subcarrier_spacings: typing.Union[float, tuple[float, ...]] = (15e3, 30e3, 60e3)
    frame_range: typing.Union[int, tuple[int, int]] = (0, 1)
    frame_slots: typing.Union[str, None] = None
    symbol_range: typing.Union[int, tuple[int, typing.Optional[int]]] = (0, None)
    generation: typing.Literal['4G', '5G'] = '5G'

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.frame_range, tuple) and self.frame_range[0] > 0:
            assert self.frame_range[1] is not None
            assert self.frame_range[1] >= self.frame_range[0]
        if isinstance(self.symbol_range, tuple) and self.symbol_range[0] > 0:
            assert self.symbol_range[1] is not None
            assert self.symbol_range[1] >= self.symbol_range[0]


class CellularResourcePowerHistogram(
    Analysis,
    kw_only=True,
    frozen=True,
    dict=True,
):
    window: types.WindowType
    subcarrier_spacing: float
    power_low: float
    power_high: float
    power_resolution: float
    average_rbs: typing.Union[bool, typing.Literal['half']] = False
    average_slots: bool = False
    guard_bandwidths: tuple[float, float] = (0, 0)
    frame_slots: typing.Union[str, None] = None
    special_symbols: typing.Union[str, None] = None

    cyclic_prefix: typing.Union[
        typing.Literal['normal'], typing.Literal['extended']
    ] = 'normal'

    lo_bandstop: typing.Optional[float] = None


class ChannelPowerTimeSeries(
    Analysis,
    kw_only=True,
    frozen=True,
):
    detector_period: fractions.Fraction
    power_detectors: tuple[str, ...] = ('rms', 'peak')


class ChannelPowerHistogram(
    ChannelPowerTimeSeries,
    kw_only=True,
    frozen=True,
):
    power_low: float
    power_high: float
    power_resolution: float


class CyclicChannelPower(Analysis, kw_only=True, frozen=True):
    cyclic_period: float
    detector_period: fractions.Fraction
    power_detectors: tuple[str, ...] = ('rms', 'peak')
    cyclic_statistics: tuple[typing.Union[str, float], ...] = ('min', 'mean', 'max')


class IQWaveform(
    Analysis,
    kw_only=True,
    frozen=True,
):
    start_time_sec: typing.Optional[float] = None
    stop_time_sec: typing.Optional[float] = None


class PowerSpectralDensity(FrequencyAnalysisSpecBase, kw_only=True, frozen=True):
    time_statistic: tuple[typing.Union[str, float], ...] = ('mean',)


class SpectrogramHistogram(
    Spectrogram,
    kw_only=True,
    frozen=True,
):
    power_low: float
    power_high: float
    power_resolution: float


class SpectrogramHistogramRatio(
    SpectrogramHistogram,
    kw_only=True,
    frozen=True,
):
    pass
