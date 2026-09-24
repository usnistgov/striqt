from __future__ import annotations as __

from fractions import Fraction
from typing import Any, Callable, Literal, TYPE_CHECKING

from .. import specs

from ..lib import dataarrays, register
from ..lib.register import registry
from ..lib.util import np

import striqt.waveform as sw

if TYPE_CHECKING:
    from ..specs.structs import _Cellular5GNRSSBSync, _Cellular5GNRSSBCorrelator
    from ..lib.typing import Array, P, R, WrappedAnalysis
    from typing import Sequence


def hint_keywords(
    func: Callable[P, Any],
) -> Callable[[WrappedAnalysis[..., R]], WrappedAnalysis[P, R]]:
    """fill in type hints for the analysis parameters"""
    return lambda f: f  # pyright: ignore


@specs.helpers.lru_cache_on_converted(specs.AnalysisCapture)
def sync_params(
    capture: specs.AnalysisCapture,
    spec: _Cellular5GNRSSBCorrelator,
    kind: Literal['pss', 'sss'],
) -> sw.ofdm.SyncParams:
    """the 3GPP synchronization layout implied by `spec`, from `sw.ofdm.pss_params`
    or `sw.ofdm.sss_params` according to `kind`"""
    if kind == 'pss':
        build = sw.ofdm.pss_params
    else:
        build = sw.ofdm.sss_params

    return build(
        sample_rate=spec.sample_rate,
        subcarrier_spacing=spec.subcarrier_spacing,
        discovery_periodicity=spec.discovery_periodicity,
        shared_spectrum=spec.shared_spectrum,
        max_lag_symbols=spec.max_lag_symbols,
        symbol_indexes=spec.symbol_indexes,
        center_frequency=capture.center_frequency,
    )


def ssb_block_count(
    duration: float,
    spec: specs.Cellular5GNRPSSCorrelator
    | specs.Cellular5GNRSSSCorrelator
    | specs.Cellular5GNRSSBSpectrogram,
) -> int:
    """the number of synchronization blocks in `duration`, limited to
    `spec.max_block_count` and never fewer than 1"""
    total_blocks = round(duration / spec.discovery_periodicity)
    if spec.max_block_count is None:
        count = total_blocks
    else:
        count = min(spec.max_block_count, total_blocks)

    return max(count, 1)


@registry.coordinates(
    dtype='uint16', attrs={'standard_name': r'Cell Sector ID ($N_{ID}^\text{(2)}$)'}
)
@specs.helpers.lru_cache_on_converted(specs.Capture)
def cellular_cell_id2(capture: specs.Capture, spec: Any):
    values = np.array([0, 1, 2], dtype='uint16')
    return values


@registry.coordinates(dtype='uint16', attrs={'standard_name': 'SSB beam index'})
@specs.helpers.lru_cache_on_converted(specs.AnalysisCapture)
def cellular_ssb_beam_index(capture: specs.AnalysisCapture, spec: _Cellular5GNRSSBSync):
    # pss_params and sss_params return the same number of symbol indexes
    params = sync_params(capture, spec, 'sss')

    return list(range(len(params.symbol_indexes)))


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Time Elapsed', 'units': 's'}
)
@specs.helpers.lru_cache_on_converted(specs.Capture)
def cellular_ssb_start_time(
    capture: specs.Capture,
    spec: specs.Cellular5GNRPSSCorrelator | specs.Cellular5GNRSSSCorrelator,
):
    # the bare Capture projection carries no center_frequency, so the cell search
    # case is resolved here as for an unknown band
    params = sync_params(capture, spec, 'pss')
    count = ssb_block_count(params.duration, spec)

    return np.arange(count).astype('float32') * spec.discovery_periodicity


@registry.coordinates(dtype='float32', attrs={'standard_name': 'Lag', 'units': 's'})
@specs.helpers.lru_cache_on_converted(specs.AnalysisCapture)
def cellular_ssb_lag(capture: specs.AnalysisCapture, spec: _Cellular5GNRSSBCorrelator):
    # pss_params and sss_params agree on lag_count
    params = sync_params(capture, spec, 'pss')
    offs = round(spec.sample_rate * spec.delay)
    return np.arange(offs, offs + params.lag_count) / spec.sample_rate


def capture_sample_count(capture: specs.Capture) -> int:
    return round(capture.duration * capture.sample_rate)


def check_statistics(field: str, statistics: Sequence[str | float]) -> None:
    """raise ValueError unless each entry of the spec field names a statistic that
    `stat_ufunc_from_shorthand` implements"""
    from striqt.waveform.lib.power_analysis import stat_ufunc_from_shorthand

    for statistic in statistics:
        try:
            stat_ufunc_from_shorthand(statistic)
        except ValueError as ex:
            raise ValueError(
                f'{field} entry {statistic!r} is not a supported statistic: {ex} '
                f'({field}: {tuple(statistics)})'
            ) from ex


def check_frequency_band(
    field: str,
    nfft: int,
    sample_rate: float,
    bandwidth: float,
    *,
    offset: float = 0.0,
    offset_field: str = 'frequency_offset',
) -> int:
    """check that the band `field` selects lands on an `nfft`-bin frequency grid,
    returning the number of bins it spans.

    `sw.fourier.slice_freqs` applies the same rules to the array; only the vocabulary
    changes here, because an odd `nfft` reads as an off-grid DC bin there.
    """
    try:
        band = sw.fourier.slice_freqs(nfft, sample_rate, bandwidth, offset=offset)
    except ValueError as ex:
        if offset == 0:
            about = 'centered at baseband DC'
            quantities = f'{field}: {bandwidth}'
        else:
            about = f'centered at {offset_field}'
            quantities = f'{field}: {bandwidth}, {offset_field}: {offset}'
        raise ValueError(
            f'{field} must select a band of whole bins {about} on the '
            f'{nfft}-bin frequency grid: {ex} '
            f'({quantities}, sample_rate: {sample_rate}, '
            f'frequency_resolution: {sample_rate / nfft})'
        ) from ex

    return band.stop - band.start


@specs.helpers.lru_cache_on_converted(
    specs.AnalysisCapture, specs.Cellular5GNRSSSCorrelator
)
def validated_5g_ssb_sync_params(
    capture: specs.AnalysisCapture, spec: _Cellular5GNRSSBCorrelator
) -> sw.ofdm.SyncParams:
    """check the 3GPP sync layout implied by a (capture, SSB correlator spec) pair,
    returning the sync parameters it resolves to.

    `sss_params` is the stricter of the two parameter builders and reaches every check
    in `pss_params`, so one validator covers the PSS and SSS measurements alike. It
    also warms the `get_3gpp_phy` design, which the correlators would otherwise pay for
    after the acquisition.

    The resampling arithmetic mirrors `get_5g_ssb_iq` on the whole capture, the way
    `get_5g_ssb_iq` here calls it.
    """
    params = sync_params(capture, spec, 'sss')

    size_in = capture_sample_count(capture)
    frequency_step = capture.sample_rate / size_in
    if not sw.isroundmod(spec.frequency_offset, frequency_step):
        raise ValueError(
            'frequency_offset must be a counting-number multiple of '
            'sample_rate/duration, the frequency step of the resampler '
            f'(frequency_offset: {spec.frequency_offset}, '
            f'sample_rate: {capture.sample_rate}, duration: {capture.duration}, '
            f'frequency step: {frequency_step})'
        )

    size_out = round(size_in * spec.sample_rate / capture.sample_rate)
    shift = round(size_in * spec.frequency_offset / capture.sample_rate)
    try:
        sw.fourier.resample_edges(size_in, size_out, shift)
    except ValueError as ex:
        raise ValueError(
            f'the capture cannot be resampled to the synchronization block: {ex} '
            f'(duration: {capture.duration}, capture sample_rate: '
            f'{capture.sample_rate}, sample_rate: {spec.sample_rate}, '
            f'frequency_offset: {spec.frequency_offset}, '
            f'capture samples: {size_in}, block samples: {size_out})'
        ) from ex

    try:
        sw.ofdm.sync_frame_count(size_out, params)
    except ValueError as ex:
        raise ValueError(
            f'duration must hold whole 10 ms frames for the correlator: {ex} '
            f'(duration: {capture.duration}, sample_rate: {spec.sample_rate}, '
            f'subcarrier_spacing: {spec.subcarrier_spacing}, '
            f'symbol_indexes: {spec.symbol_indexes!r}, '
            f'max_lag_symbols: {spec.max_lag_symbols})'
        ) from ex

    return params


ssb_iq_cache = register.KwArgCache([dataarrays.CAPTURE_DIM, 'spec'])


@ssb_iq_cache.apply
def get_5g_ssb_iq(
    iq: Array,
    capture: specs.Capture,
    spec: _Cellular5GNRSSBCorrelator,
    oaresample=False,
) -> Array:
    """return a sync block waveform, which returns IQ that is recentered
    at baseband frequency spec.frequency_offset and downsampled to spec.sample_rate."""

    return sw.ofdm.get_5g_ssb_iq(
        iq,
        discovery_periodicity=spec.discovery_periodicity,
        fs_out=spec.sample_rate,
        fs_in=capture.sample_rate,
        subcarrier_spacing=spec.subcarrier_spacing,
        frequency_offset=spec.frequency_offset,
        delay=spec.delay,
        # max_block_count=spec.max_block_count,
        oaresample=oaresample,
    )


def cellular_stft_window_fractions(
    cyclic_prefix: Literal['normal', 'extended'],
) -> tuple[Fraction, Fraction]:
    """the (fractional_overlap, window_fill) that hop one STFT window per OFDM symbol.

    They hold only at a frequency_resolution of half the subcarrier spacing, where one
    FFT window spans 2 subcarrier periods and a slot spans 15 of them (3GPP TS 38.211
    Section 5.3.1), so a `window_fill` of 15/28 is the mean share of the window taken by
    one of the 14 normal-cyclic-prefix symbols in a slot.
    """
    if cyclic_prefix == 'normal':
        return Fraction(13, 28), Fraction(15, 28)
    else:
        # cyclic_prefix is a Literal, so 'extended' is the only other value
        return Fraction(11, 24), Fraction(13, 24)


# %% tolerance


def quantization_dB(dtype, limit_digits: int | None = None) -> float:
    """worst-case error from rounding a dB level to `limit_digits` decimals and storing
    it as `dtype`, for levels within 200 dB of 0 dB"""
    # a tolerance function never sees output values, so the storage rounding is bounded
    # over the level range rather than at the level actually produced
    level_range_dB = 200.0
    step = float(np.spacing(np.asarray(level_range_dB, dtype=dtype)))
    decimal = 0.0 if limit_digits is None else 10.0 ** (-limit_digits)
    return (decimal + step) / 2


def level_tolerance(
    *,
    amplitude_rms: float,
    size: int,
    power_rtol: float,
    log_tol: dict[str, float],
    quantization: float = 0.0,
    power_rms: float = 0.0,
    dtype='complex64',
) -> specs.Tolerance:
    """the `specs.Tolerance` of a dB power output, from its error terms.

    `amplitude_rms` is the relative rms error of the amplitude the power is taken from
    (input roundoff and any FFT passes, added in quadrature by the caller);
    `power_rtol` the worst-case relative error of squaring it and of any reduction
    summed in order (`sw.power_analysis.bin_power_rtol`, `sw.arrays.accum_rtol`); `power_rms` the rms
    relative error of a reduction over a contiguous axis (`sw.power_analysis.bin_power_rms`), whose
    worst element over `size` outputs is `peak_factor` times it; `log_tol` the dB
    conversion budget
    (`sw.power_analysis.log_conversion_tol`); `quantization` the storage rounding (`quantization_dB`).

    The peak amplitude error over `size` output elements adds the structured roundoff
    that lands on the matched-filter bin of the matched input: a tone's FFT bin, or an
    impulse's sample after the resampler. Both arrive the same way, so the term applies
    whenever there is any amplitude error at all.
    `off_peak_dBc` holds the depths at which the rms and the peak amplitude errors
    equal an element's own amplitude; `util.elementwise_atol` diverges at the latter.
    """
    peak_amplitude = sw.fourier.peak_factor(size) * amplitude_rms
    if amplitude_rms > 0:
        peak_amplitude += sw.fourier.on_peak_roundoff(dtype)
    additive = (
        sw.power_analysis.linear_tolerance_dB(power_rtol)
        + log_tol['atol']
        + quantization
    )
    if amplitude_rms > 0:
        off_peak_dBc = specs.ErrorBound(
            rms=float(sw.fourier.rms_tolerance_dBc(amplitude_rms)),
            peak=float(sw.fourier.rms_tolerance_dBc(peak_amplitude)),
        )
    else:
        off_peak_dBc = None
    return specs.Tolerance(
        units='dB',
        rtol=log_tol['rtol'],
        on_peak=specs.ErrorBound(
            rms=float(
                sw.power_analysis.level_tolerance_dB(amplitude_rms)
                + sw.power_analysis.linear_tolerance_dB(power_rms)
                + additive
            ),
            peak=float(
                sw.power_analysis.level_tolerance_dB(peak_amplitude)
                + sw.power_analysis.linear_tolerance_dB(
                    sw.fourier.peak_factor(size) * power_rms
                )
                + additive
            ),
        ),
        off_peak_dBc=off_peak_dBc,
    )
