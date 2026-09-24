from __future__ import annotations as __

from typing import TYPE_CHECKING, Any, Callable

import striqt.waveform as sw

from .. import specs
from ..lib.register import registry  # noqa: F401
from ..lib.util import np

if TYPE_CHECKING:
    from typing import Sequence

    from ..lib.typing import P, R, WrappedAnalysis


def hint_keywords(
    func: Callable[P, Any],
) -> Callable[[WrappedAnalysis[..., R]], WrappedAnalysis[P, R]]:
    """fill in type hints for the analysis parameters"""
    return lambda f: f  # pyright: ignore


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
