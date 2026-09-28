from __future__ import annotations as __

from typing import TYPE_CHECKING, Any, Callable

import striqt.waveform as sw

from .. import specs
from ..lib.register import registry  # noqa: F401
from ..lib.util import np

if TYPE_CHECKING:
    from typing import Sequence

    from striqt.waveform.lib.typing import DTypeLike

    from ..lib.typing import P, R, WrappedAnalysis


def hint_keywords(
    func: Callable[P, Any],
) -> Callable[[WrappedAnalysis[..., R]], WrappedAnalysis[P, R]]:
    """fill in type hints for the analysis parameters"""
    return lambda f: f


def capture_sample_count(capture: specs.Capture) -> int:
    return round(capture.duration * capture.sample_rate)


def check_statistics(field: str, statistics: Sequence[str | float]) -> None:
    """raise `ValueError` for any `statistics` entry that is not a supported statistic.

    Supported are 'min', 'max', 'peak', 'mean', 'rms', 'median' and quantiles in
    ``[0, 1]`` (`sw.power_analysis.stat_ufunc_from_shorthand`); `field` is the spec
    field named in the message.
    """
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
    """return the bin count of the band `field` selects, checking it lies on the grid.

    `sw.fourier.slice_freqs` applies the same rules to the array; only the vocabulary
    changes here, because an odd `nfft` reads as an off-grid DC bin there.

    Args:
        nfft: number of bins across `sample_rate`, which sets the grid spacing
        sample_rate: in S/s
        bandwidth: width of the band in Hz, rounded to a whole number of bins
        offset: center of the band relative to baseband DC in Hz
        offset_field: name of the spec field that set `offset`, for the message

    Returns:
        the number of bins the band spans

    Raises:
        ValueError: if `bandwidth` is negative, `offset` is off the grid, or the band
            extends beyond the sampled bandwidth
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


def quantization_dB(dtype: DTypeLike, limit_digits: int | None = None) -> float:
    """bound the error (in dB) of rounding and storing a dB level.

    The level is rounded to `limit_digits` decimals and stored as `dtype`; the bound
    holds for levels within 200 dB of 0 dB.

    Args:
        dtype: the floating-point storage type of the level
        limit_digits: decimals kept by rounding; `None` skips the decimal rounding

    Returns:
        a positive bound in dB set by the rounding step of `limit_digits` and the
        spacing of `dtype` at 200 dB; the storage term alone when `limit_digits` is
        `None`
    """
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
    dtype: DTypeLike = 'complex64',
) -> specs.Tolerance:
    """combine the error terms of a dB power output into its `specs.Tolerance`.

    The peak amplitude error over `size` output elements adds the structured roundoff
    that lands on the matched-filter bin of the matched input: a tone's FFT bin, or an
    impulse's sample after the resampler. Both arrive the same way, so the term applies
    whenever there is any amplitude error at all. `off_peak_dBc` holds the depths at
    which the rms and the peak amplitude errors equal an element's own amplitude;
    `util.elementwise_atol` diverges at the latter.

    Args:
        amplitude_rms: relative rms error of the amplitude the power is taken from
            (input roundoff and any FFT passes, added in quadrature by the caller)
        size: number of output elements the bound holds over
        power_rtol: worst-case relative error of squaring the amplitude and of any
            reduction summed in order (`sw.power_analysis.bin_power_rtol`,
            `sw.arrays.accum_rtol`)
        log_tol: the dB conversion budget, keyed ``'rtol'`` and ``'atol'``
            (`sw.power_analysis.log_conversion_tol`)
        quantization: storage rounding in dB (`quantization_dB`)
        power_rms: rms relative error of a reduction over a contiguous axis
            (`sw.power_analysis.bin_power_rms`); its worst case over `size` outputs
            enters the peak bound
        dtype: complex dtype of the amplitude, which sets the on-peak roundoff term

    Returns:
        the `specs.Tolerance` with ``units='dB'``; `off_peak_dBc` is `None` when
        `amplitude_rms` is 0
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
