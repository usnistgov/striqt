from __future__ import annotations as __

import math
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


def capture_noise_bandwidth(capture: specs.Capture) -> float:
    """return the noise bandwidth (in Hz) of the IQ that a measurement receives.

    Returns:
        `analysis_bandwidth` when finite, otherwise `sample_rate`
    """
    if math.isfinite(capture.analysis_bandwidth):
        return float(capture.analysis_bandwidth)
    else:
        return float(capture.sample_rate)


def bandwidth_attrs(
    noise_bandwidth: float,
    units: str,
    quantity: str,
    *,
    resolution_bandwidth: float | None = None,
) -> dict[str, Any]:
    """return the attrs of a power output with noise bandwidth `noise_bandwidth`.

    White noise with power spectral density N0 has a mean power of
    ``N0 * noise_bandwidth`` in the output.

    Args:
        noise_bandwidth: in Hz
        units: the unit of the output values, ``'dBm'`` or ``'√mW'``
        quantity: the name of the output, which begins `long_name`
        resolution_bandwidth: the equivalent noise bandwidth (in Hz) of the analysis
            window, which sets the output level of a CW tone; None omits the key

    Returns:
        a dict with ``'units'``, ``'noise_bandwidth'``, ``'long_name'`` as
        ``'<quantity> (<units>/<bandwidth>)'`` such as
        ``'Power spectral density (dBm/360 kHz)'``, and ``'resolution_bandwidth'``
        when not None
    """
    from ..lib.dataarrays import format_units

    attrs: dict[str, Any] = {
        'units': units,
        'noise_bandwidth': float(noise_bandwidth),
        'long_name': f'{quantity} ({units}/{format_units(noise_bandwidth, "Hz")})',
    }
    if resolution_bandwidth is not None:
        attrs['resolution_bandwidth'] = float(resolution_bandwidth)
    return attrs


def check_statistics(field: str, statistics: Sequence[str | float]) -> None:
    """check that each entry of `statistics` names a supported statistic.

    Args:
        field: name of the spec field that set `statistics`, named in the error
        statistics: ``'min'``, ``'max'``, ``'peak'``, ``'mean'``, ``'rms'``,
            ``'median'``, or a quantile in ``[0, 1]``
            (`sw.power_analysis.stat_ufunc_from_shorthand`)

    Raises:
        ValueError: if an entry of `statistics` is not a supported statistic
    """
    from striqt.waveform.lib.power import stat_ufunc_from_shorthand

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
    """return the number of bins in the band that `field` selects.

    `offset` must fall on the ``sample_rate / nfft`` grid. The band must lie within
    ``±sample_rate / 2``. A `bandwidth` between grid multiples is accepted and
    rounded to the nearest whole number of bins.

    Args:
        field: name of the spec field that set `bandwidth`, named in the error
        nfft: number of bins across `sample_rate`, which sets the grid spacing
        sample_rate: in S/s
        bandwidth: width of the band in Hz
        offset: center of the band in Hz relative to baseband DC
        offset_field: name of the spec field that set `offset`, named in the error

    Returns:
        the number of bins the band spans after rounding, at least 0

    Raises:
        ValueError: if `bandwidth` is negative, `offset` is off the grid, or the
            rounded band extends beyond ``±sample_rate / 2``
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

    The bound holds for levels within 200 dB of 0 dB.

    Args:
        dtype: the floating-point storage type of the level
        limit_digits: decimals kept by rounding; `None` selects no decimal rounding

    Returns:
        a positive bound in dB set by the rounding step of `limit_digits` and the
        spacing of `dtype` at 200 dB
    """
    # tolerance functions run before the output exists. The storage spacing is taken
    # at the top of a 200 dB level range instead of at the level produced.
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
    """combine the error terms of a dB power output into one `specs.Tolerance`.

    The peak amplitude error over `size` output elements includes the on-peak
    roundoff of `dtype` whenever `amplitude_rms` is nonzero. `off_peak_dBc` gives the
    depths (in dBc) at which the rms and the peak amplitude errors equal the amplitude
    of an element.

    Args:
        amplitude_rms: relative rms error of the amplitude underlying the power
            (input roundoff and FFT roundoff, combined in quadrature by the caller)
        size: number of output elements covered by the bound
        power_rtol: worst-case relative error of squaring the amplitude and of any
            reduction summed in order (`sw.power_analysis.bin_power_rtol`,
            `sw.arrays.accum_rtol`)
        log_tol: the dB conversion budget, keyed ``'rtol'`` and ``'atol'``
            (`sw.power_analysis.log_conversion_tol`)
        quantization: storage rounding in dB (`quantization_dB`)
        power_rms: rms relative error of a reduction over a contiguous axis
            (`sw.power_analysis.bin_power_rms`)
        dtype: complex dtype of the amplitude, which sets the on-peak roundoff term

    Returns:
        the `specs.Tolerance` in dB, with `off_peak_dBc` None when `amplitude_rms`
        is 0
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
