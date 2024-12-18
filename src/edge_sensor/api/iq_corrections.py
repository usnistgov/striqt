from __future__ import annotations
import functools
import typing
import pickle
import gzip
from math import ceil

from channel_analysis.api import filters
from . import util

from .radio import RadioDevice, get_capture_buffer_sizes, design_capture_filter
from .radio.base import needs_stft
from . import structs
from scipy.signal._arraytools import axis_slice

if typing.TYPE_CHECKING:
    import numpy as np
    import xarray as xr
    import scipy
    import iqwaveform
else:
    np = util.lazy_import('numpy')
    xr = util.lazy_import('xarray')
    scipy = util.lazy_import('scipy')
    iqwaveform = util.lazy_import('iqwaveform')


@functools.lru_cache
def read_calibration_corrections(path):
    with gzip.GzipFile(path, 'rb') as fd:
        return pickle.load(fd)


def save_calibration_corrections(path, corrections: 'xr.Dataset'):
    with gzip.GzipFile(path, 'wb') as fd:
        pickle.dump(corrections, fd)


def _y_factor_temperature(
    power: 'xr.DataArray', enr_dB: float, Tamb: float, Tref=290.0
) -> 'xr.Dataset':
    Toff = Tamb
    Ton = Tref * 10 ** (enr_dB / 10.0)

    # compute the Y-factor from measured power
    Pon = power.sel(noise_diode_enabled=True, drop=True)
    Poff = power.sel(noise_diode_enabled=False, drop=True)
    Y = Pon / Poff

    # compute receive noise temperature from the Y-factor
    T = (Ton - Y * Toff) / (Y - 1)
    T.name = 'T'
    T.attrs = {'units': 'K'}

    return T


def _limit_nyquist_bandwidth(data: 'xr.DataArray') -> 'xr.DataArray':
    """replace float('inf') analysis bandwidth with the Nyquist bandwidth"""

    # return bandwidth with same shape as dataset.channel_power_time_series
    bw = data.analysis_bandwidth.broadcast_like(data).copy().squeeze()
    sample_rate = data.sample_rate.broadcast_like(data).squeeze()
    where = bw.values == float('inf')
    bw.values[where] = sample_rate.values[where]
    return bw


def _y_factor_power_corrections(
    dataset: 'xr.Dataset', enr_dB: float, Tamb: float, Tref=290.0
) -> 'xr.Dataset':
    # TODO: check that this works for xr.DataArray inputs in (enr_dB, Tamb)

    kwargs = dict(list(locals().items())[1:])

    k = scipy.constants.Boltzmann * 1000  # scaled from W/K to mW/K
    enr = 10 ** (enr_dB / 10.0)

    power = (
        dataset.channel_power_time_series.sel(power_detector='rms', drop=True)
        .pipe(lambda x: 10 ** (x / 10.0))
        .mean(dim='time_elapsed')
    )
    power.name = 'RMS power'

    Pon = power.sel(noise_diode_enabled=True, drop=True)
    Poff = power.sel(noise_diode_enabled=False, drop=True)
    Y = Pon / Poff

    noise_figure = enr_dB - 10 * np.log10(Y - 1)
    noise_figure.name = 'Noise figure'
    noise_figure.attrs = {'units': 'dB'}

    T = Tref * (10 ** (noise_figure / 10) - 1)  # _y_factor_temperature(power, **kwargs)
    T.name = 'Noise temperature'
    T.attrs = {'units': 'K'}

    B = _limit_nyquist_bandwidth(T)

    power_correction = (k * (T + enr * Tref) * B) / Pon
    power_correction.name = 'Input power scaling correction'
    power_correction.attrs = {'units': 'mW/fs'}

    return xr.Dataset(
        {
            'temperature': T,
            'noise_figure': noise_figure,
            'power_correction': power_correction,
        }
    )


def _y_factor_frequency_response_correction(
    dataset: 'xr.DataArray',
    fc_temperatures: 'xr.DataArray',
    enr_dB: float,
    Tamb: float,
    Tref=290,
):
    spectrum = dataset.persistence_spectrum.sel(
        persistence_statistic='mean', drop=True
    ).pipe(lambda x: 10 ** (x / 10.0))

    fc_T = fc_temperatures
    all_T = _y_factor_temperature(spectrum, enr_dB=20.87, Tamb=294.5389)

    # normalize the power correction at each center frequency, and then average the result across center frequency
    baseband_frequency_response = (fc_T.broadcast_like(all_T) / all_T).median(
        dim='center_frequency'
    )

    baseband_frequency_response.name = 'Baseband power scaling correction'
    baseband_frequency_response.attrs = {'units': 'unitless'}

    return baseband_frequency_response


def compute_y_factor_corrections(
    dataset: 'xr.Dataset', enr_dB: float, Tamb: float, Tref=290.0
) -> 'xr.Dataset':
    kwargs = locals()
    ret = _y_factor_power_corrections(**kwargs)
    # ret['baseband_frequency_response'] = _y_factor_frequency_response_correction(
    #     **kwargs, fc_temperatures=ret.temperature
    # )
    return ret


def _describe_missing_data(corrections: 'xr.Dataset', exact_matches: dict):
    misses = []
    cal = corrections.power_correction.copy()

    invalid_matches = dict(exact_matches)
    # remove the valid matches
    for field, value in exact_matches.items():
        try:
            cal = cal.sel({field: value}, drop=True)
        except KeyError:
            pass
        else:
            del invalid_matches[field]

    # now note the remainder
    for field, value in invalid_matches.items():
        try:
            cal.sel({field: value})
        except KeyError:
            misses += [
                f'{repr(value)} in {repr(field)} (available: '
                + ', '.join([str(v) for v in cal[field].values])
                + ')'
            ]
    return '; '.join(misses)


def resampling_correction(
    iq: 'iqwaveform.util.Array',
    capture: structs.RadioCapture,
    radio: RadioDevice,
    force_calibration: typing.Optional['xr.Dataset'] = None,
    *,
    axis=0,
    out=None,
):
    """apply a bandpass filter implemented through STFT overlap-and-add.

    Args:
        iq: the input waveform
        capture: the capture filter specification structure
        radio: the radio instance that performed the capture
        force_calibration: if specified, this calibration dataset is used rather than loading from file
        axis: the axis of `x` along which to compute the filter

    Returns:
        the filtered IQ capture
    """

    fs_backend, _, analysis_filter = design_capture_filter(
        radio.base_clock_rate, capture
    )
    nfft = analysis_filter['nfft']
    nfft_out, noverlap, overlap_scale, _ = iqwaveform.fourier._ola_filter_parameters(
        iq.size,
        window=analysis_filter['window'],
        nfft_out=analysis_filter.get('nfft_out', nfft),
        nfft=nfft,
        extend=True,
    )

    xp = util.import_cupy_with_fallback()

    # _, buf_size = get_capture_buffer_sizes(radio, capture)
    # if out is None:
    #     # create a buffer large enough for post-processing seeded with a copy of the IQ
    #     if nfft_out > nfft:
    #         buf_size = ceil(buf_size * nfft_out / nfft)
    #     buf_size = max(buf_size, iq.shape[axis])
    #     buf = xp.empty(buf_size, dtype=iq.dtype)
    # else:
    #     if out.size < buf.size:
    #         raise ValueError('resampling output buffer is too small')
    #     buf = out

    iq = xp.asarray(iq)

    if force_calibration is not None:
        corrections = force_calibration
    elif radio.calibration:
        corrections = read_calibration_corrections(radio.calibration)
    else:
        corrections = None

    if corrections is None:
        power_scale = None
    else:
        # these fields must match the calibration conditions exactly
        exact_matches = dict(
            channel=capture.channel,
            gain=capture.gain,
            lo_shift=capture.lo_shift,
            sample_rate=capture.sample_rate,
            analysis_bandwidth=capture.analysis_bandwidth or np.inf,
            host_resample=capture.host_resample,
        )

        try:
            sel = corrections.power_correction.sel(**exact_matches, drop=True)
        except KeyError:
            misses = _describe_missing_data(corrections, exact_matches)
            exc = KeyError(f'calibration is not available for this capture: {misses}')
        else:
            exc = None

        if exc is not None:
            raise exc

        for name in ('duration', 'radio_id', 'delay'):
            if name in sel.coords:
                sel = sel.drop(name)

        # allow interpolation between sample points in these fields
        sel = (
            sel.squeeze(drop=True)
            .dropna('center_frequency')
            .interp(center_frequency=capture.center_frequency)
        )

        if not np.isfinite(sel):
            raise ValueError('no calibration data available for this capture')

        power_scale = float(sel)

    if not needs_stft(analysis_filter, capture):
        # no filtering or resampling needed
        iq = iq[: round(capture.duration * capture.sample_rate)]
        if power_scale is not None:
            iq *= np.sqrt(power_scale)
        return iq

    w = iqwaveform.fourier._get_window(
        analysis_filter['window'], nfft, fftbins=False, xp=xp
    )

    # set the passband roughly equal to the 3 dB bandwidth based on ENBW
    freq_res = fs_backend / nfft
    enbw = freq_res * iqwaveform.fourier.equivalent_noise_bandwidth(
        analysis_filter['window'], nfft, fftbins=False
    )
    passband = analysis_filter['passband']

    freqs, _, xstft = iqwaveform.fourier.stft(
        iq,
        fs=fs_backend,
        window=w,
        nperseg=nfft,
        noverlap=round(nfft * overlap_scale),
        axis=axis,
        truncate=False,
        # out=buf,
    )

    if nfft_out < nfft:
        # downsample applies the filter as well
        freqs, xstft = iqwaveform.fourier.downsample_stft(
            freqs,
            xstft,
            nfft_out=nfft_out,
            passband=passband,
            axis=axis,
            # out=buf,
        )
    elif nfft_out > nfft:
        # upsample
        if np.isfinite(capture.analysis_bandwidth):
            iqwaveform.fourier.zero_stft_by_freq(
                freqs,
                xstft,
                passband=(passband[0] + enbw / 2, passband[1] - enbw / 2),
                axis=axis,
            )

        # padded_shape = list(xstft.shape)
        # padded_shape[axis+1] += pad_left+pad_right
        # padded_xstft = iqwaveform.fourier._truncated_buffer(buf, padded_shape)

        # start with the actual data, to make sure we don't overwrite it in the underlying buffer
        # axis_slice(padded_xstft, pad_left, padded_xstft.shape[axis+1] - pad_right, axis=axis+1)[:] = xstft
        # axis_slice(padded_xstft, 0, pad_left, axis=axis+1)[:] = 0
        # axis_slice(padded_xstft, padded_xstft.shape[axis+1] - pad_right, None, axis=axis+1)[:] = 0

        # xstft = padded_xstft

        pad_left = (nfft_out - nfft) // 2
        pad_right = pad_left + (nfft_out - nfft) % 2

        xstft = iqwaveform.util.pad_along_axis(
            xstft, [[pad_left, pad_right]], axis=axis + 1
        )

    else:
        # nfft_out == nfft
        iqwaveform.fourier.zero_stft_by_freq(
            freqs,
            xstft,
            passband=(passband[0] + enbw, passband[1] - enbw),
            axis=axis,
        )

    iq = iqwaveform.fourier.istft(
        xstft,
        nfft=nfft_out,
        noverlap=noverlap,
        # out=buf,
        axis=axis,
    )

    # start the capture after the transient holdoff window
    iq_size_out = round(capture.duration * capture.sample_rate)
    i0 = nfft_out // 2
    assert i0 + iq_size_out <= iq.shape[axis]
    iq = iq[i0 : i0 + iq_size_out]

    if power_scale is None and nfft == nfft_out:
        pass
    else:
        # voltage_scale = (power_scale or 1) * nfft_out / nfft
        iq *= np.sqrt(power_scale or 1) * np.sqrt(nfft_out / nfft)

    return iq
