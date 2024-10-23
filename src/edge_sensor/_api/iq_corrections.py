from __future__ import annotations
import functools
import typing
import pickle
import gzip
from math import ceil

from channel_analysis._api import filters
from . import util

from .radio import RadioDevice, get_capture_buffer_sizes, design_capture_filter
from . import structs


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


def save_calibration_corrections(path, corrections: 'xr.DatasetType'):
    with gzip.GzipFile(path, 'wb') as fd:
        pickle.dump(corrections, fd)


def _y_factor_temperature(
    power: 'xr.DataArray', enr_dB: float, Tamb: float, Tref=290.0
) -> xr.Dataset:
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


def _y_factor_power_corrections(
    dataset: 'xr.DatasetType', enr_dB: float, Tamb: float, Tref=290.0
) -> xr.Dataset:
    # TODO: check that this works for xr.DataArray inputs in (enr_dB, Tamb)

    kwargs = dict(list(locals().items())[1:])

    k = scipy.constants.Boltzmann * 1000  # scaled from W/K to mW/K
    B = dataset.analysis_bandwidth
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
    dataset: 'xr.DatasetType', enr_dB: float, Tamb: float, Tref=290.0
) -> xr.Dataset:
    kwargs = locals()
    ret = _y_factor_power_corrections(**kwargs)
    # ret['baseband_frequency_response'] = _y_factor_frequency_response_correction(
    #     **kwargs, fc_temperatures=ret.temperature
    # )
    return ret


def resampling_correction(
    iq: 'iqwaveform.util.Array',
    capture: structs.RadioCapture,
    radio: RadioDevice,
    force_calibration: typing.Optional[xr.Dataset] = None,
    *,
    axis=0,
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

    # create a buffer large enough for post-processing seeded with a copy of the IQ
    _, buf_size = get_capture_buffer_sizes(radio, capture)

    if nfft_out > nfft:
        buf_size = ceil(buf_size * nfft_out / nfft)
    buf = xp.empty(buf_size, dtype='complex64')
    buf[: iq.size] = xp.asarray(iq)
    iq = buf[: iq.size]

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
            analysis_bandwidth=capture.analysis_bandwidth,
            host_resample=capture.host_resample,
        )

        try:
            sel = corrections.power_correction.sel(**exact_matches, drop=True)
        except KeyError as ex:
            print(exact_matches)
            raise

        if 'duration' in sel.coords:
            sel = sel.drop('duration')

        if 'radio_id' in sel.coords:
            sel = sel.drop('radio_id')

        # allow interpolation between sample points in these fields
        sel = (
            sel.squeeze(drop=True)
            .dropna('center_frequency')
            .interp(center_frequency=capture.center_frequency)
        )

        if not np.isfinite(sel):
            raise ValueError('no calibration data available for this capture')

        power_scale = float(sel)

    if nfft == nfft_out:
        # no resample
        if power_scale is not None:
            iq *= np.sqrt(power_scale)

        if analysis_filter['passband'] != (None, None):
            iq = filters.iir_filter(
                iq,
                capture,
                passband_ripple=0.5,
                stopband_attenuation=80,
                transition_bandwidth=500e3,
                out=iq,
            )

        return iq

    w = iqwaveform.fourier._get_window(
        analysis_filter['window'], nfft, fftbins=False, xp=xp
    )

    freqs, _, xstft = iqwaveform.fourier.stft(
        iq,
        fs=fs_backend,
        window=w,
        nperseg=nfft,
        noverlap=round(nfft * overlap_scale),
        axis=axis,
        truncate=False,
        out=buf,
    )

    # set the passband roughly equal to the 3 dB bandwidth based on ENBW
    freq_res = fs_backend / nfft
    enbw = freq_res * iqwaveform.fourier.equivalent_noise_bandwidth(
        analysis_filter['window'], nfft, fftbins=False
    )
    passband = analysis_filter['passband']

    if nfft_out < nfft:
        # downsample already does the filter
        freqs, xstft = iqwaveform.fourier.downsample_stft(
            freqs,
            xstft,
            nfft_out=nfft_out,
            passband=passband,
            axis=axis,
            out=buf,
        )
    elif nfft_out > nfft:
        pad_left = (nfft_out - nfft) // 2
        pad_right = pad_left + (nfft_out - nfft) % 2

        if capture.analysis_bandwidth is not None:
            iqwaveform.fourier.zero_stft_by_freq(
                freqs,
                xstft,
                passband=(passband[0] + enbw / 2, passband[1] - enbw / 2),
                axis=axis,
            )

        xstft = iqwaveform.util.pad_along_axis(
            xstft, [[pad_left, pad_right]], axis=axis + 1
        )

    else:
        # filter
        iqwaveform.fourier.zero_stft_by_freq(
            freqs,
            xstft,
            passband=(passband[0] + enbw, passband[1] - enbw),
            axis=axis,
        )

    iq = iqwaveform.fourier.istft(
        xstft,
        size=round(capture.duration * capture.sample_rate),
        nfft=nfft_out,
        noverlap=noverlap,
        out=buf,
        axis=axis,
    )

    if power_scale is not None:
        iq *= np.sqrt(power_scale)

    return iq
