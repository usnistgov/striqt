from __future__ import annotations
from iqwaveform import fourier
from functools import lru_cache
import xarray as xr
import zarr
import numpy as np
from scipy.constants import Boltzmann
from channel_analysis import waveform
from typing import Optional
import pickle
import gzip

from .radio import util
from . import structs
from .util import import_cupy_with_fallback_warning


@lru_cache
def read_calibration_corrections(path):
    with gzip.GzipFile(path, 'rb') as fd:
        return pickle.load(fd)


def save_calibration_corrections(path, corrections: xr.Dataset):
    with gzip.GzipFile(path, 'wb') as fd:
        pickle.dump(corrections, fd)


def _y_factor_temperature(
    power: xr.DataArray, enr_dB: float, Tamb: float, Tref=290.0
) -> xr.Dataset:
    Toff = Tamb
    Ton = Tref * 10 ** (enr_dB / 10.0)


    Y = power.sel(noise_diode_enabled=True, drop=True) / power.sel(
        noise_diode_enabled=False, drop=True
    )

    T = (Ton - Y * Toff) / (Y - 1)
    T.name = 'T'
    T.attrs = {'units': 'K'}

    return T


def _y_factor_power_corrections(
    dataset: xr.Dataset, enr_dB: float, Tamb: float, Tref=290.0
) -> xr.Dataset:
    # TODO: check that this works for xr.DataArray inputs in (enr_dB, Tamb)

    kwargs = dict(list(locals().items())[1:])

    power = (
        dataset.power_time_series.sel(power_detector='rms', drop=True)
        .pipe(lambda x: 10 ** (x / 10.0))
        .mean(dim='time_elapsed')
    )
    power.name = 'RMS power'

    T = _y_factor_temperature(power, **kwargs)
    T.name = 'Noise temperature'
    T.attrs = {'units': 'K'}

    noise_figure = 10 * np.log10(T / Tref + 1)
    noise_figure.name = 'Noise figure'
    noise_figure.attrs = {'units': 'dB'}

    power_correction = (Boltzmann * power.analysis_bandwidth * 1000 * T) / (
        power.sel(noise_diode_enabled=True, drop=True)
    )
    power_correction.name = 'Input power scaling correction'
    power_correction.attrs = {'units': 'mW'}

    return xr.Dataset(
        {
            'temperature': T,
            'noise_figure': noise_figure,
            'power_correction': power_correction,
        }
    )


def _y_factor_frequency_response_correction(
    dataset: xr.DataArray,
    fc_temperatures: xr.DataArray,
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
    dataset: xr.Dataset, enr_dB: float, Tamb: float, Tref=290.0
) -> xr.Dataset:
    kwargs = locals()
    ret = _y_factor_power_corrections(**kwargs)
    ret['baseband_frequency_response'] = _y_factor_frequency_response_correction(
        **kwargs, fc_temperatures=ret.temperature
    )
    return ret


def resampling_correction(
    iq: fourier.Array,
    capture: structs.RadioCapture,
    radio: util.RadioBase,
    force_calibration: Optional[xr.Dataset] = None,
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

    xp = import_cupy_with_fallback_warning()

    # create a buffer large enough for post-processing seeded with a copy of the IQ
    _, buf_size = util.get_capture_buffer_sizes(radio._master_clock_rate, capture)
    buf = xp.empty(buf_size, dtype='complex64')
    iq = buf[: iq.size] = xp.asarray(iq)

    fs_backend, lo_offset, analysis_filter = util.design_capture_filter(
        radio._master_clock_rate, capture
    )

    fft_size = analysis_filter['fft_size']

    fft_size_out, noverlap, overlap_scale, _ = fourier._ola_filter_parameters(
        iq.size,
        window=analysis_filter['window'],
        fft_size_out=analysis_filter.get('fft_size_out', fft_size),
        fft_size=fft_size,
        extend=True,
    )

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
        sel = corrections.power_correction.sel(
            channel=capture.channel,
            gain=capture.gain,
            lo_shift=capture.lo_shift,
            sample_rate=capture.sample_rate,
            analysis_bandwidth=capture.analysis_bandwidth,
            gpu_resample=capture.gpu_resample,
            drop=True,
        )

        # allow interpolation between sample points in these fields
        sel = sel.interp(center_frequency=capture.center_frequency)

        power_scale = float(sel)

    if fft_size == fft_size_out:
        # nothing to do here
        if power_scale is not None:
            iq *= power_scale
        if analysis_filter['passband'] != (None, None):
            iq = waveform.iir_filter(
                iq,
                capture,
                passband_ripple=0.5,
                stopband_attenuation=80,
                transition_bandwidth=500e3,
                out=iq,
            )

        return iq[util.TRANSIENT_HOLDOFF_WINDOWS * fft_size_out :]

    w = fourier._get_window(analysis_filter['window'], fft_size, fftbins=False, xp=xp)

    freqs, _, xstft = fourier.stft(
        iq,
        fs=fs_backend,
        window=w,
        nperseg=analysis_filter['fft_size'],
        noverlap=round(analysis_filter['fft_size'] * overlap_scale),
        axis=axis,
        truncate=False,
        out=buf,
    )

    # set the passband roughly equal to the 3 dB bandwidth based on ENBW
    enbw = (
        fs_backend
        / fft_size
        * fourier.equivalent_noise_bandwidth(
            analysis_filter['window'], fft_size, fftbins=False
        )
    )
    passband = analysis_filter['passband']

    if fft_size_out != analysis_filter['fft_size']:
        freqs, xstft = fourier.downsample_stft(
            freqs,
            xstft,
            fft_size_out=fft_size_out,
            passband=passband,
            axis=axis,
            out=buf,
        )
    else:
        fourier.zero_stft_by_freq(
            freqs,
            xstft,
            passband=(passband[0] + enbw, passband[1] - enbw),
            axis=axis,
        )

    iq = fourier.istft(
        xstft,
        iq.shape[axis],
        fft_size=fft_size_out,
        noverlap=noverlap,
        out=buf,
        axis=axis,
    )

    if power_scale is not None:
        iq *= np.sqrt(power_scale)

    return iq[util.TRANSIENT_HOLDOFF_WINDOWS * fft_size_out :]
