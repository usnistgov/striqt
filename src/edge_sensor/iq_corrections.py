from __future__ import annotations
from functools import lru_cache
import typing
import pickle
import gzip

from channel_analysis import waveform, type_stubs

from .radio import RadioDevice, get_capture_buffer_sizes, design_capture_filter
from .radio.base import TRANSIENT_HOLDOFF_WINDOWS
from . import structs
from .util import import_cupy_with_fallback

import labbench as lb

if typing.TYPE_CHECKING:
    import numpy as np
    import xarray as xr
    import scipy
    import iqwaveform
else:
    np = lb.util.lazy_import('numpy')
    xr = lb.util.lazy_import('xarray')
    scipy = lb.util.lazy_import('scipy')
    iqwaveform = lb.util.lazy_import('iqwaveform')


@lru_cache
def read_calibration_corrections(path):
    with gzip.GzipFile(path, 'rb') as fd:
        return pickle.load(fd)


def save_calibration_corrections(path, corrections: type_stubs.DatasetType):
    with gzip.GzipFile(path, 'wb') as fd:
        pickle.dump(corrections, fd)


def _y_factor_temperature(
    power: type_stubs.DataArrayType, enr_dB: float, Tamb: float, Tref=290.0
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
    dataset: type_stubs.DatasetType, enr_dB: float, Tamb: float, Tref=290.0
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

    power_correction = (
        scipy.constants.Boltzmann * power.analysis_bandwidth * 1000 * T
    ) / (power.sel(noise_diode_enabled=True, drop=True))
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
    dataset: type_stubs.DataArrayType,
    fc_temperatures: type_stubs.DataArrayType,
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
    dataset: type_stubs.DatasetType, enr_dB: float, Tamb: float, Tref=290.0
) -> xr.Dataset:
    kwargs = locals()
    ret = _y_factor_power_corrections(**kwargs)
    ret['baseband_frequency_response'] = _y_factor_frequency_response_correction(
        **kwargs, fc_temperatures=ret.temperature
    )
    return ret


def resampling_correction(
    iq: iqwaveform.fourier.Array,
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

    xp = import_cupy_with_fallback()

    # create a buffer large enough for post-processing seeded with a copy of the IQ
    _, buf_size = get_capture_buffer_sizes(radio, capture)
    buf = xp.empty(buf_size, dtype='complex64')
    iq = buf[: iq.size] = xp.asarray(iq)

    fs_backend, lo_offset, analysis_filter = design_capture_filter(
        radio._master_clock_rate, capture
    )

    fft_size = analysis_filter['fft_size']

    fft_size_out, noverlap, overlap_scale, _ = (
        iqwaveform.fourier._ola_filter_parameters(
            iq.size,
            window=analysis_filter['window'],
            fft_size_out=analysis_filter.get('fft_size_out', fft_size),
            fft_size=fft_size,
            extend=True,
        )
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

        return iq[TRANSIENT_HOLDOFF_WINDOWS * fft_size_out :]

    w = iqwaveform.fourier._get_window(
        analysis_filter['window'], fft_size, fftbins=False, xp=xp
    )

    freqs, _, xstft = iqwaveform.fourier.stft(
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
        * iqwaveform.fourier.equivalent_noise_bandwidth(
            analysis_filter['window'], fft_size, fftbins=False
        )
    )
    passband = analysis_filter['passband']

    if fft_size_out != analysis_filter['fft_size']:
        freqs, xstft = iqwaveform.fourier.downsample_stft(
            freqs,
            xstft,
            fft_size_out=fft_size_out,
            passband=passband,
            axis=axis,
            out=buf,
        )
    else:
        iqwaveform.fourier.zero_stft_by_freq(
            freqs,
            xstft,
            passband=(passband[0] + enbw, passband[1] - enbw),
            axis=axis,
        )

    out_size = round(capture.duration * capture.sample_rate)
    iq = iqwaveform.fourier.istft(
        xstft,
        out_size,
        fft_size=fft_size_out,
        noverlap=noverlap,
        out=buf,
        axis=axis,
    )

    if power_scale is not None:
        iq *= np.sqrt(power_scale)

    return iq
