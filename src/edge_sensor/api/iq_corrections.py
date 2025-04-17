from __future__ import annotations
import functools
import msgspec
import typing
import pickle
import gzip
from pathlib import Path
import array_api_compat

from . import util
from .captures import split_capture_channels
from channel_analysis.api.util import except_on_low_memory

from .radio import base, RadioDevice, design_capture_filter
from . import structs

if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import xarray as xr
    import scipy
    import iqwaveform
    import labbench as lb

else:
    np = util.lazy_import('numpy')
    xr = util.lazy_import('xarray')
    scipy = util.lazy_import('scipy')
    iqwaveform = util.lazy_import('iqwaveform')
    lb = util.lazy_import('labbench')


@functools.lru_cache
def read_calibration_corrections(path):
    if path is None:
        return None

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
    spectrum = dataset.power_spectral_density.sel(
        frequency_statistic='mean', drop=True
    ).pipe(lambda x: 10 ** (x / 10.0))

    all_T = _y_factor_temperature(spectrum, enr_dB=20.87, Tamb=294.5389)

    # normalize the power correction at each center frequency, and then average the result across center frequency

    temp_norm = fc_temperatures.broadcast_like(all_T) / all_T
    frequency_response = temp_norm.median(dim='center_frequency')

    frequency_response.name = 'Baseband power scaling correction'
    frequency_response.attrs = {'units': 'unitless'}

    return frequency_response


def compute_y_factor_corrections(
    dataset: 'xr.Dataset', enr_dB: float, Tamb: float, Tref=290.0
) -> 'xr.Dataset':
    kwargs = locals()
    ret = _y_factor_power_corrections(**kwargs)
    # ret['baseband_frequency_response'] = _y_factor_frequency_response_correction(
    #     **kwargs, fc_temperatures=ret.temperature
    # )
    return ret


def summarize_noise_figure(corrections: 'xr.Dataset', **sel) -> 'pd.DataFrame':
    max_gain = float(corrections.gain.max())
    max_nf = corrections.noise_figure.sel(gain=max_gain, **sel, drop=True).squeeze()
    stacked = max_nf.stack(condition=max_nf.dims).dropna('condition')
    return stacked.to_dataframe()[['noise_figure']]


def summarize_power_corrections(corrections: 'xr.Dataset', **sel) -> 'pd.DataFrame':
    max_gain = float(corrections.gain.max())
    corr = corrections.power_correction.sel(gain=max_gain, **sel, drop=True).squeeze()
    stacked = corr.stack(condition=corr.dims).dropna('condition')
    return stacked.to_dataframe()[['power_correction']]


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


@functools.lru_cache()
def lookup_power_correction(
    cal_data: Path | 'xr.Dataset' | None, capture: structs.RadioCapture, xp
):
    if isinstance(cal_data, xr.Dataset):
        corrections = cal_data
    elif cal_data:
        corrections = read_calibration_corrections(cal_data)
    else:
        return None

    power_scale = []

    for capture_chan in split_capture_channels(capture):
        # these fields must match the calibration conditions exactly
        exact_matches = dict(
            channel=capture_chan.channel,
            gain=capture_chan.gain,
            lo_shift=capture_chan.lo_shift,
            sample_rate=capture_chan.sample_rate,
            analysis_bandwidth=capture_chan.analysis_bandwidth or np.inf,
            host_resample=capture_chan.host_resample,
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

        sel = sel.squeeze(drop=True).dropna('center_frequency')

        if sel.size == 0:
            raise ValueError(
                'no calibration data is available for this combination of sampling parameters'
            )
        elif capture_chan.center_frequency > sel.center_frequency.max():
            raise ValueError(
                f'center_frequency {capture_chan.center_frequency / 1e6} MHz exceeds calibration max {sel.center_frequency.max() / 1e6} MHz'
            )
        elif capture_chan.center_frequency < sel.center_frequency.min():
            raise ValueError(
                f'center_frequency {capture_chan.center_frequency / 1e6} MHz is below calibration min {sel.center_frequency.min() / 1e6} MHz'
            )

        # allow interpolation between sample points in these fields
        sel = sel.interp(center_frequency=capture_chan.center_frequency)

        power_scale.append(float(sel))

    return xp.asarray(power_scale, dtype='float32')


def _power_scale(cal_power_scale, dtype_iq_scale):
    if cal_power_scale is None and dtype_iq_scale is None:
        return None

    if dtype_iq_scale is None:
        dtype_iq_scale = 1
    if cal_power_scale is None:
        cal_power_scale = 1

    return cal_power_scale * (dtype_iq_scale**2)


def resampling_correction(
    iq: 'iqwaveform.util.Array',
    capture: structs.RadioCapture,
    radio: RadioDevice,
    force_calibration: typing.Optional['xr.Dataset'] = None,
    *,
    overwrite_x=False,
    axis=1,
):
    """apply a bandpass filter implemented through STFT overlap-and-add.

    Args:
        iq: the input waveform, as a pinned array
        capture: the capture filter specification structure
        radio: the radio instance that performed the capture
        force_calibration: if specified, this calibration dataset is used rather than loading from file
        axis: the axis of `x` along which to compute the filter

    Returns:
        the filtered IQ capture
    """

    xp = iqwaveform.util.array_namespace(iq)

    if array_api_compat.is_cupy_array(iq):
        util.configure_cupy()

    if radio._transport_dtype == 'int16':
        dtype_scale = 1.0 / float(np.iinfo(radio._transport_dtype).max)
    else:
        dtype_scale = None

    bare_capture = msgspec.structs.replace(capture, start_time=None)
    cal_data = radio.calibration if force_calibration is None else force_calibration
    cal_scale = lookup_power_correction(cal_data, bare_capture, xp)

    power_scale = _power_scale(cal_scale, dtype_scale)

    fs, _, analysis_filter = design_capture_filter(radio.base_clock_rate, capture)

    except_on_low_memory()

    needs_resample = base.needs_resample(analysis_filter, capture)

    # apply the filter here, where the size of y is minimized
    if np.isfinite(capture.analysis_bandwidth):
        h = iqwaveform.design_fir_lpf(
            bandwidth=capture.analysis_bandwidth,
            sample_rate=fs,
            transition_bandwidth=250e3,
            numtaps=base.FILTER_SIZE,
            xp=xp,
        )
        pad = base._get_filter_pad(capture)
        iq = iqwaveform.oaconvolve(iq, h[xp.newaxis, :], 'same', axes=axis)
        iq = iqwaveform.util.axis_slice(iq, pad, iq.shape[axis], axis=axis)

    if not needs_resample:
        # bail here if filtering or resampling needed
        size = round(capture.duration * capture.sample_rate)
        iq = iq[:, :size]
        if power_scale is not None:
            iq *= np.sqrt(power_scale)
        return iq

    except_on_low_memory()

    size_out = round(capture.duration * capture.sample_rate)
    iq = iqwaveform.fourier.resample(
        iq,
        size_out,
        overwrite_x=overwrite_x,
        axis=axis,
        scale=1 if power_scale is None else power_scale,
    )

    assert iq.shape[axis] == size_out

    # nfft = analysis_filter['nfft']
    # nfft_out, noverlap, overlap_scale, _ = iqwaveform.fourier._ola_filter_parameters(
    #     iq.size,
    #     window=analysis_filter['window'],
    #     nfft_out=analysis_filter.get('nfft_out', nfft),
    #     nfft=nfft,
    #     extend=True,
    # )

    # y = iqwaveform.stft(
    #     iq,
    #     fs=fs,
    #     window=analysis_filter['window'],
    #     nperseg=nfft,
    #     noverlap=round(nfft * overlap_scale),
    #     axis=axis,
    #     truncate=False,
    #     overwrite_x=overwrite_x,
    #     return_axis_arrays=False,
    # )

    # # resample
    # except_on_low_memory()
    # if nfft_out < nfft:
    #     # downsample by trimming frequency
    #     freqs = iqwaveform.fftfreq(nfft, 1 / fs)
    #     freqs, y = iqwaveform.fourier.downsample_stft(
    #         freqs, y, nfft_out=nfft_out, axis=axis, out=y
    #     )
    # elif nfft_out > nfft:
    #     # upsample by zero-padding frequency
    #     pad_left = (nfft_out - nfft) // 2
    #     pad_right = pad_left + (nfft_out - nfft) % 2
    #     y = iqwaveform.util.pad_along_axis(y, [[pad_left, pad_right]], axis=axis + 1)

    # if filter_domain == 'frequency':
    #     lb.util.logger.debug('applying filter in frequency domain')
    #     y = iqwaveform.fourier.stft_fir_lowpass(
    #         y,
    #         sample_rate=capture.sample_rate,
    #         bandwidth=capture.analysis_bandwidth,
    #         transition_bandwidth=500e3,
    #         axis=axis,
    #         out=y
    #     )
    # del iq

    # # reconstruct into a resampled waveform
    # except_on_low_memory()
    # iq = iqwaveform.istft(
    #     y, nfft=nfft_out, noverlap=noverlap, axis=axis, overwrite_x=True
    # )
    # scale = iq.size/size_in

    # start the capture after the padding for transients
    # size_out = round(capture.duration * capture.sample_rate)
    # assert size_out <= iq.shape[axis]
    # iq = iqwaveform.util.axis_slice(iq, -size_out, iq.shape[axis], axis=axis)
    # assert iq.shape[axis] == size_out

    # # apply final scaling
    # if power_scale is not None:
    #     scale *= np.sqrt(power_scale)
    # iq *= scale

    return iq
