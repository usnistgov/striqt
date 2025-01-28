from __future__ import annotations
import functools
import msgspec
import typing
import pickle
import gzip
from pathlib import Path

from . import util
from .captures import split_capture_channels
from channel_analysis.api.util import free_mempool_on_low_memory, compute_lock

from .radio import base, RadioDevice, design_capture_filter
from . import structs

if typing.TYPE_CHECKING:
    import numpy as np
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

    return xp.asarray(power_scale, dtype='float32')[:, np.newaxis]


@compute_lock()
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

    xp = iqwaveform.fourier.array_namespace(iq)

    with lb.stopwatch('power correction lookup', threshold=10e-3, logger_level='debug'):
        bare_capture = msgspec.structs.replace(capture, start_time=None)
        power_scale = lookup_power_correction(
            force_calibration or radio.calibration, bare_capture, xp
        )

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

    if not base.needs_stft(analysis_filter, capture):
        # no filtering or resampling needed
        print('return direct - ', type(iq))
        iq = iq[:, : round(capture.duration * capture.sample_rate)]
        if power_scale is not None:
            iq *= np.sqrt(power_scale)
        return iq

    # set the passband roughly equal to the 3 dB bandwidth based on ENBW
    freq_res = fs_backend / nfft
    enbw_bins = iqwaveform.fourier.equivalent_noise_bandwidth(
        analysis_filter['window'], nfft, fftbins=False
    )
    enbw = enbw_bins * freq_res 
    passband = analysis_filter['passband']

    freqs, _, y = iqwaveform.fourier.stft(
        iq,
        fs=fs_backend,
        window=analysis_filter['window'],
        nperseg=nfft,
        noverlap=round(nfft * overlap_scale),
        axis=axis,
        truncate=False,
        overwrite_x=overwrite_x
    )

    free_mempool_on_low_memory()

    if nfft_out < nfft:
        # downsample applies the filter as well
        freqs, y = iqwaveform.fourier.downsample_stft(
            freqs,
            y,
            nfft_out=nfft_out,
            passband=passband,
            axis=axis
        )
    elif nfft_out > nfft:
        # upsample
        if np.isfinite(capture.analysis_bandwidth):
            iqwaveform.fourier.zero_stft_by_freq(
                freqs,
                y,
                passband=(passband[0] + enbw / 2, passband[1] - enbw / 2),
                axis=axis,
            )

        pad_left = (nfft_out - nfft) // 2
        pad_right = pad_left + (nfft_out - nfft) % 2

        y = iqwaveform.util.pad_along_axis(
            y, [[pad_left, pad_right]], axis=axis + 1
        )

    else:
        # nfft_out == nfft
        iqwaveform.fourier.zero_stft_by_freq(
            freqs,
            y,
            passband=(passband[0] + enbw, passband[1] - enbw),
            axis=axis,
        )

    del iq

    free_mempool_on_low_memory()

    iq = iqwaveform.fourier.istft(
        y, nfft=nfft_out, noverlap=noverlap, axis=axis, overwrite_x=True
    )

    del y

    free_mempool_on_low_memory()

    # start the capture after the transient holdoff window
    iq_size_out = round(capture.duration * capture.sample_rate)
    i0 = nfft_out // 2
    assert i0 + iq_size_out <= iq.shape[axis]
    iq = iqwaveform.util.axis_slice(iq, i0, i0 + iq_size_out, axis=axis)

    if power_scale is None and nfft == nfft_out:
        pass
    elif power_scale is None:
        iq *= np.sqrt(nfft_out / nfft)
    else:
        iq *= np.sqrt(power_scale) * np.sqrt(nfft_out / nfft)

    return iq
