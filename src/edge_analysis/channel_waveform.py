from __future__ import annotations
import iqwaveform
from functools import lru_cache
from iqwaveform.power_analysis import iq_to_cyclic_power
import numpy as np
from scipy import signal
from iqwaveform.util import array_namespace
import xarray as xr
from array_api_strict._typing import Array


@lru_cache
def equivalent_noise_bandwidth(window, N):
    """return the equivalent noise bandwidth (ENBW) of a window, in bins"""
    w = iqwaveform.fourier._get_window(window, N)
    return len(w) * np.sum(w**2) / np.sum(w) ** 2


@lru_cache(8)
def generate_iir_lpf(
    passband_ripple_dB: float | int,
    stopband_attenuation_dB: float | int,
    cutoff_Hz: float | int,
    transition_bandwidth_Hz: float | int,
    fs: float | int,
) -> np.ndarray:
    """
    Generate an elliptic IIR low pass filter.


    Args:
        passband_ripple_dB:
            Maximum passband ripple below unity gain, in dB.
        stopband_attenuation_dB:
            Minimum stopband attenuation, in dB.
        cutoff_Hz:
            Filter cutoff frequency, in Hz.
        transition_bandwidth_Hz:
            Passband-to-stopband transition width, in Hz.
        fs:
            Sampling rate, in Hz.

    Returns:
        Second-order sections (sos) representation of the IIR filter.
    """

    # Generate filter
    ord, wn = signal.ellipord(
        cutoff_Hz,
        cutoff_Hz + transition_bandwidth_Hz,
        passband_ripple_dB,
        stopband_attenuation_dB,
        False,
        fs,
    )
    sos = signal.ellip(
        ord,
        passband_ripple_dB,
        stopband_attenuation_dB,
        wn,
        'lowpass',
        False,
        'sos',
        fs,
    )

    return sos


@lru_cache
def _label_detector_coords(detectors: tuple[str]):
    array = xr.DataArray(
        list(detectors), dims='power_detector', attrs={'label': 'Power detector'}
    )
    return {array.dims[0]: array}


@lru_cache
def _label_cyclic_power_coords(
    fs: float,
    cyclic_period: float,
    detector_period: float,
    cyclic_statistics: list[str],
):
    cyclic_statistic = xr.DataArray(
        list(cyclic_statistics),
        dims='cyclic_statistic',
        attrs={'label': 'Cyclic statistic'},
    )

    lag_count = int(np.rint(cyclic_period / detector_period))

    cyclic_lag = xr.DataArray(
        np.arange(lag_count) * detector_period,
        dims='cyclic_lag',
        attrs={'label': 'Cyclic lag', 'units': 's'},
    )

    return {cyclic_statistic.dims[0]: cyclic_statistic, cyclic_lag.dims[0]: cyclic_lag}


def cyclic_channel_power(
    iq,
    fs,
    analysis_bandwidth,
    cyclic_period: float,
    detector_period: float,
    detectors: list[str] = ('rms', 'peak'),
    cyclic_statistics: list[str] = ('min', 'mean', 'max'),
) -> xr.DataArray:
    metadata = {'cyclic_period': cyclic_period, 'detector_period': detector_period}

    detectors = tuple(detectors)
    cyclic_statistics = tuple(cyclic_statistics)

    data_dict = iq_to_cyclic_power(
        iq,
        1 / fs,
        cyclic_period=cyclic_period,
        detector_period=detector_period,
        detectors=detectors,
        cycle_stats=cyclic_statistics,
    )

    detector_coords = _label_detector_coords(detectors)
    cyclic_coords = _label_cyclic_power_coords(
        fs,
        cyclic_period=cyclic_period,
        detector_period=detector_period,
        cyclic_statistics=cyclic_statistics,
    )
    coords = xr.Coordinates({**detector_coords, **cyclic_coords})

    attrs = {
        'label': 'Channel power',
        'units': f'dBm/{analysis_bandwidth/1e6} MHz',
        **metadata,
    }

    return xr.DataArray(
        [[v for v in vset.values()] for vset in data_dict.values()],
        coords=coords,
        dims=list(coords.keys()),
        attrs=attrs,
        name='cyclic_channel_power',
    )


@lru_cache
def _label_detector_time_coords(detector_period: float, length: int):
    array = xr.DataArray(
        np.arange(length) * detector_period,
        dims='time_elapsed',
        attrs={'label': 'Acquisition time elapsed', 'units': 's'},
    )

    return {
        array.dims[0]: array,
    }


def power_time_series(
    iq,
    *,
    fs: float,
    analysis_bandwidth: float,
    detector_period: float,
    detectors=('rms', 'peak'),
) -> xr.DataArray:
    metadata = {'detector_period': detector_period}

    data = [
        iqwaveform.powtodB(
            iqwaveform.iq_to_bin_power(
                iq, Ts=1 / fs, Tbin=detector_period, kind=detector
            )
        )
        for detector in detectors
    ]

    time_coords = _label_detector_time_coords(detector_period, len(data[0]))
    detector_coords = _label_detector_coords(detectors)

    coords = {**detector_coords, **time_coords}

    return xr.DataArray(
        data,
        coords=coords,
        dims=list(coords.keys()),
        name='power_time_series',
        attrs={
            'label': 'Channel power',
            'units': f'dBm/{analysis_bandwidth/1e6} MHz',
            **metadata,
        },
    )


@lru_cache
def _get_apd_bins(lo, hi, count, xp=np):
    return xp.linspace(lo, hi, count, dtype='float32')


@lru_cache
def _label_apd_power_bins(lo, hi, count, xp, units):
    params = dict(locals())
    del params['units']

    bins = _get_apd_bins(**params)
    array = xr.DataArray(
        bins, dims='channel_power', attrs={'label': 'Channel power', 'units': units}
    )
    return {array.dims[0]: array}


def amplitude_probability_distribution(
    iq,
    *,
    fs: float = None,
    analysis_bandwidth: float,
    power_low: float,
    power_high: float,
    power_count: float,
) -> xr.DataArray:
    xp = array_namespace(iq)

    bin_params = {'lo': power_low, 'hi': power_high, 'count': power_count}

    bins = _get_apd_bins(xp=xp, **bin_params)
    ccdf = iqwaveform.sample_ccdf(iqwaveform.envtodB(iq), bins)
    units = f'dBm/{analysis_bandwidth/1e6} MHz'

    return xr.DataArray(
        ccdf,
        coords=_label_apd_power_bins(xp=xp, units=units, **bin_params),
        name='amplitude_probability_distribution',
        attrs={'label': 'Amplitude probability distribution', **bin_params},
    )


@lru_cache
def _baseband_frequency_to_coords(
    fs: float, fft_size: int, time_size: int, overlap_frac: float = 0, xp=np
):
    freqs, times = iqwaveform.fourier._get_stft_axes(**locals())

    array = xr.DataArray(
        freqs,
        dims='baseband_frequency',
        attrs={'label': 'Baseband frequency', 'units': 'Hz'},
    )
    return {array.dims[0]: array}


@lru_cache
def _persistence_stats_to_coords(stat_names):
    array = xr.DataArray(
        [str(n) for n in stat_names],
        dims='persistence_statistic',
        attrs={'label': 'Persistence statistic'},
    )
    return {array.dims[0]: array}


def persistence_spectrum(
    iq: Array,
    *,
    fs: float,
    analysis_bandwidth=None,
    window,
    resolution: float,
    fractional_overlap=0,
    quantiles: list[float],
) -> xr.DataArray:
    # TODO: support other persistence statistics, such as mean

    if not iqwaveform.power_analysis.isroundmod(fs, resolution):
        # need fs/resolution to give us a counting number
        raise ValueError('fs/resolution must be a counting number')

    fft_size = int(fs / resolution)
    enbw = resolution * equivalent_noise_bandwidth(window, fft_size)
    metadata = {
        'window': window,
        'resolution_Hz': resolution,
        'fractional_overlap': fractional_overlap,
        'noise_bandwidth_Hz': enbw,
        'fft_size': fft_size
    }

    xp = array_namespace(iq)

    _, times, X = iqwaveform.stft(iq, window=window, fs=fs, nperseg=fft_size)
    spectrum = xp.quantile(iqwaveform.envtopow(X), quantiles, axis=0)

    freq_coords = _baseband_frequency_to_coords(
        fs=fs, fft_size=fft_size, time_size=len(times), overlap_frac=fractional_overlap
    )

    stat_coords = _persistence_stats_to_coords(tuple(quantiles))

    coords = {**freq_coords, **stat_coords}


    data = xr.DataArray(
        iqwaveform.powtodB(spectrum.T),
        dims=coords.keys(),
        coords=coords,
        name='persistence_spectrum',
        attrs={
            'label': 'Power spectral density',
            'units': f'dBm/{enbw/1e3:0.3f} kHz',
            **metadata,
        },
    )

    return data


def from_spec(
    iq,
    fs,
    analysis_bandwidth,
    *,
    filter_spec: dict = {
        'passband_ripple_dB': 0.1,
        'stopband_attenuation_dB': 70,
        'transition_bandwidth_Hz': 250e3,
    },
    analysis_spec: dict[str, dict[str]] = {},
):
    VALID_FUNCS = {
        'power_time_series',
        'persistence_spectrum',
        'amplitude_probability_distribution',
        'cyclic_channel_power',
    }

    if len(analysis_spec.keys() - VALID_FUNCS) > 0:
        raise KeyError(
            f'analysis_spec keys may only be analysis function names: {VALID_FUNCS}'
        )

    if filter_spec is not None:
        sos = generate_iir_lpf(cutoff_Hz=analysis_bandwidth / 2, fs=fs, **filter_spec)
        iq = signal.sosfilt(sos.astype('float32'), iq)

    acq_kws = {'iq': iq, 'fs': fs, 'analysis_bandwidth': analysis_bandwidth}

    arrays = [
        globals()[func_name](**acq_kws, **func_kws)
        for func_name, func_kws in analysis_spec.items()
    ]

    return xr.Dataset(
        {
            a.name: a
            for a in arrays
        },
        attrs={
            'fs': fs,
            'analysis_bandwidth': analysis_bandwidth,
            'filter_specification': filter_spec,
        },
    )
