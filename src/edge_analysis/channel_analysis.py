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
    return len(w) * np.sum(w**2) / np.sum(w)**2


@lru_cache(8)
def generate_iir_lpf(
    rp_dB: (float|int),
    rs_dB: (float|int),
    cutoff_Hz: (float|int),
    width_Hz: (float|int),
    fs: (float|int),
) -> np.ndarray:
    """
    Generate an elliptic IIR low pass filter.

    Parameters
    ----------
    rp_dB: (float, int)
        Maximum passband ripple below unity gain, in dB.
    rs_dB: (float, int)
        Minimum stopband attenuation, in dB.
    cutoff_Hz: (float, int)
        Filter cutoff frequency, in Hz.
    width_Hz: (float, int)
        Passband-to-stopband transition width, in Hz.
    Fs_MHz: (float, int)
        Sampling rate, in MHz.
    plot_response: bool
        If True, plot the filter response.

    Returns
    -------
    sos: numpy.ndarray
        Second-order sections representation of the IIR filter.
    """

    # Generate filter
    ord, wn = signal.ellipord(cutoff_Hz, cutoff_Hz + width_Hz, rp_dB, rs_dB, False, fs)
    sos = signal.ellip(ord, rp_dB, rs_dB, wn, "lowpass", False, "sos", fs)

    return sos


@lru_cache
def _label_detector_coords(
    detectors: tuple[str]
):
    array = xr.DataArray(list(detectors), dims='power_detector', attrs={'label': 'Power detector'})
    return {array.dims[0]: array}


@lru_cache
def _label_cyclic_power_coords(
    fs: float,
    cyclic_period: float,
    detector_period: float,
    cyclic_statistics: list[str]
):
    cyclic_statistic = xr.DataArray(list(cyclic_statistics), dims='cyclic_statistic', attrs={'label': 'Cyclic statistic'})

    lag_count = int(np.rint(cyclic_period/detector_period))

    cyclic_lag = xr.DataArray(
        np.arange(lag_count) * detector_period,
        dims='cyclic_lag',
        attrs={'label': 'Cyclic lag', 'units': 's'}
    )

    return {
        cyclic_statistic.dims[0]: cyclic_statistic,
        cyclic_lag.dims[0]: cyclic_lag
    }


def cyclic_channel_power(
    iq, fs, analysis_bandwidth, cyclic_period: float, detector_period: float,
    detectors: list[str]=('rms', 'peak'), cyclic_statistics: list[str]=('min', 'mean', 'max')
) -> xr.DataArray:
    metadata = dict(locals())
    del metadata['iq'], metadata['detectors'], metadata['cyclic_statistics']

    detectors = tuple(detectors)
    cyclic_statistics = tuple(cyclic_statistics)

    data_dict = iq_to_cyclic_power(
        iq, 1/fs,
        cyclic_period=cyclic_period,
        detector_period=detector_period,
        detectors=detectors,
        cycle_stats=cyclic_statistics,
    )

    detector_coords = _label_detector_coords(detectors)
    cyclic_coords = _label_cyclic_power_coords(fs, cyclic_period=cyclic_period, detector_period=detector_period, cyclic_statistics=cyclic_statistics)
    coords = xr.Coordinates({**detector_coords, **cyclic_coords})

    attrs = {
        'label': 'Channel power',
        'units': f'dBm/{analysis_bandwidth/1e6} MHz',
        **metadata
    }

    return xr.DataArray(
        [[v for v in vset.values()] for vset in data_dict.values()],
        coords = coords,
        dims = list(coords.keys()),
        attrs = attrs
    )

@lru_cache
def _label_detector_time_coords(
    detector_period: float,
    length: int
):

    array = xr.DataArray(
        np.arange(length) * detector_period,
        dims='time_elapsed',
        attrs={'label': 'Acquisition time elapsed', 'units': 's'}
    )

    return {
        array.dims[0]: array,
    }

def power_time_series(iq, *, fs: float, analysis_bandwidth: float, detector_period: float, detectors=('rms', 'peak')) -> xr.DataArray:
    metadata = dict(locals())
    del metadata['iq'], metadata['detectors']

    data = [
        iqwaveform.powtodB(iqwaveform.iq_to_bin_power(iq, Ts=1/fs, Tbin=detector_period, kind=detector))
        for detector in detectors
    ]

    time_coords = _label_detector_time_coords(detector_period, len(data[0]))
    detector_coords = _label_detector_coords(detectors)

    coords = {**detector_coords, **time_coords}

    return xr.DataArray(
        data,
        coords = coords,
        dims = list(coords.keys()),
        attrs = {
            'label': 'Channel power',
            'units': f'dBm/{analysis_bandwidth/1e6} MHz',
            **metadata
        }
    )

@lru_cache
def _get_apd_bins(lo, hi, count, xp=np):
    return xp.linspace(lo, hi, count, dtype='float32')

@lru_cache
def _label_apd_power_bins(lo, hi, count, xp, units):
    params = dict(locals())
    del params['units']

    bins = _get_apd_bins(**params)
    array = xr.DataArray(bins, dims='channel_power', attrs={'label': 'Channel power', 'units': units})
    return {array.dims[0]: array}

def amplitude_probability_distribution(iq, *, analysis_bandwidth, power_low, power_high, power_count) -> xr.DataArray:
    metadata = dict(locals())
    del metadata['iq']

    xp = array_namespace(iq)

    bin_params = {'lo': power_low, 'hi': power_high, 'count': power_count}

    bins = _get_apd_bins(xp=xp, **bin_params)
    ccdf = iqwaveform.sample_ccdf(iqwaveform.envtodB(iq), bins)
    units = f'dBm/{analysis_bandwidth/1e6} MHz'

    return xr.DataArray(
        ccdf,
        coords=_label_apd_power_bins(xp=xp, units=units, **bin_params),
        attrs={
            'label': 'Amplitude probability distribution',
            **metadata
        }
    )

@lru_cache
def _baseband_frequency_to_coords(fs: float, fft_size: int, time_size: int, overlap_frac: float = 0, xp=np):
    freqs, times = iqwaveform.fourier._get_stft_axes(**locals())

    array = xr.DataArray(
        freqs,
        dims='baseband_frequency',
        attrs={'label': 'Baseband frequency', 'units': 'Hz'}
    )
    return {array.dims[0]: array}

@lru_cache
def _persistence_stats_to_coords(stat_names):
    array = xr.DataArray([str(n) for n in stat_names], dims='persistence_statistic', attrs={'label': 'Persistence statistic'})
    return {array.dims[0]: array}

def persistence_spectrum(iq: Array, *, fs: float, window, fres: float, overlap_frac=0, quantiles: list[float]) -> xr.DataArray:
    persistence_args = dict(locals())
    del persistence_args['iq']

    metadata = dict(persistence_args)
    del metadata['quantiles']

    # TODO: validate fs % fres
    # TODO: implement other statistics, such as mean
    xp = array_namespace(iq)

    fft_size = int(fs/fres)
    _, times, X = iqwaveform.stft(iq, window=window, fs=fs, nperseg=fft_size)
    spectrum = xp.quantile(iqwaveform.envtopow(X), quantiles, axis=0)

    freq_coords = _baseband_frequency_to_coords(
        fs=fs, fft_size=fft_size, time_size=len(times), overlap_frac=overlap_frac
    )

    stat_coords = _persistence_stats_to_coords(tuple(quantiles))

    coords = {**freq_coords, **stat_coords}

    enbw = fres*equivalent_noise_bandwidth(window, fft_size)

    data = xr.DataArray(
        iqwaveform.powtodB(spectrum.T),
        dims=coords.keys(),
        coords=coords,
        attrs={
            'label': 'Power spectral density',
            'units': f'dBm/{enbw/1e3:0.3f} kHz',
            'enbw': enbw,
            **metadata
        }
    )

    return data