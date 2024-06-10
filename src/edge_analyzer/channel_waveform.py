from __future__ import annotations
import iqwaveform
from functools import lru_cache
from iqwaveform.power_analysis import iq_to_cyclic_power
import numpy as np
from scipy import signal
from iqwaveform.util import array_namespace
from iqwaveform import fourier
import xarray as xr
from array_api_strict._typing import Array
from array_api_compat import is_cupy_array, is_numpy_array, is_torch_array, array_namespace
from dataclasses import dataclass
from collections import UserDict


@lru_cache
def equivalent_noise_bandwidth(window: str|tuple[str,float], N):
    """return the equivalent noise bandwidth (ENBW) of a window, in bins"""
    w = iqwaveform.fourier._get_window(window, N)
    return len(w) * np.sum(w**2) / np.sum(w) ** 2


@lru_cache(8)
def generate_iir_lpf(
    passband_ripple_dB: float | int,
    stopband_attenuation_dB: float | int,
    cutoff_Hz: float | int,
    transition_bandwidth_Hz: float | int,
    sample_rate_Hz: float | int,
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
        sample_rate_Hz:
            Sampling rate, in Hz.

    Returns:
        Second-order sections (sos) representation of the IIR filter.
    """

    order, wn = signal.ellipord(
        cutoff_Hz,
        cutoff_Hz + transition_bandwidth_Hz,
        passband_ripple_dB,
        stopband_attenuation_dB,
        False,
        sample_rate_Hz,
    )
    
    sos = signal.ellip(
        order,
        passband_ripple_dB,
        stopband_attenuation_dB,
        wn,
        'lowpass',
        False,
        'sos',
        sample_rate_Hz,
    )

    return sos


def _to_maybe_nested_numpy(obj: tuple|list|dict|Array):
    """convert an array, or a container of arrays, into a numpy array (or container of numpy arrays)"""

    ret = []

    if isinstance(obj, (tuple, list)):
        return [_to_maybe_nested_numpy(item) for item in obj]
    elif isinstance(obj, dict):
        return [_to_maybe_nested_numpy(item) for item in obj.values()]
    elif is_torch_array(obj):
        return obj.cpu()
    elif is_cupy_array(obj):
        return obj.get()
    elif is_numpy_array(obj):
        return obj
    else:
        raise TypeError(f'obj type {type(obj)} is unrecognized')


@dataclass
class ChannelAnalysisResult(UserDict):
    """represents the return result from a channel analysis function.

    A method for delayed conversion to `xarray.DataArray` allows cupy to initiate
    the evaluation of multiple concurrent analyses before materializing
    them on the CPU.
    """

    data: Array
    name: str
    coords: xr.Coordinates
    attrs: list[str]

    def to_xarray(self):
        return xr.DataArray(
            _to_maybe_nested_numpy(self.data),
            coords=xr.Coordinates(self.coords),
            dims=self.coords.dims,
            name=self.name,
            attrs=self.attrs,
        )


@lru_cache
def _power_time_series_coords(detectors: tuple[str], detector_period: float, length: int):
    time = xr.DataArray(
        np.arange(length) * detector_period,
        dims='time_elapsed',
        attrs={'label': 'Acquisition time elapsed', 'units': 's'},
    )

    detector_list = xr.DataArray(
        list(detectors), dims='power_detector', attrs={'label': 'Power detector'}
    )

    return xr.Coordinates({
        detector_list.dims[0]: detector_list,
        time.dims[0]: time,
    })


def power_time_series(
    iq,
    *,
    sample_rate_Hz: float,
    analysis_bandwidth_Hz: float,
    detector_period: float,
    detectors=('rms', 'peak'),
) -> callable[[],xr.DataArray]:


    Ts = 1/sample_rate_Hz
    data = [
        iqwaveform.powtodB(iqwaveform.iq_to_bin_power(iq, Ts=Ts, Tbin=detector_period, kind=detector))
        for detector in detectors
    ]

    coords = _power_time_series_coords(detectors, detector_period, len(data[0]))

    metadata = {
        'detector_period': detector_period,
        'label': 'Channel power',
        'units': f'dBm/{analysis_bandwidth_Hz/1e6} MHz',
    }

    return ChannelAnalysisResult(
        data=data, name='power_time_series', coords=coords, attrs=metadata
    )


@lru_cache
def _cyclic_channel_power_cyclic_coords(
    sample_rate_Hz: float,
    cyclic_period: float,
    detector_period: float,
    detectors: list[str],
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

    detector_coords = _power_time_series_coords(detectors, detector_period=detector_period, length=1)['power_detector']

    return xr.Coordinates({
        'power_detector': detector_coords,
        cyclic_statistic.dims[0]: cyclic_statistic,
        cyclic_lag.dims[0]: cyclic_lag,
    })


def cyclic_channel_power(
    iq,
    sample_rate_Hz,
    analysis_bandwidth_Hz,
    cyclic_period: float,
    detector_period: float,
    detectors: list[str] = ('rms', 'peak'),
    cyclic_statistics: list[str] = ('min', 'mean', 'max'),
) -> callable[[],xr.DataArray]:
    metadata = {}

    detectors = tuple(detectors)
    cyclic_statistics = tuple(cyclic_statistics)

    data_dict = iq_to_cyclic_power(
        iq,
        1 / sample_rate_Hz,
        cyclic_period=cyclic_period,
        detector_period=detector_period,
        detectors=detectors,
        cycle_stats=cyclic_statistics,
    )

    coords = _cyclic_channel_power_cyclic_coords(
        sample_rate_Hz,
        cyclic_period=cyclic_period,
        detector_period=detector_period,
        detectors=detectors,
        cyclic_statistics=cyclic_statistics,
    )

    metadata = {
        'label': 'Channel power',
        'units': f'dBm/{analysis_bandwidth_Hz/1e6} MHz',
        'cyclic_period': cyclic_period,
        'detector_period': detector_period        
    }

    return ChannelAnalysisResult(
        data=data_dict, name='cyclic_channel_power', coords=coords, attrs=metadata
    )


@lru_cache
def _get_apd_bins(lo, hi, count, xp=np):
    return xp.linspace(lo, hi, count)


@lru_cache
def _label_apd_power_bins(lo, hi, count, xp, units):
    params = dict(locals())
    del params['units']

    bins = _get_apd_bins(**params)
    array = xr.DataArray(
        bins, dims='channel_power', attrs={'label': 'Channel power', 'units': units}
    )
    return xr.Coordinates({array.dims[0]: array})


def amplitude_probability_distribution(
    iq,
    *,
    sample_rate_Hz: float = None,
    analysis_bandwidth_Hz: float,
    power_low: float,
    power_high: float,
    power_count: float,
) -> callable[[],xr.DataArray]:
    xp = array_namespace(iq)

    bin_params = {'lo': power_low, 'hi': power_high, 'count': power_count}

    bins = _get_apd_bins(xp=xp, **bin_params)
    coords = _label_apd_power_bins(xp=np, units=f'dBm/{analysis_bandwidth_Hz/1e6} MHz', **bin_params)

    ccdf = iqwaveform.sample_ccdf(iqwaveform.envtodB(iq), bins)

    return ChannelAnalysisResult(
        data=ccdf, name='amplitude_probability_distribution', coords=coords, attrs=bin_params
    )


@lru_cache
def _persistence_spectrum_coords(
    sample_rate_Hz: float, analysis_bandwidth_Hz: float, fft_size: int, time_size: int,
    stat_names: tuple[str], overlap_frac: float = 0, truncate: bool = True, xp=np
):
    freqs, times = iqwaveform.fourier._get_stft_axes(
        fs=sample_rate_Hz,
        fft_size=fft_size,
        time_size=time_size,
        overlap_frac=overlap_frac,
        xp=np,
    )

    if truncate:
        which_freqs = np.abs(freqs) <= analysis_bandwidth_Hz/2
        freqs = freqs[which_freqs]

    freqs = xr.DataArray(
        freqs,
        dims='baseband_frequency',
        attrs={'label': 'Baseband frequency', 'units': 'Hz'},
    )

    stats = xr.DataArray(
        [str(n) for n in stat_names],
        dims='persistence_statistic',
        attrs={'label': 'Persistence statistic'},
    )

    return xr.Coordinates({stats.dims[0]: stats, freqs.dims[0]: freqs})


def persistence_spectrum(
    iq: Array,
    *,
    sample_rate_Hz: float,
    analysis_bandwidth_Hz=None,
    window,
    resolution: float,
    fractional_overlap=0,
    quantiles: list[float],
    truncate = True,
    dB = True,
) -> callable[[],xr.DataArray]:
    # TODO: support other persistence statistics, such as mean

    if not iqwaveform.power_analysis.isroundmod(sample_rate_Hz, resolution):
        # need sample_rate_Hz/resolution to give us a counting number
        raise ValueError('sample_rate_Hz/resolution must be a counting number')

    fft_size = int(sample_rate_Hz / resolution)
    enbw = resolution * equivalent_noise_bandwidth(window, fft_size)

    metadata = {
        'window': window,
        'resolution_Hz': resolution,
        'fractional_overlap': fractional_overlap,
        'noise_bandwidth_Hz': enbw,
        'fft_size': fft_size,
        'truncate': truncate,
        'label': 'Power spectral density',
        'units': f'dBm/{enbw/1e3:0.3f} kHz'
    }

    xp = array_namespace(iq)

    noverlap=int(np.rint(fractional_overlap*fft_size))
    freqs, times, spg = iqwaveform.fourier.spectrogram(
        iq, window=window, fs=sample_rate_Hz, nperseg=fft_size, noverlap=noverlap
    )

    if truncate:
        which_freqs = np.abs(freqs) <= analysis_bandwidth_Hz/2
        spg = spg[:,which_freqs]

    # if is_cupy_array(spg):
    #     import cupy
    #     cupy.cuda.runtime.deviceSynchronize()
    #     spg = xp.ascontiguousarray(spg)

    if dB:
        spg = iqwaveform.powtodB(spg, eps=1e-25)

    data = xp.quantile(spg, xp.asarray(quantiles, dtype=xp.float32), axis=0)

    coords = _persistence_spectrum_coords(
        sample_rate_Hz=sample_rate_Hz,
        analysis_bandwidth_Hz=analysis_bandwidth_Hz,
        fft_size=fft_size,
        time_size=spg.shape[0],
        overlap_frac=fractional_overlap,
        stat_names=tuple(quantiles),
        truncate=truncate
    )

    return ChannelAnalysisResult(
        data=data, name='persistence_spectrum', coords=coords, attrs=metadata
    )


def from_spec(
    iq,
    sample_rate_Hz,
    analysis_bandwidth_Hz,
    *,
    filter_spec: dict = {
        'passband_ripple_dB': 0.1,
        'stopband_attenuation_dB': 70,
        'transition_bandwidth_Hz': 250e3,
    },
    analysis_spec: dict[str, dict[str]] = {},
    overwrite_iq = False
):
    xp = array_namespace(iq)

    acq_kws = {
        'iq': iq,
        'sample_rate_Hz': sample_rate_Hz,
        'analysis_bandwidth_Hz': analysis_bandwidth_Hz,
    }

    filter_spec = dict(filter_spec)

    kws = filter_spec.pop('ola', None)
    if kws is not None:
        iq = fourier.ola_filter(
            iq,
            fs=sample_rate_Hz,
            passband=(-analysis_bandwidth_Hz/2, analysis_bandwidth_Hz/2),
            axis=0,
            **kws
        )

        spg = None

    kws = filter_spec.pop('iir', None)
    if kws is not None:
        sos = generate_iir_lpf(
            cutoff_Hz=analysis_bandwidth_Hz / 2,
            sample_rate_Hz=sample_rate_Hz,
            **kws,
        )

        if is_cupy_array(iq):
            from . import cuda_filter
            sos = xp.asarray(sos)
            iq = cuda_filter.sosfilt(sos.astype('float32'), iq)

        else:
            iq = signal.sosfilt(sos.astype('float32'), iq)

        spg = None

    if len(filter_spec) > 0:
        raise ValueError(f'unrecognized filter specification keys: {list(filter_spec.keys())}')

    if is_cupy_array(iq):
        # sync all operations in order to prevent copies of previous operations being
        # evaluated in parallel
        import cupy
        cupy.cuda.runtime.deviceSynchronize()

    analysis_spec = dict(analysis_spec)
    results = []

    func_kws = analysis_spec.pop('power_time_series', None)
    if func_kws is not None:
        results += [power_time_series(**acq_kws, **func_kws)]

    func_kws = analysis_spec.pop('persistence_spectrum', None)
    if func_kws is not None:
        results += [persistence_spectrum(**acq_kws, **func_kws)]

    func_kws = analysis_spec.pop('amplitude_probability_distribution', None)
    if func_kws is not None:
        results += [amplitude_probability_distribution(**acq_kws, **func_kws)]

    func_kws = analysis_spec.pop('cyclic_channel_power', None)
    if func_kws is not None:
        results += [cyclic_channel_power(**acq_kws, **func_kws)]

    if len(analysis_spec) > 0:
        raise ValueError(f'invalid analysis_spec key(s): {list(analysis_spec.keys())}')

    if is_cupy_array(iq):
        # sync all operations in order to prevent copies of previous operations being
        # evaluated in parallel
        import cupy
        cupy.cuda.runtime.deviceSynchronize()

    # materialize the xarrays on the cpu
    xarrays = {
        res.name: res.to_xarray()
        for res in results
    }

    metadata = {
        'sample_rate_Hz': sample_rate_Hz,
        'analysis_bandwidth_Hz': analysis_bandwidth_Hz,
        'filter_specification': filter_spec,
    }

    return xr.Dataset(xarrays, attrs=metadata)
