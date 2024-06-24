"""Wrap iqwaveform to perform DSP evaluation of baseband waveforms"""

from __future__ import annotations
import iqwaveform
from functools import lru_cache
import numpy as np
from scipy import signal
from iqwaveform.power_analysis import iq_to_cyclic_power
from iqwaveform.util import array_namespace, array_stream, set_input_domain
from iqwaveform import fourier
import xarray as xr
import pandas as pd
from array_api_strict._typing import Array
from array_api_compat import is_cupy_array, is_numpy_array, is_torch_array
from dataclasses import dataclass
from collections import UserDict
from inspect import signature
from . import metadata

@lru_cache
def equivalent_noise_bandwidth(window: str | tuple[str, float], N):
    """return the equivalent noise bandwidth (ENBW) of a window, in bins"""
    w = iqwaveform.fourier._get_window(window, N)
    return len(w) * np.sum(w**2) / np.sum(w) ** 2


@lru_cache(8)
def _generate_iir_lpf(
    passband_ripple_dB: float | int,
    stopband_attenuation_dB: float | int,
    bandwidth_Hz: float | int,
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
        bandwidth_Hz:
            Filter cutoff frequency, in Hz.
        transition_bandwidth_Hz:
            Passband-to-stopband transition width, in Hz.
        sample_rate_Hz:
            Sampling rate, in Hz.

    Returns:
        Second-order sections (sos) representation of the IIR filter.
    """

    order, wn = signal.ellipord(
        bandwidth_Hz,
        bandwidth_Hz + transition_bandwidth_Hz,
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


def _sync_if_cuda(obj: Array):
    if is_cupy_array(obj):
        import cupy
        cupy.cuda.Stream.null.synchronize()


def _to_maybe_nested_numpy(obj: tuple | list | dict | Array):
    """convert an array, or a container of arrays, into a numpy array (or container of numpy arrays)"""

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
def _power_time_series_coords(
    detectors: tuple[str], detector_period: float, length: int
):
    time = xr.DataArray(
        np.arange(length) * detector_period,
        dims='time_elapsed',
        attrs={'label': 'Acquisition time elapsed', 'units': 's'},
    )

    detector_list = xr.DataArray(
        list(detectors), dims='power_detector', attrs={'label': 'Power detector'}
    )

    return xr.Coordinates(
        {
            detector_list.dims[0]: detector_list,
            time.dims[0]: time,
        }
    )


def power_time_series(
    iq,
    *,
    sample_rate_Hz: float,
    analysis_bandwidth_Hz: float,
    detector_period: float,
    detectors=('rms', 'peak')
) -> callable[[], xr.DataArray]:
    Ts = 1 / sample_rate_Hz

    xp = array_namespace(iq)
    dtype = xp.finfo(iq.dtype).dtype

    data = [
        iqwaveform.powtodB(
            iqwaveform.iq_to_bin_power(iq, Ts=Ts, Tbin=detector_period, kind=detector)
        ).astype(dtype)
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

    detector_coords = _power_time_series_coords(
        detectors, detector_period=detector_period, length=1
    )['power_detector']

    return xr.Coordinates(
        {
            'power_detector': detector_coords,
            cyclic_statistic.dims[0]: cyclic_statistic,
            cyclic_lag.dims[0]: cyclic_lag,
        }
    )


def cyclic_channel_power(
    iq,
    sample_rate_Hz,
    analysis_bandwidth_Hz,
    cyclic_period: float,
    detector_period: float,
    detectors: list[str] = ('rms', 'peak'),
    cyclic_statistics: list[str] = ('min', 'mean', 'max'),
) -> callable[[], xr.DataArray]:
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
        'detector_period': detector_period,
    }

    return ChannelAnalysisResult(
        data=data_dict, name='cyclic_channel_power', coords=coords, attrs=metadata
    )


@lru_cache
def _bin_apd(lo, hi, count, xp=np):
    return xp.linspace(lo, hi, count)


@lru_cache
def _amplitude_probability_distribution_coords(lo, hi, count, xp, units):
    params = dict(locals())
    del params['units']

    bins = _bin_apd(**params).astype(np.float32)
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
) -> callable[[], xr.DataArray]:
    xp = array_namespace(iq)
    dtype = xp.finfo(iq.dtype).dtype

    bin_params = {'lo': power_low, 'hi': power_high, 'count': power_count}

    bins = _bin_apd(xp=xp, **bin_params)
    coords = _amplitude_probability_distribution_coords(
        xp=np, units=f'dBm/{analysis_bandwidth_Hz/1e6} MHz', **bin_params
    )

    ccdf = iqwaveform.sample_ccdf(iqwaveform.envtodB(iq), bins).astype(dtype)

    return ChannelAnalysisResult(
        data=ccdf,
        name='amplitude_probability_distribution',
        coords=coords,
        attrs={'label': 'CCDF', **bin_params},
    )


@lru_cache
def _persistence_spectrum_coords(
    sample_rate_Hz: float,
    analysis_bandwidth_Hz: float,
    fft_size: int,
    stat_names: tuple[str],
    overlap_frac: float = 0,
    truncate: bool = True,
    xp=np,
):
    freqs, _ = iqwaveform.fourier._get_stft_axes(
        fs=sample_rate_Hz,
        fft_size=fft_size,
        time_size=1,
        overlap_frac=overlap_frac,
        xp=np,
    )

    if truncate:
        which_freqs = np.abs(freqs) <= analysis_bandwidth_Hz / 2
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
    x: Array,
    *,
    sample_rate_Hz: float,
    analysis_bandwidth_Hz=None,
    window,
    resolution: float,
    fractional_overlap=0,
    quantiles: list[float],
    truncate=True,
    dB=True,
) -> callable[[], xr.DataArray]:

    # TODO: support other persistence statistics, such as mean
    if iqwaveform.power_analysis.isroundmod(sample_rate_Hz, resolution):
        # need sample_rate_Hz/resolution to give us a counting number
        fft_size = round(sample_rate_Hz / resolution)
    else:
        raise ValueError('sample_rate_Hz/resolution must be a counting number')

    enbw = resolution * equivalent_noise_bandwidth(window, fft_size)

    metadata = {
        'window': window,
        'resolution_Hz': resolution,
        'fractional_overlap': fractional_overlap,
        'noise_bandwidth_Hz': enbw,
        'fft_size': fft_size,
        'truncate': truncate,
        'label': 'Power spectral density',
        'units': f'dBm/{enbw/1e3:0.3f} kHz',
    }

    data = fourier.persistence_spectrum(
        x,
        fs=sample_rate_Hz,
        bandwidth=analysis_bandwidth_Hz,
        window=window,
        resolution=resolution,
        fractional_overlap=fractional_overlap,
        quantiles=quantiles,
        truncate=True,
        dB=dB,
    )

    coords = _persistence_spectrum_coords(
        sample_rate_Hz=sample_rate_Hz,
        analysis_bandwidth_Hz=analysis_bandwidth_Hz,
        fft_size=fft_size,
        overlap_frac=fractional_overlap,
        stat_names=tuple(quantiles),
        truncate=truncate,
    )

    return ChannelAnalysisResult(
        data=data, name='persistence_spectrum', coords=coords, attrs=metadata
    )


def iq_waveform(
    iq,
    *,
    sample_rate_Hz: float,
    analysis_bandwidth_Hz=None,
    start_time_sec=None,
    stop_time_sec=None
) -> callable[[], xr.DataArray]:
    """package the IQ recording with optional clipping"""

    metadata = {
        'label': 'IQ waveform',
        'units': 'V',
        'start_time_sec': start_time_sec,
        'stop_time_sec': stop_time_sec
    }

    if start_time_sec is None:
        start = None
    else:
        start = int(start_time_sec*sample_rate_Hz)

    if stop_time_sec is None:
        stop = None
    else:
        stop = int(stop_time_sec*sample_rate_Hz)

    coords = xr.Coordinates({'iq_sample': pd.RangeIndex(start, stop, name='iq_sample')})

    return ChannelAnalysisResult(
        data=iq[start:stop],
        name='iq_waveform',
        coords=coords,
        attrs=metadata,
    )


def iir_filter(
    iq: Array,
    *,
    sample_rate_Hz: float,
    bandwidth_Hz: float | int,
    passband_ripple_dB: float | int,
    stopband_attenuation_dB: float | int,
    transition_bandwidth_Hz: float | int,
    out=None
):
    filter_kws = dict(locals())
    for name in ('iq', 'bandwidth_Hz', 'out'):
        del filter_kws[name]

    sos = _generate_iir_lpf(
        # scipy filter design assumes real-valued waveforms. for complex,
        # account for this by halving the bandwidth
        bandwidth_Hz=bandwidth_Hz/2, 
        **filter_kws
    )

    if is_cupy_array(iq):
        from . import cuda_filter

        sos = xp.asarray(sos)
        return cuda_filter.sosfilt(sos.astype('float32'), iq)

    else:
        return signal.sosfilt(sos.astype('float32'), iq)


def ola_filter(
    iq: Array,
    *,
    sample_rate_Hz: float,
    bandwidth_Hz: float,
    fft_size: int,
    window: str|tuple = 'hamming',
    out=None,
    cache=None
):
    return fourier.ola_filter(
        iq,
        fs=sample_rate_Hz,
        passband=(-bandwidth_Hz / 2, bandwidth_Hz / 2),
        fft_size=fft_size,
        window=window,
        out=out,
        cache=cache
    )


def _compatible_filter_and_spectrum(sample_rate_Hz, filter_spec, persistence_kws):
    filter_spec = dict(
        bandwidth_Hz=None, sample_rate_Hz=sample_rate_Hz, **filter_spec
    )
    persistence_kws = dict(
        sample_rate_Hz=sample_rate_Hz, analysis_bandwidth_Hz=None, **persistence_kws
    )

    sig1 = signature(ola_filter).bind(None, **filter_spec).arguments
    sig2 = signature(persistence_spectrum).bind(None, **persistence_kws).arguments
    sig2['fft_size'] = round(sample_rate_Hz / persistence_kws['resolution'])

    for arg in ('fft_size', 'window'):
        if sig1[arg] != sig2[arg]:
            reuse_ola_stft = False
            break
    else:
        reuse_ola_stft = True

    return reuse_ola_stft


def from_spec(
    iq,
    sample_rate_Hz,
    analysis_bandwidth_Hz,
    *,
    filter_spec: dict = {
        'fft_size': 1024,
        'window': 'hamming',  # 'hamming', 'blackman', or 'blackmanharris'
    },
    analysis_spec: dict[str, dict[str]] = {}
):
    xp = array_namespace(iq)

    acq_kws = {
        'sample_rate_Hz': sample_rate_Hz,
        'analysis_bandwidth_Hz': analysis_bandwidth_Hz,
    }

    iq_in = iq
    filter_metadata = filter_spec
    filter_spec = dict(filter_spec)
    analysis_spec = dict(analysis_spec)

    # first: everything that doesn't need the filter output
    # filter_stream = array_stream(iq)
    # spectrum_stream = array_stream(iq)

    cache = {}

    if filter_spec is not None and 'persistence_spectrum' in analysis_spec:
        reuse_ola_stft = _compatible_filter_and_spectrum(sample_rate_Hz, filter_spec, analysis_spec['persistence_spectrum'])
    else:
        reuse_ola_stft = False

    iq = ola_filter(
        iq, bandwidth_Hz=analysis_bandwidth_Hz, sample_rate_Hz=sample_rate_Hz,
        cache=cache if reuse_ola_stft else None,
        **filter_spec
    )

    _sync_if_cuda(iq)

    # then: analyses that need filtered output
    results = {}

    for func in (persistence_spectrum, power_time_series, cyclic_channel_power, amplitude_probability_distribution, iq_waveform):
        # check for each allowed function in the specification
        try:
            func_kws = analysis_spec.pop(func.__name__)
        except KeyError:
            pass
        else:
            if func is persistence_spectrum and 'stft' in cache:
                x = cache['stft'][::2]
                domain = 'frequency'
            else:
                x = iq
                domain = 'time'

            with set_input_domain(domain):
                results[func.__name__] = func(x, **acq_kws, **func_kws)

    if len(analysis_spec) > 0:
        # anything left refers to an invalid function invalid
        raise ValueError(f'invalid analysis_spec key(s): {list(analysis_spec.keys())}')

    _sync_if_cuda(iq)

    # materialize as xarrays on the cpu
    xarrays = {res.name: res.to_xarray() for res in results.values()}
    xarrays.update(metadata.build_diagnostic_data())

    attrs = {
        'sample_rate_Hz': sample_rate_Hz,
        'analysis_bandwidth_Hz': analysis_bandwidth_Hz,
        'filter': filter_metadata or [],
        **metadata.build_metadata(),
    }

    return xr.Dataset(xarrays, attrs=attrs)