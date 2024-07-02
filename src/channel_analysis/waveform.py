"""wrap lower-level iqwaveform DSP calls to accept physical inputs and return xarray.DataArray"""

from __future__ import annotations
import iqwaveform
from functools import lru_cache
import numpy as np
from iqwaveform.util import Array, array_namespace
from iqwaveform import fourier, power_analysis

from scipy import signal
import xarray as xr
import pandas as pd

from array_api_compat import is_cupy_array, is_numpy_array, is_torch_array
from dataclasses import dataclass
from collections import UserDict
from .sources import WaveformSource
from . import config

import typing
import msgspec

TDecoratedFunc = typing.Callable[..., typing.Any]


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
def equivalent_noise_bandwidth(window: str | tuple[str, float], N):
    """return the equivalent noise bandwidth (ENBW) of a window, in bins"""
    w = iqwaveform.fourier._get_window(window, N)
    return len(w) * np.sum(w**2) / np.sum(w) ** 2


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


def _analysis_parameter_kwargs(locals_: dict, omit=('iq', 'source', 'out')) -> dict:
    """return the analysis parameters from the locals() evaluated at the beginning of analysis function"""

    return {k: v for k, v in locals_.items() if k not in omit}


@lru_cache
def _power_time_series_coords(
    detectors: tuple[str], detector_period: float, length: int
):
    time = xr.DataArray(
        np.arange(length) * detector_period,
        dims='time_elapsed',
        attrs={'label': 'System time elapsed', 'units': 's'},
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


@config.registry.include
def power_time_series(
    iq,
    source: WaveformSource,
    *,
    detector_period: float,
    detectors: tuple[str, ...] = ('rms', 'peak'),
) -> callable[[], xr.DataArray]:
    Ts = 1 / source.sample_rate

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
        'units': f'dBm/{source.analysis_bandwidth/1e6} MHz',
    }

    return ChannelAnalysisResult(
        data=data, name='power_time_series', coords=coords, attrs=metadata
    )


@lru_cache
def _cyclic_channel_power_cyclic_coords(
    source: WaveformSource,
    cyclic_period: float,
    detector_period: float,
    detectors: tuple[str, ...],
    cyclic_statistics: tuple[str, ...],
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


@config.registry.include
def cyclic_channel_power(
    iq,
    source: WaveformSource,
    *,
    cyclic_period: float,
    detector_period: float,
    detectors: tuple[str, ...] = ('rms', 'peak'),
    cyclic_statistics: tuple[str, ...] = ('min', 'mean', 'max'),
) -> callable[[], xr.DataArray]:
    metadata = {}

    detectors = tuple(detectors)
    cyclic_statistics = tuple(cyclic_statistics)

    data_dict = power_analysis.iq_to_cyclic_power(
        iq,
        1 / source.sample_rate,
        cyclic_period=cyclic_period,
        detector_period=detector_period,
        detectors=detectors,
        cycle_stats=cyclic_statistics,
    )

    coords = _cyclic_channel_power_cyclic_coords(
        source.sample_rate,
        cyclic_period=cyclic_period,
        detector_period=detector_period,
        detectors=detectors,
        cyclic_statistics=cyclic_statistics,
    )

    metadata = {
        'label': 'Channel power',
        'units': f'dBm/{source.analysis_bandwidth/1e6} MHz',
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
    params = _analysis_parameter_kwargs(locals(), omit=('units',))

    bins = _bin_apd(**params).astype(np.float32)
    array = xr.DataArray(
        bins, dims='channel_power', attrs={'label': 'Channel power', 'units': units}
    )
    return xr.Coordinates({array.dims[0]: array})


@config.registry.include
def amplitude_probability_distribution(
    iq,
    source: WaveformSource,
    *,
    power_low: float,
    power_high: float,
    power_count: int,
) -> callable[[], xr.DataArray]:
    xp = array_namespace(iq)
    dtype = xp.finfo(iq.dtype).dtype

    bin_params = {'lo': power_low, 'hi': power_high, 'count': power_count}

    bins = _bin_apd(xp=xp, **bin_params)
    coords = _amplitude_probability_distribution_coords(
        xp=np, units=f'dBm/{source.analysis_bandwidth/1e6} MHz', **bin_params
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
    source: WaveformSource,
    fft_size: int,
    stat_names: tuple[str],
    overlap_frac: float = 0,
    truncate: bool = True,
    xp=np,
):
    freqs, _ = iqwaveform.fourier._get_stft_axes(
        fs=source.sample_rate,
        fft_size=fft_size,
        time_size=1,
        overlap_frac=overlap_frac,
        xp=np,
    )

    if truncate:
        which_freqs = np.abs(freqs) <= source.analysis_bandwidth / 2
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


@config.registry.include
def persistence_spectrum(
    x: Array,
    source: WaveformSource,
    *,
    window: typing.Any,
    resolution: float,
    quantiles: tuple[float, ...],
    fractional_overlap: float = 0,
    truncate: bool = True,
    dB: bool = True,
) -> callable[[], xr.DataArray]:
    # TODO: support other persistence statistics, such as mean
    if iqwaveform.power_analysis.isroundmod(source.sample_rate, resolution):
        # need source.sample_rate/resolution to give us a counting number
        fft_size = round(source.sample_rate / resolution)
    else:
        raise ValueError('sample_rate/resolution must be a counting number')

    enbw = resolution * equivalent_noise_bandwidth(window, fft_size)

    metadata = {
        'window': window,
        'resolution': resolution,
        'fractional_overlap': fractional_overlap,
        'noise_bandwidth': enbw,
        'fft_size': fft_size,
        'truncate': truncate,
        'label': 'Power spectral density',
        'units': f'dBm/{enbw/1e3:0.3f} kHz',
    }

    data = fourier.persistence_spectrum(
        x,
        fs=source.sample_rate,
        bandwidth=source.analysis_bandwidth,
        window=window,
        resolution=resolution,
        fractional_overlap=fractional_overlap,
        quantiles=quantiles,
        truncate=True,
        dB=dB,
    )

    coords = _persistence_spectrum_coords(
        source,
        fft_size=fft_size,
        overlap_frac=fractional_overlap,
        stat_names=tuple(quantiles),
        truncate=truncate,
    )

    return ChannelAnalysisResult(
        data=data, name='persistence_spectrum', coords=coords, attrs=metadata
    )


@lru_cache(8)
def _generate_iir_lpf(
    source: WaveformSource,
    *,
    passband_ripple: float | int,
    stopband_attenuation: float | int,
    transition_bandwidth: float | int,
) -> np.ndarray:
    """
    Generate an elliptic IIR low pass filter for complex-valued waveforms.

    Args:
        passband_ripple:
            Maximum amplitude ripple in the passband of the frequency-response below unity gain (dB)
        stopband_attenuation:
            Maximum amplitude ripple in the passband of the frequency-response below unity gain (dB).
        transition_bandwidth:
            Passband-to-stopband transition width (Hz)

    Returns:
        Second-order sections (sos) representation of the IIR filter.
    """

    order, wn = signal.ellipord(
        source.analysis_bandwidth / 2,
        source.analysis_bandwidth / 2 + transition_bandwidth,
        passband_ripple,
        stopband_attenuation,
        False,
        source.sample_rate,
    )

    sos = signal.ellip(
        order,
        passband_ripple,
        stopband_attenuation,
        wn,
        'lowpass',
        False,
        'sos',
        source.sample_rate,
    )

    return sos


@config.registry.include
def iq_waveform(
    iq,
    source: WaveformSource,
    *,
    start_time_sec: typing.Optional[float] = None,
    stop_time_sec: typing.Optional[float] = None,
) -> callable[[], xr.DataArray]:
    """package the IQ recording with optional clipping"""

    metadata = {
        'label': 'IQ waveform',
        'units': 'V',
        'start_time_sec': start_time_sec,
        'stop_time_sec': stop_time_sec,
    }

    if start_time_sec is None:
        start = None
    else:
        start = int(start_time_sec * source.sample_rate)

    if stop_time_sec is None:
        stop = None
    else:
        stop = int(stop_time_sec * source.sample_rate)

    coords = xr.Coordinates({'iq_sample': pd.RangeIndex(start, stop, name='iq_sample')})

    return ChannelAnalysisResult(
        data=iq[start:stop].copy(),
        name='iq_waveform',
        coords=coords,
        attrs=metadata,
    )


def iir_filter(
    iq: Array,
    source: WaveformSource,
    *,
    passband_ripple: float | int,
    stopband_attenuation: float | int,
    transition_bandwidth: float | int,
    out=None,
):
    xp = array_namespace(iq)

    filter_kws = _analysis_parameter_kwargs(locals())

    sos = _generate_iir_lpf(source, **filter_kws)

    if is_cupy_array(iq):
        from . import cuda_filter

        sos = xp.asarray(sos)
        return cuda_filter.sosfilt(sos.astype('float32'), iq)

    else:
        return signal.sosfilt(sos.astype('float32'), iq)


def ola_filter(
    iq: Array,
    source: WaveformSource,
    *,
    fft_size: int,
    window: typing.Any = 'hamming',
    out=None,
    cache=None,
):
    kwargs = _analysis_parameter_kwargs(locals())

    return fourier.ola_filter(
        iq,
        fs=source.sample_rate,
        passband=(-source.analysis_bandwidth / 2, source.analysis_bandwidth / 2),
        **kwargs,
    )


def from_spec(
    iq,
    source: WaveformSource,
    *,
    analysis_spec: str | dict | config.AnalysisStruct = {},
    cache={},
):
    analysis_spec = dict(analysis_spec)

    # then: analyses that need filtered output
    results = {}

    schema = config.registry.tostruct()
    analysis_spec = msgspec.to_builtins(msgspec.convert(analysis_spec, schema))

    # evaluate each possible analysis function if specified
    for func in config.registry.keys():
        func_kws = analysis_spec.pop(func.__name__)

        if func_kws:
            results[func.__name__] = func(iq, source, **func_kws)

    if len(analysis_spec) > 0:
        # anything left refers to an invalid function
        raise ValueError(f'invalid analysis_spec key(s): {list(analysis_spec.keys())}')

    # materialize as xarrays on the cpu
    xarrays = {res.name: res.to_xarray() for res in results.values()}
    # xarrays.update(metadata.build_index_variables())

    return xr.Dataset(xarrays, attrs=source.build_metadata())
