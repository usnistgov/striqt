"""wrap lower-level iqwaveform DSP calls to accept physical inputs and return xarray.DataArray"""

from __future__ import annotations

from dataclasses import dataclass
from collections import UserDict
from functools import lru_cache

import numpy as np
from scipy import signal
import xarray as xr
import pandas as pd
import iqwaveform
from iqwaveform.util import Array, array_namespace
from iqwaveform import fourier, power_analysis
from frozendict import frozendict

from array_api_compat import is_cupy_array, is_numpy_array, is_torch_array
from . import structs

import typing
import msgspec

TDecoratedFunc = typing.Callable[..., typing.Any]

registry = structs.KeywordConfigRegistry(structs.ChannelAnalysis)

IQ_WAVEFORM_INDEX_NAME = 'iq_index'


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


def _analysis_parameter_kwargs(locals_: dict, omit=('iq', 'capture', 'out')) -> dict:
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


@registry
def power_time_series(
    iq,
    capture: structs.Capture,
    *,
    detector_period: float,
    detectors: tuple[str, ...] = ('rms', 'peak'),
) -> callable[[], xr.DataArray]:
    Ts = 1 / capture.sample_rate

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
        'units': f'dBm/{(capture.analysis_bandwidth or capture.sample_rate)/1e6} MHz',
    }

    return ChannelAnalysisResult(
        data=data, name='power_time_series', coords=coords, attrs=metadata
    )


@lru_cache
def _cyclic_channel_power_cyclic_coords(
    capture: structs.Capture,
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


@registry
def cyclic_channel_power(
    iq,
    capture: structs.Capture,
    *,
    cyclic_period: float,
    detector_period: float,
    detectors: tuple[str, ...] = ('rms', 'peak'),
    cyclic_statistics: tuple[str, ...] = ('min', 'mean', 'max'),
) -> callable[[], xr.DataArray]:
    metadata = {}

    detectors = tuple(detectors)
    cyclic_statistics = tuple(cyclic_statistics)

    print(locals())

    data_dict = power_analysis.iq_to_cyclic_power(
        iq,
        1 / capture.sample_rate,
        cyclic_period=cyclic_period,
        detector_period=detector_period,
        detectors=detectors,
        cycle_stats=cyclic_statistics,
    )

    coords = _cyclic_channel_power_cyclic_coords(
        capture.sample_rate,
        cyclic_period=cyclic_period,
        detector_period=detector_period,
        detectors=detectors,
        cyclic_statistics=cyclic_statistics,
    )

    metadata = {
        'label': 'Channel power',
        'units': f'dBm/{(capture.analysis_bandwidth or capture.sample_rate)/1e6} MHz',
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


@registry
def amplitude_probability_distribution(
    iq,
    capture: structs.Capture,
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
        xp=np,
        units=f'dBm/{(capture.analysis_bandwidth or capture.sample_rate)/1e6} MHz',
        **bin_params,
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
    capture: structs.Capture,
    fft_size: int,
    stat_names: tuple[str],
    overlap_frac: float = 0,
    truncate: bool = True,
    xp=np,
):
    freqs, _ = iqwaveform.fourier._get_stft_axes(
        fs=capture.sample_rate,
        fft_size=fft_size,
        time_size=1,
        overlap_frac=overlap_frac,
        xp=np,
    )

    if truncate and capture.analysis_bandwidth is not None:
        which_freqs = np.abs(freqs) <= capture.analysis_bandwidth / 2
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


@registry
def persistence_spectrum(
    x: Array,
    capture: structs.Capture,
    *,
    window: typing.Any,
    resolution: float,
    statistics: tuple[typing.Union[str, float], ...],
    fractional_overlap: float = 0,
    truncate: bool = True,
    dB: bool = True,
) -> callable[[], xr.DataArray]:
    # TODO: support other persistence statistics, such as mean
    if iqwaveform.power_analysis.isroundmod(capture.sample_rate, resolution):
        # need capture.sample_rate/resolution to give us a counting number
        fft_size = round(capture.sample_rate / resolution)
    else:
        raise ValueError('sample_rate/resolution must be a counting number')

    if isinstance(window, list):
        # lists break lru_cache
        window = tuple(window)

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
        fs=capture.sample_rate,
        bandwidth=capture.analysis_bandwidth,
        window=window,
        resolution=resolution,
        fractional_overlap=fractional_overlap,
        statistics=statistics,
        truncate=truncate,
        dB=dB,
    )

    coords = _persistence_spectrum_coords(
        capture,
        fft_size=fft_size,
        overlap_frac=fractional_overlap,
        stat_names=tuple(statistics),
        truncate=truncate,
    )

    return ChannelAnalysisResult(
        data=data, name='persistence_spectrum', coords=coords, attrs=metadata
    )


@lru_cache(8)
def _generate_iir_lpf(
    capture: structs.Capture,
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
        capture.analysis_bandwidth / 2,
        capture.analysis_bandwidth / 2 + transition_bandwidth,
        passband_ripple,
        stopband_attenuation,
        False,
        capture.sample_rate,
    )

    sos = signal.ellip(
        order,
        passband_ripple,
        stopband_attenuation,
        wn,
        'lowpass',
        False,
        'sos',
        capture.sample_rate,
    )

    return sos


@registry
def iq_waveform(
    iq,
    capture: structs.Capture,
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
        start = int(start_time_sec * capture.sample_rate)

    if stop_time_sec is None:
        stop = None
    else:
        stop = int(stop_time_sec * capture.sample_rate)

    coords = xr.Coordinates(
        {
            IQ_WAVEFORM_INDEX_NAME: pd.RangeIndex(
                start, stop, name=IQ_WAVEFORM_INDEX_NAME
            )
        }
    )

    return ChannelAnalysisResult(
        data=iq[start:stop].copy(),
        name='iq_waveform',
        coords=coords,
        attrs=metadata,
    )


def iir_filter(
    iq: Array,
    capture: structs.Capture,
    *,
    passband_ripple: float | int,
    stopband_attenuation: float | int,
    transition_bandwidth: float | int,
    out=None,
):
    xp = array_namespace(iq)

    filter_kws = _analysis_parameter_kwargs(locals())

    sos = _generate_iir_lpf(capture, **filter_kws)

    if is_cupy_array(iq):
        from . import cuda_filter

        sos = xp.asarray(sos)
        return cuda_filter.sosfilt(sos.astype('float32'), iq)

    else:
        return signal.sosfilt(sos.astype('float32'), iq)


def ola_filter(
    iq: Array,
    capture: structs.Capture,
    *,
    fft_size: int,
    window: typing.Any = 'hamming',
    out=None,
    cache=None,
):
    kwargs = _analysis_parameter_kwargs(locals())

    return fourier.ola_filter(
        iq,
        fs=capture.sample_rate,
        passband=(-capture.analysis_bandwidth / 2, capture.analysis_bandwidth / 2),
        **kwargs,
    )


def to_analysis_spec(
    obj: str | dict | structs.ChannelAnalysis,
) -> structs.ChannelAnalysis:
    """coerces a yaml string or dictionary of dictionaries into channel analysis spec"""

    struct = registry.spec_type()

    if isinstance(obj, (dict, registry.base_struct)):
        return msgspec.convert(obj, struct)
    elif isinstance(obj, str):
        return msgspec.yaml.decode(obj, type=struct)
    else:
        return TypeError('unrecognized type')


def _evaluate_raw_channel_analysis(iq: Array, capture: structs.Capture, *, spec: str | dict | structs.ChannelAnalysis):
    # round-trip for type conversion and validation
    spec = msgspec.convert(spec, registry.spec_type())
    spec_dict = msgspec.to_builtins(spec)

    results = {}

    # evaluate each possible analysis function if specified
    for name, func_kws in spec_dict.items():
        func = registry[type(getattr(spec, name))]

        if func_kws:
            results[name] = func(iq, capture, **func_kws)

    return results

def _package_channel_analysis(capture: structs.Capture, results: dict[str, structs.ChannelAnalysis]):
    # materialize as xarrays
    xarrays = {res.name: res.to_xarray() for res in results.values()}
    # capture.analysis_filter = dict(capture.analysis_filter)
    # capture = msgspec.convert(capture, type=type(capture))
    attrs = msgspec.to_builtins(capture, builtin_types=(frozendict,))
    if isinstance(capture, structs.FilteredCapture):
        attrs['analysis_filter'] = dict(capture.analysis_filter)
    return xr.Dataset(xarrays, attrs=attrs)


def analyze_by_spec(
    iq: Array, capture: structs.Capture, *, spec: str | dict | structs.ChannelAnalysis
):
    """evaluate a set of different channel analyses on the iq waveform as specified by spec"""

    results = _evaluate_raw_channel_analysis(iq, capture, spec=spec)
    return _package_channel_analysis(capture, results)