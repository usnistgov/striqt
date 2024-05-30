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
import array_api_compat

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


@lru_cache
def _label_detector_coords(detectors: tuple[str]):
    array = xr.DataArray(
        list(detectors), dims='power_detector', attrs={'label': 'Power detector'}
    )
    return {array.dims[0]: array}


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


def _to_maybe_nested_numpy(obj):
    ret = []

    if isinstance(obj, (tuple, list)):
        return [_to_maybe_nested_numpy(item) for item in obj]
    elif isinstance(obj, dict):
        return [_to_maybe_nested_numpy(item) for item in obj.values()]
    elif array_api_compat.is_torch_array(obj):
        return obj.cpu()
    elif array_api_compat.is_cupy_array(obj):
        return obj.get()
    elif array_api_compat.is_numpy_array(obj):
        return obj
    else:
        raise TypeError(f'obj type {type(obj)} is unrecognized')
        

def power_time_series(
    iq,
    *,
    sample_rate_Hz: float,
    analysis_bandwidth_Hz: float,
    detector_period: float,
    detectors=('rms', 'peak'),
) -> callable[[],xr.DataArray]:

    metadata = {'detector_period': detector_period}

    data = [
        iqwaveform.powtodB(
            iqwaveform.iq_to_bin_power(
                iq, Ts=1 / sample_rate_Hz, Tbin=detector_period, kind=detector
            )
        )
        for detector in detectors
    ]

    time_coords = _label_detector_time_coords(detector_period, len(data[0]))
    detector_coords = _label_detector_coords(detectors)

    coords = {**detector_coords, **time_coords}

    return lambda: xr.DataArray(
        _to_maybe_nested_numpy(data),
        coords=coords,
        dims=list(coords.keys()),
        name='power_time_series',
        attrs={
            'label': 'Channel power',
            'units': f'dBm/{analysis_bandwidth_Hz/1e6} MHz',
            **metadata,
        },
    )


@lru_cache
def _label_cyclic_power_coords(
    sample_rate_Hz: float,
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
    sample_rate_Hz,
    analysis_bandwidth_Hz,
    cyclic_period: float,
    detector_period: float,
    detectors: list[str] = ('rms', 'peak'),
    cyclic_statistics: list[str] = ('min', 'mean', 'max'),
) -> callable[[],xr.DataArray]:
    metadata = {'cyclic_period': cyclic_period, 'detector_period': detector_period}

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

    detector_coords = _label_detector_coords(detectors)
    cyclic_coords = _label_cyclic_power_coords(
        sample_rate_Hz,
        cyclic_period=cyclic_period,
        detector_period=detector_period,
        cyclic_statistics=cyclic_statistics,
    )
    coords = xr.Coordinates({**detector_coords, **cyclic_coords})

    attrs = {
        'label': 'Channel power',
        'units': f'dBm/{analysis_bandwidth_Hz/1e6} MHz',
        **metadata,
    }

    return lambda: xr.DataArray(
        _to_maybe_nested_numpy(data_dict),
        coords=coords,
        dims=list(coords.keys()),
        attrs=attrs,
        name='cyclic_channel_power',
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
    return {array.dims[0]: array}


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
    ccdf = iqwaveform.sample_ccdf(iqwaveform.envtodB(iq), bins)
    units = f'dBm/{analysis_bandwidth_Hz/1e6} MHz'

    return lambda: xr.DataArray(
        ccdf,
        coords=_label_apd_power_bins(xp=np, units=units, **bin_params),
        name='amplitude_probability_distribution',
        attrs={'label': 'Amplitude probability distribution', **bin_params},
    )


@lru_cache
def _baseband_frequency_to_coords(
    sample_rate_Hz: float, analysis_bandwidth_Hz: float, fft_size: int, time_size: int, overlap_frac: float = 0, truncate: bool = False, xp=np
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
    sample_rate_Hz: float,
    analysis_bandwidth_Hz=None,
    window,
    resolution: float,
    fractional_overlap=0,
    quantiles: list[float],
    truncate = False,
    bypass: bool = False,
    dB = False,
    spg = None
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
        'truncate': truncate
    }

    xp = array_namespace(iq)

    if spg is None:
        noverlap=int(np.rint(fractional_overlap*fft_size))
        freqs, times, spg = iqwaveform.fourier.spectrogram(
            iq, window=window, fs=sample_rate_Hz, nperseg=fft_size, noverlap=noverlap
        )

        if truncate:
            which_freqs = np.abs(freqs) <= analysis_bandwidth_Hz/2
            spg = spg[:,which_freqs]

    else:
        # TODO: 
        spg = spg[::2]

    if bypass:
        return spg

    import cupy
    cupy.cuda.runtime.deviceSynchronize()
    # spg = xp.ascontiguousarray(spg)

    if dB:
        spg = iqwaveform.powtodB(spg, eps=1e-25)

    data = xp.quantile(spg, xp.asarray(quantiles, dtype=xp.float32), axis=0)

    freq_coords = _baseband_frequency_to_coords(
        sample_rate_Hz=sample_rate_Hz,
        analysis_bandwidth_Hz=analysis_bandwidth_Hz,
        fft_size=fft_size,
        time_size=spg.shape[0],
        overlap_frac=fractional_overlap,
    )

    stat_coords = _persistence_stats_to_coords(tuple(quantiles))

    coords = {**freq_coords, **stat_coords}

    return lambda: xr.DataArray(
        _to_maybe_nested_numpy(data).T,
        dims=coords.keys(),
        coords=coords,
        name='persistence_spectrum',
        attrs={
            'label': 'Power spectral density',
            'units': f'dBm/{enbw/1e3:0.3f} kHz',
            **metadata,
        },
    )

    xp = array_api_compat.array_namespace(x)
    
    if out is None:
        ret = xp.empty_like(x)
    else:
        ret = out

    overlap_factor = int(np.rint(1/fractional_overlap))
    noverlap = fft_size // overlap_factor    

    freqs, times, X = fourier.stft(
        x, fs=sample_rate_Hz, window=window, nperseg=fft_size, noverlap=noverlap
    )
    X[..., xp.abs(freqs) > bandwidth_Hz/2] = 1e-35
    print('stft shape: ', X.shape)
    if array_api_compat.is_cupy_array(x):
        from cupyx import scipy
        x_inv = scipy.fft.ifft(xp.fft.fftshift(X, axes=-1), axis=-1)
    else:
        x_inv = fourier.ifft(xp.fft.fftshift(X, axes=-1), axis=-1)

    #x_inv = x_inv.reshape((overlap_factor, x_inv.shape[0]//overlap_factor, x_inv.shape[1]))

    print('x_inv: ', x_inv.shape)
    ret = x_inv[:,0]
    for i in range(x_inv.shape[1]):
        ret += xp.roll(x_inv[:,i], i*noverlap, axis=0)

    ret = ret.flatten()
    #ret = ret[fft_size:-fft_size]

    # ret[:] = x_inv.sum(axis=1).flatten()
    # ret[:] = x_inv[:,0,:].flatten()
    # apply overlap
    # ret[:-fft_size//overlap_factor] = x_inv[:,0].flatten()[:-fft_size//overlap_factor]
    # ret[-fft_size//overlap_factor:] = 0
    # for i in range(1,overlap_factor):
    #     istart = i*noverlap
    #     iend = ((i-overlap_factor+1)*noverlap) or None
    #     ret[istart:iend] += x_inv[:,i].flatten()[istart:iend]

    if return_spectrogram:
        # in-memory location of the real part of X
        x_real = X.T.view('float32')[:,::2].T

        # store mag^2 by overwriting the real part
        ret_spectrum = xp.abs(X)#, out=x_real)
        ret_spectrum *= ret_spectrum

        return ret, ret_spectrum
    else:
        return ret


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
    VALID_FUNCS = {
        'power_time_series',
        'persistence_spectrum',
        'amplitude_probability_distribution',
        'cyclic_channel_power',
    }

    xp = array_api_compat.array_namespace(iq)

    if len(analysis_spec.keys() - VALID_FUNCS) > 0:
        raise KeyError(
            f'analysis_spec keys may only be analysis function names: {VALID_FUNCS}'
        )

    if filter_spec == 'cupy':
        # TODO: allow a real spec here
        iq = fourier.ola_filter(
            iq,
            passband=(-analysis_bandwidth_Hz/2, analysis_bandwidth_Hz/2),
            fs=sample_rate_Hz,
            nperseg=768*16,
            noverlap=512*16,
            window='blackman'
            # return_spectrogram=True
        )
        spg = None
        import cupy
        cupy.cuda.runtime.deviceSynchronize()

    elif filter_spec is not None:
        sos = generate_iir_lpf(
            cutoff_Hz=analysis_bandwidth_Hz / 2,
            sample_rate_Hz=sample_rate_Hz,
            **filter_spec,
        )

        if array_api_compat.is_cupy_array(iq):
            from . import cuda_filter
            sos = xp.asarray(sos)
            iq = cuda_filter.sosfilt(sos.astype('float32'), iq)
            import cupy
            cupy.cuda.runtime.deviceSynchronize()

        else:
            iq = signal.sosfilt(sos.astype('float32'), iq)

        spg = None
    
    acq_kws = {
        'iq': iq,
        'sample_rate_Hz': sample_rate_Hz,
        'analysis_bandwidth_Hz': analysis_bandwidth_Hz,
    }

    # if array_api_compat.is_cupy_array(iq):
    #     import cupy as cp

    #     streams = {
    #         name: cp.cuda.stream.Stream(non_blocking=True)
    #         for name in analysis_spec.keys()
    #     }
    # else:
    #     streams = {}
    
    # get everything running in parallel first
    xarray_funcs = []
    for func_name, func_kws in analysis_spec.items():
        # if func_name in streams:
        #     with streams[func_name]:
        #         ret = globals()[func_name](**acq_kws, **func_kws)
        # else:

        if False:#func_name == 'persistence_spectrum':
            ret = globals()[func_name](spg=spg, **acq_kws, **func_kws)
        else:
            ret = globals()[func_name](**acq_kws, **func_kws)
        
        xarray_funcs.append(ret) 

    # then materialize the xarrays on the cpu
    xarrays = {}
    for func in xarray_funcs:
        xa = func()
        xarrays[xa.name] = xa

    return xr.Dataset(
        xarrays,
        attrs={
            'sample_rate_Hz': sample_rate_Hz,
            'analysis_bandwidth_Hz': analysis_bandwidth_Hz,
            'filter_specification': filter_spec,
        },
    )
