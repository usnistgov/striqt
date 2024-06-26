""""""

from array_api_strict._typing import Array
from array_api_compat import is_cupy_array
from iqwaveform.util import array_namespace, array_stream, set_input_domain

from channel_analysis.waveform import (
    ola_filter,
    persistence_spectrum,
    power_time_series,
    amplitude_probability_distribution,
    cyclic_channel_power,
    iq_waveform,
)

from inspect import signature
import xarray as xr

from . import metadata


def _compatible_filter_and_spectrum(sample_rate_Hz, filter_spec, persistence_kws):
    filter_spec = dict(bandwidth_Hz=None, sample_rate_Hz=sample_rate_Hz, **filter_spec)
    persistence_kws = dict(
        sample_rate_Hz=sample_rate_Hz, analysis_bandwidth_Hz=None, **persistence_kws
    )

    sig1 = signature(ola_filter).bind(None, **filter_spec).arguments
    sig2 = signature(persistence_spectrum).bind(None, **persistence_kws).arguments
    sig2['fft_size'] = round(sample_rate_Hz / persistence_kws['resolution'])

    if sig1['window'] != 'hamming':
        return False

    for arg in ('fft_size', 'window'):
        if sig1[arg] != sig2[arg]:
            return False

    return True



def _sync_if_cuda(obj: Array):
    if is_cupy_array(obj):
        import cupy

        cupy.cuda.Stream.null.synchronize()


def from_spec(
    iq,
    sample_rate_Hz,
    analysis_bandwidth_Hz,
    *,
    filter_spec: dict = {
        'fft_size': 1024,
        'window': 'hamming',  # 'hamming', 'blackman', or 'blackmanharris'
    },
    analysis_spec: dict[str, dict[str]] = {},
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

    stream = array_stream(iq, non_blocking=True, null=True, ptds=True)
    stream.use()

    cache = {}

    # first: everything that doesn't need the filter output
    if filter_spec is not None and 'persistence_spectrum' in analysis_spec:
        reuse_ola_stft = _compatible_filter_and_spectrum(
            sample_rate_Hz, filter_spec, analysis_spec['persistence_spectrum']
        )
    else:
        reuse_ola_stft = False

    iq = ola_filter(
        iq,
        bandwidth_Hz=analysis_bandwidth_Hz,
        sample_rate_Hz=sample_rate_Hz,
        cache=cache if reuse_ola_stft else None,
        **filter_spec,
    )

    stream.synchronize()

    # then: analyses that need filtered output
    results = {}

    for func in (
        power_time_series,
        cyclic_channel_power,
        amplitude_probability_distribution,
        iq_waveform,
        persistence_spectrum,
    ):
        # check for each allowed function in the specification
        try:
            func_kws = analysis_spec.pop(func.__name__)
        except KeyError:
            pass
        else:
            if func is persistence_spectrum and 'stft' in cache:
                # for now, this is hard-coded to assume overlap factor of 2 (hamming window)
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
