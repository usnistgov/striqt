"""
Package analysis of a given waveform into an xarray.Dataset,
based a specification for calls to channel_analysis.waveform.
"""

from iqwaveform.util import set_input_domain
import xarray as xr

from channel_analysis.waveform import (
    persistence_spectrum,
    power_time_series,
    amplitude_probability_distribution,
    cyclic_channel_power,
    iq_waveform,
)

from channel_analysis.sources import WaveformSource

from . import host

# def _compatible_filter_and_spectrum(sample_rate_Hz, filter_spec, persistence_kws):
#     filter_spec = dict(bandwidth_Hz=None, sample_rate_Hz=sample_rate_Hz, **filter_spec)
#     persistence_kws = dict(
#         sample_rate_Hz=sample_rate_Hz, analysis_bandwidth_Hz=None, **persistence_kws
#     )

#     sig1 = signature(ola_filter).bind(None, **filter_spec).arguments
#     sig2 = signature(persistence_spectrum).bind(None, **persistence_kws).arguments
#     sig2['fft_size'] = round(sample_rate_Hz / persistence_kws['resolution'])

#     if sig1['window'] != 'hamming':
#         return False

#     for arg in ('fft_size', 'window'):
#         if sig1[arg] != sig2[arg]:
#             return False

#     return True
# def _sync_if_cuda(obj: Array):
#     if is_cupy_array(obj):
#         import cupy
#         cupy.cuda.Stream.null.synchronize()


def from_spec(
    iq, source: WaveformSource, *, analysis_spec: dict[str, dict[str]] = {}, cache={}
):
    analysis_spec = dict(analysis_spec)

    # then: analyses that need filtered output
    results = {}

    # evaluate each possible analysis function if specified
    for func in (
        power_time_series,
        cyclic_channel_power,
        amplitude_probability_distribution,
        iq_waveform,
        persistence_spectrum,
    ):
        try:
            func_kws = analysis_spec.pop(func.__name__)
        except KeyError:
            pass
        else:
            if func is persistence_spectrum and 'stft' in cache:
                # TODO: generalize this?
                # for now, this is hard-coded to assume overlap factor of 2 (hamming window)
                x = cache['stft'][::2]
                domain = 'frequency'
            else:
                x = iq
                domain = 'time'

            with set_input_domain(domain):
                results[func.__name__] = func(x, source, **func_kws)

    if len(analysis_spec) > 0:
        # anything left refers to an invalid function invalid
        raise ValueError(f'invalid analysis_spec key(s): {list(analysis_spec.keys())}')

    # materialize as xarrays on the cpu
    xarrays = {res.name: res.to_xarray() for res in results.values()}
    xarrays.update(host.host_index_variables())

    attrs = {
        'sample_rate': source.sample_rate,
        'analysis_bandwidth': source.analysis_bandwidth,
        **source.build_metadata(),
    }

    return xr.Dataset(xarrays, attrs=attrs)
