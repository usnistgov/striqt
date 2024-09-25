from __future__ import annotations
import dataclasses
import typing

import numpy as np

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr
import iqwaveform

from .._api import structs, type_stubs
from ._common import as_registered_channel_analysis
from ._persistence_spectrum import equivalent_noise_bandwidth
from ._spectrogram_ccdf import SpectrogramCoords, SpectrogramPowerBinAxis
from ._channel_power_histogram import make_power_histogram_bin_edges


@dataclasses.dataclass
class SpectrogramHistogram(AsDataArray):
    ccdf: Data[SpectrogramPowerBinAxis, np.float32]
    spectrogram_power_bin: Coordof[SpectrogramCoords]
    standard_name: Attr[str] = 'Fraction of counts'


@as_registered_channel_analysis(SpectrogramHistogram)
def spectrogram_histogram(
    iq: type_stubs.ArrayType,
    capture: structs.Capture,
    *,
    window: typing.Union[str, tuple[str, float]],
    frequency_resolution: float,
    power_low: float,
    power_high: float,
    power_resolution: float,
    fractional_overlap: float = 0,
    frequency_bin_averaging: int = None,
):
    xp = iqwaveform.util.array_namespace(iq)
    dtype = xp.finfo(iq.dtype).dtype

    # TODO: integrate this back into iqwaveform
    if iqwaveform.isroundmod(capture.sample_rate, frequency_resolution):
        nfft = round(capture.sample_rate / frequency_resolution)
    else:
        raise ValueError('sample_rate/resolution must be a counting number')

    if isinstance(window, list):
        # lists break lru_cache
        window = tuple(window)

    enbw = frequency_resolution * equivalent_noise_bandwidth(window, nfft)

    if iqwaveform.isroundmod(capture.sample_rate, frequency_resolution):
        nfft = round(capture.sample_rate / frequency_resolution)
        noverlap = round(fractional_overlap * nfft)
    else:
        # need sample_rate_Hz/resolution to give us a counting number
        raise ValueError('sample_rate_Hz/resolution must be a counting number')

    freqs, _, spg = iqwaveform.fourier.spectrogram(
        iq,
        window=window,
        fs=capture.sample_rate,
        nperseg=nfft,
        noverlap=noverlap,
        axis=0,
    )

    # truncate to the analysis bandwidth
    bw_args = (-capture.analysis_bandwidth / 2, +capture.analysis_bandwidth / 2)
    ilo, ihi = iqwaveform.fourier._freq_band_edges(
        freqs[0], freqs[-1], freqs.size, *bw_args
    )
    spg = spg[:, ilo:ihi]

    if frequency_bin_averaging is not None:
        trim = spg.shape[1] % (2 * frequency_bin_averaging)
        if trim > 0:
            spg = spg[:, trim // 2 : -trim // 2 :]
        spg = iqwaveform.fourier.to_blocks(spg, frequency_bin_averaging, axis=1)
        spg = spg.mean(axis=2)

    spg = iqwaveform.powtodB(spg, eps=1e-25, out=spg)

    bin_edges = make_power_histogram_bin_edges(
        power_low=power_low,
        power_high=power_high,
        power_resolution=power_resolution,
        xp=xp,
    )

    counts, _ = xp.histogram(spg.flatten(), bin_edges)
    data = counts.astype(dtype) / xp.sum(counts)

    metadata = {
        'window': window,
        'frequency_resolution': frequency_resolution,
        'fractional_overlap': fractional_overlap,
        'noise_bandwidth': enbw,
        'fft_size': nfft,
        # 'units': f'dBm/{enbw/1e3:0.3f} kHz',
    }

    return data, metadata
