from __future__ import annotations
import dataclasses
import functools
import typing

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr

from ..api.registry import register_xarray_measurement
from ._spectrogram import equivalent_noise_bandwidth, truncate_freqs
from ._channel_power_ccdf import make_power_bins, ChannelPowerCoords

from ..api import structs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')

# Axis and coordinates
SpectrogramPowerBinAxis = typing.Literal['spectrogram_power_bin']


@dataclasses.dataclass
class SpectrogramPowerBinCoords:
    data: Data[SpectrogramPowerBinAxis, np.float32]
    standard_name: Attr[str] = 'Spectrogram bin power'
    units: Attr[str] = 'dBm'

    @staticmethod
    @functools.lru_cache
    def factory(
        capture: structs.Capture,
        *,
        window: typing.Union[str, tuple[str, float]],
        frequency_resolution: float,
        power_low: float,
        power_high: float,
        power_resolution: float,
        **_,
    ) -> dict[str, np.ndarray]:
        """returns a dictionary of coordinate values, keyed by axis dimension name"""
        bins = ChannelPowerCoords.factory(
            capture,
            power_low=power_low,
            power_high=power_high,
            power_resolution=power_resolution,
        )

        if iqwaveform.isroundmod(capture.sample_rate, frequency_resolution):
            # need capture.sample_rate/resolution to give us a counting number
            nfft = round(capture.sample_rate / frequency_resolution)
        else:
            raise ValueError('sample_rate/resolution must be a counting number')

        if isinstance(window, list):
            # lists break lru_cache
            window = tuple(window)

        enbw = frequency_resolution * equivalent_noise_bandwidth(window, nfft)

        return bins, {'units': f'dBm/{enbw/1e3:0.0f} kHz'}


@dataclasses.dataclass
class SpectrogramCCDF(AsDataArray):
    ccdf: Data[SpectrogramPowerBinAxis, np.float32]
    spectrogram_power_bin: Coordof[SpectrogramPowerBinCoords]
    standard_name: Attr[str] = 'CCDF'
    long_name: Attr[str] = r'Fraction of counts centered in bin channel power level'


@register_xarray_measurement(SpectrogramCCDF)
def spectrogram_ccdf(
    iq: 'iqwaveform.util.Array',
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

    if np.isfinite(capture.analysis_bandwidth):
        spg = truncate_freqs(spg, nfft, capture.sample_rate, capture.analysis_bandwidth, axis=0)

    if frequency_bin_averaging is not None:
        trim = spg.shape[1] % (2 * frequency_bin_averaging)
        if trim > 0:
            spg = spg[:, trim // 2 : -trim // 2 :]
        spg = iqwaveform.fourier.to_blocks(spg, frequency_bin_averaging, axis=1)
        spg = spg.mean(axis=2)

    spg = iqwaveform.powtodB(spg, eps=1e-25, out=spg)

    bins = make_power_bins(
        power_low=power_low,
        power_high=power_high,
        power_resolution=power_resolution,
        xp=xp,
    )
    data = iqwaveform.sample_ccdf(spg.flatten(), bins).astype(dtype)

    metadata = {
        'window': window,
        'frequency_resolution': frequency_resolution,
        'fractional_overlap': fractional_overlap,
        'noise_bandwidth': enbw,
        'fft_size': nfft,
        # 'units': f'dBm/{enbw/1e3:0.3f} kHz',
    }

    return data, metadata
