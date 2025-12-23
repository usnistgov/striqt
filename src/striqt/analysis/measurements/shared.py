from __future__ import annotations as __

import decimal
import fractions
import typing

from .. import specs

from ..lib import dataarrays, register, util
from ..lib.register import registry

if typing.TYPE_CHECKING:
    from types import ModuleType
    from typing_extensions import ParamSpec

    import array_api_compat
    import numpy as np
    import pandas as pd

    import striqt.waveform as iqwaveform
    from striqt.waveform._typing import ArrayType

    _P = ParamSpec('_P')
    _R = typing.TypeVar('_R', covariant=True)

else:
    iqwaveform = util.lazy_import('striqt.waveform')
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')
    array_api_compat = util.lazy_import('array_api_compat')


class _AnalysisProtocol(typing.Protocol[_P, _R]):
    def __call__(
        self,
        iq: 'iqwaveform.util.ArrayType',
        capture: specs.Capture,
        as_xarray: specs.types.AsXArray = True,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R: ...

    __name__: str


def hint_keywords(
    func: typing.Callable[_P, typing.Any],
) -> typing.Callable[[typing.Callable[..., _R]], _AnalysisProtocol[_P, _R]]:
    """fill in type hints for the analysis parameters"""
    return lambda f: f  # type: ignore


@registry.coordinates(
    dtype='uint16', attrs={'standard_name': r'Cell Sector ID ($N_{ID}^\text{(2)}$)'}
)
@util.lru_cache()
def cellular_cell_id2(capture: specs.Capture, spec: typing.Any):
    values = np.array([0, 1, 2], dtype='uint16')
    return values


@registry.coordinates(dtype='uint16', attrs={'standard_name': 'SSB beam index'})
@util.lru_cache()
def cellular_ssb_beam_index(
    capture: specs.Capture, spec: specs.Cellular5GNRSSBSync
):
    # pss_params and sss_params return the same number of symbol indexes
    params = iqwaveform.ofdm.sss_params(
        sample_rate=spec.sample_rate,
        subcarrier_spacing=spec.subcarrier_spacing,
        discovery_periodicity=spec.discovery_periodicity,
        shared_spectrum=spec.shared_spectrum,
    )

    return list(range(len(params.symbol_indexes)))


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Time Elapsed', 'units': 's'}
)
@util.lru_cache()
def cellular_ssb_start_time(
    capture: specs.Capture, spec: specs.Cellular5GNRSSBSync
):
    # pss_params and sss_params return the same number of symbol indexes
    params = iqwaveform.ofdm.pss_params(
        sample_rate=spec.sample_rate,
        subcarrier_spacing=spec.subcarrier_spacing,
        discovery_periodicity=spec.discovery_periodicity,
        shared_spectrum=spec.shared_spectrum,
    )
    total_blocks = round(params.duration / spec.discovery_periodicity)
    if spec.max_block_count is None:
        count = total_blocks
    else:
        count = min(spec.max_block_count, total_blocks)

    return np.arange(max(count, 1)) * spec.discovery_periodicity


@registry.coordinates(
    dtype='float32', attrs={'standard_name': 'Symbol lag', 'units': 's'}
)
@util.lru_cache()
def cellular_ssb_lag(
    capture: specs.Capture, spec: specs.Cellular5GNRSSBCorrelator
):
    params = iqwaveform.ofdm.sss_params(
        sample_rate=spec.sample_rate,
        subcarrier_spacing=spec.subcarrier_spacing,
        discovery_periodicity=spec.discovery_periodicity,
        shared_spectrum=spec.shared_spectrum,
    )

    max_len = 2 * round(spec.sample_rate / spec.subcarrier_spacing + params.min_cp_size)

    if spec.trim_cp:
        max_len = max_len - params.min_cp_size

    name = cellular_ssb_lag.__name__
    return pd.RangeIndex(0, max_len, name=name) / spec.sample_rate


def empty_5g_ssb_correlation(
    iq,
    *,
    capture: specs.Capture,
    spec: specs.Cellular5GNRSSBCorrelator,
    coord_factories: list[typing.Callable],
    dtype='complex64',
):
    xp = iqwaveform.util.array_namespace(iq)
    meas_ax_shape = [len(f(capture, spec)) for f in coord_factories]
    new_shape = iq.shape[:-1] + tuple(meas_ax_shape)
    return xp.full(new_shape, 0, dtype=dtype)


def correlate_sync_sequence(
    ssb_iq,
    sync_seq,
    *,
    spec: specs.Cellular5GNRSSBCorrelator,
    params: iqwaveform.ofdm.SyncParams,
    cell_id_split: int | None = None,
):
    """correlate the IQ of a synchronization block against a synchronization sequence.

    Arguments:
        ssb_iq: The synchronization block IQ waveform (e.g., from `get_5g_ssb_iq`)
        sync_seq: The reference sequence (e.g. from `striqt.waveform.ofdm.pss_5g_nr` or `striqt.waveform.ofdm.sss_5g_nr`)
        spec: The measurement specification
        params: The cell synchronization parameters (e.g. from `striqt.waveform.ofdm.pss_params` or `striqt.waveform.ofdm.sss_params`)
        cell_id_split: if not None, operate on groups of this size along the cell id axis (to reduce memory usage)
    """
    xp = iqwaveform.util.array_namespace(ssb_iq)

    slot_count = params.slot_count
    corr_size = params.corr_size
    frames_per_sync = params.frames_per_sync

    # set up broadcasting to new dimensions:
    # (port index, cell Nid, sync block index, IQ sample index)
    iq_bcast = ssb_iq.reshape((ssb_iq.shape[0], -1, params.frame_size))
    iq_bcast = iq_bcast[:, xp.newaxis, ::frames_per_sync, :corr_size]
    template_bcast = sync_seq[xp.newaxis, :, xp.newaxis, :]
    # pad_size = template_bcast.shape[-1]
    # template_bcast  = iqwaveform.util.pad_along_axis(template_bcast, [[0,pad_size]], axis=3)

    # TODO: this would need to support multiple different prefixes for 4G LTE
    offs = round(spec.sample_rate / spec.subcarrier_spacing) + 2 * params.min_cp_size

    # cp_samples = round(9 / 128 * spec.sample_rate / spec.subcarrier_spacing)
    # offs = round(spec.sample_rate / spec.subcarrier_spacing + 2 * cp_samples)

    R_shape = list(max(a, b) for a, b in zip(iq_bcast.shape, template_bcast.shape))
    R_shape[-1] = iq_bcast.shape[-1] + template_bcast.shape[-1] - 1
    R = xp.empty(tuple(R_shape), dtype='complex64')

    # TODO: cell_id_split wasn't implemented restore it here
    for cell_id in range(template_bcast.shape[1]):
        R[:, cell_id] = iqwaveform.oaconvolve(
            iq_bcast[:, 0], template_bcast[:, cell_id], axes=2, mode='full'
        )
    R = xp.roll(R, -offs, axis=-1)[..., :corr_size]
    R = R[..., :corr_size]

    # add slot index dimension: -> (port index, cell Nid, sync block index, slot index, IQ sample index)
    excess_cp = [params.cp_offsets[i % 14] for i in params.symbol_indexes]
    if len(set(excess_cp)) != 1:
        raise ValueError('expect all 5G sync symbols to have the same excess CP')
    R = R.reshape(R.shape[:-1] + (slot_count, -1))[..., excess_cp[0] :]

    # dims -> (port index, cell Nid, sync block index, symbol pair index, IQ sample index)
    paired_symbol_shape = R.shape[:-2] + (7 * slot_count, -1)
    paired_symbol_indexes = xp.array(params.symbol_indexes, dtype='uint32') // 2
    R = R.reshape(paired_symbol_shape)[..., paired_symbol_indexes, :]

    if spec.trim_cp:
        R = R[..., : -params.min_cp_size]

    return R


ssb_iq_cache = register.KeywordArgumentCache([dataarrays.CAPTURE_DIM, 'spec'])


@ssb_iq_cache.apply
def get_5g_ssb_iq(
    iq: ArrayType,
    capture: specs.Capture,
    spec: specs.Cellular5GNRSSBCorrelator,
    oaresample=False,
) -> ArrayType:
    """return a sync block waveform, which returns IQ that is recentered
    at baseband frequency spec.frequency_offset and downsampled to spec.sample_rate."""

    xp = iqwaveform.util.array_namespace(iq)

    frequency_offset = specs.helpers.maybe_lookup_with_capture_key(
        capture,
        spec.frequency_offset,
        capture_attr='center_frequency',
        error_label='frequency_offset',
        default=None,
    )

    assert isinstance(frequency_offset, float)

    if frequency_offset is None:
        return None

    if oaresample:
        down = round(capture.sample_rate / spec.subcarrier_spacing / 8)
        up = round(down * (spec.sample_rate / capture.sample_rate))

        if up % 3 > 0:
            # ensure compatibility with the blackman window overlap of 2/3
            down = down * 3
            up = up * 3

        if spec.max_block_count is not None:
            size_in = round(
                spec.max_block_count * spec.discovery_periodicity * capture.sample_rate
            )
            iq = iq[..., :size_in]
        else:
            size_in = iq.shape[-1]

        size_out = round(up / down * size_in)

        out = xp.empty((iq.shape[0], size_out), dtype=iq.dtype)

        for i in range(out.shape[0]):
            out[i] = iqwaveform.fourier.oaresample(
                iq[i],
                fs=capture.sample_rate,
                up=up,
                down=down,
                axis=0,
                window='blackman',
                frequency_shift=frequency_offset,
            )

    else:
        if spec.max_block_count is not None:
            size_in = round(
                spec.max_block_count * spec.discovery_periodicity * capture.sample_rate
            )
            iq = iq[..., :size_in]
        else:
            size_in = iq.shape[-1]

        size_out = round(size_in * spec.sample_rate / capture.sample_rate)
        out = xp.empty((iq.shape[0], size_out), dtype=iq.dtype)
        shift = round(iq.shape[1] * frequency_offset / capture.sample_rate)

        for i in range(out.shape[0]):
            out[i] = iqwaveform.fourier.resample(
                iq[i], num=size_out, axis=0, overwrite_x=False, shift=shift
            )

    return out


def evaluate_spectrogram(
    iq: ArrayType,
    capture: specs.Capture,
    spec: specs.Spectrogram,
    *,
    dtype: typing.Union[
        typing.Literal['float16'], typing.Literal['float32']
    ] = 'float32',
    limit_digits: typing.Optional[int] = None,
    dB=True,
):
    spg, attrs = _cached_spectrogram(iq=iq, capture=capture, spec=spec)
    xp = iqwaveform.util.array_namespace(iq)

    copied = False
    if dB:
        spg = iqwaveform.powtodB(spg, eps=1e-25)
        copied = True

    if limit_digits is not None:
        spg = xp.round(spg, limit_digits, out=spg if copied else None)
        copied = True

    if dtype == 'float16':
        spg = spg.astype(dtype, copy=not copied)

    attrs = attrs | {'limit_digits': limit_digits}

    return spg, attrs


spectrogram_cache = register.KeywordArgumentCache([dataarrays.CAPTURE_DIM, 'spec'])


def truncate_spectrogram_bandwidth(x, nfft, fs, bandwidth, *, offset=0, axis=0):
    """trim an array outside of the specified bandwidth on a frequency axis"""
    edges = iqwaveform.fourier._freq_band_edges(
        nfft,
        1.0 / fs,
        cutoff_low=offset - bandwidth / 2,
        cutoff_hi=offset + bandwidth / 2,
    )
    return iqwaveform.util.axis_slice(x, *edges, axis=axis)


def null_lo(x, nfft, fs, bandwidth, *, offset=0, axis=0):
    """sets samples to nan within the specified bandwidth on a frequency axis"""
    # to make the top bound inclusive
    pad_hi = fs / nfft / 2
    edges = iqwaveform.fourier._freq_band_edges(
        nfft,
        1.0 / fs,
        cutoff_low=offset - bandwidth / 2,
        cutoff_hi=offset + bandwidth / 2 + pad_hi,
    )
    view = iqwaveform.util.axis_slice(x, *edges, axis=axis)
    view[:] = float('nan')


@spectrogram_cache.apply
def _cached_spectrogram(
    iq: ArrayType,
    capture: specs.Capture,
    spec: specs.Spectrogram,
):
    spec = spec.validate()

    if iqwaveform.isroundmod(capture.sample_rate, spec.frequency_resolution):
        nfft = round(capture.sample_rate / spec.frequency_resolution)
    else:
        raise ValueError('sample_rate/resolution must be a counting number')

    if iqwaveform.isroundmod(capture.sample_rate, spec.frequency_resolution):
        noverlap = round(spec.fractional_overlap * nfft)
    else:
        raise ValueError('sample_rate_Hz/resolution must be a counting number')

    nzero = (1 - spec.window_fill) * nfft
    if nzero.denominator == 1:
        nzero = nzero.numerator
    else:
        raise ValueError(
            '(1-window_fill) * sample_rate must be a counting-number multiple of frequency_resolution'
        )

    if spec.integration_bandwidth is None:
        frequency_bin_averaging = None
    elif iqwaveform.isroundmod(spec.integration_bandwidth, spec.frequency_resolution):
        frequency_bin_averaging = round(
            spec.integration_bandwidth / spec.frequency_resolution
        )
    else:
        raise ValueError(
            'when specified, integration_bandwidth must be a multiple of frequency_resolution'
        )

    hop_size = nfft - noverlap
    hop_period = hop_size / capture.sample_rate
    if spec.time_aperture is None:
        time_bin_averaging = None
    elif iqwaveform.isroundmod(spec.time_aperture, hop_period):
        time_bin_averaging = round(spec.time_aperture / hop_period)
    else:
        raise ValueError(
            'when specified, time_aperture must be a multiple of (1-fractional_overlap)/frequency_resolution'
        )

    spg = iqwaveform.fourier.spectrogram(
        iq,
        window=spec.window,
        fs=capture.sample_rate,
        nperseg=nfft,
        noverlap=noverlap,
        nzero=nzero,
        axis=1,
        return_axis_arrays=False,
    )

    if spec.lo_bandstop is not None:
        null_lo(spg, nfft, capture.sample_rate, spec.lo_bandstop, axis=2)

    # truncate to the analysis bandwidth
    if spec.trim_stopband and np.isfinite(capture.analysis_bandwidth):
        # stick with python arithmetic to ensure consistency with axis bounds calculations
        spg = truncate_spectrogram_bandwidth(
            spg, nfft, capture.sample_rate, bandwidth=capture.analysis_bandwidth, axis=2
        )

    if frequency_bin_averaging is not None:
        spg = iqwaveform.util.binned_mean(
            spg, frequency_bin_averaging, axis=2, fft=True
        )

        # mean -> sum
        spg *= frequency_bin_averaging

    if time_bin_averaging is not None:
        spg = iqwaveform.util.binned_mean(spg, time_bin_averaging, axis=1, fft=False)

    if spec.integration_bandwidth is None:
        enbw = spec.frequency_resolution
    else:
        enbw = spec.integration_bandwidth

    attrs = {
        'noise_bandwidth': float(enbw),
        'units': f'dBm/{enbw / 1e3:0.0f} kHz',
    }

    return spg, attrs


@util.lru_cache()
def fftfreq(nfft: int, fs: float, dtype='float64', xp: ModuleType = np) -> ArrayType:
    """compute fftfreq for a specified sample rate.

    This is meant to produce higher-precision results for
    rational sample rates in order to avoid rounding errors
    when merging captures with different sample rates.
    """

    if not array_api_compat.is_numpy_namespace(np):  # type: ignore
        return xp.asarray(fftfreq(nfft, fs, dtype))

    # high resolution rational representation of frequency resolution
    fres = decimal.Decimal(fs) / nfft
    span = range(-nfft // 2, -nfft // 2 + nfft)
    if nfft % 2 == 0:
        values = [fres * n for n in span]
    else:
        values = [fres * (n + 1) for n in span]
    return np.array(values, dtype=dtype)


@registry.coordinates(
    dtype='float64', attrs={'standard_name': 'Baseband Frequency', 'units': 'Hz'}
)
@util.lru_cache()
def spectrogram_baseband_frequency(
    capture: specs.Capture, spec: specs.Spectrogram, xp=np
) -> np.ndarray:
    if xp is not np:
        return xp.array(spectrogram_baseband_frequency(capture, spec))

    if iqwaveform.isroundmod(capture.sample_rate, spec.frequency_resolution):
        nfft = round(capture.sample_rate / spec.frequency_resolution)
    else:
        raise ValueError('sample_rate/resolution must be a counting number')

    if spec.integration_bandwidth is None:
        frequency_bin_averaging = None
    elif iqwaveform.isroundmod(spec.integration_bandwidth, spec.frequency_resolution):
        frequency_bin_averaging = round(
            spec.integration_bandwidth / spec.frequency_resolution
        )
    else:
        raise ValueError(
            'when specified, integration_bandwidth must be a multiple of frequency_resolution'
        )

    # use the striqt.waveform.fourier fftfreq for higher precision, which avoids
    # headaches when merging spectra with different sampling parameters due
    # to rounding errors.
    freqs = fftfreq(nfft, capture.sample_rate)

    if spec.trim_stopband and np.isfinite(capture.analysis_bandwidth):
        # stick with python arithmetic here for numpy/cupy consistency
        freqs = truncate_spectrogram_bandwidth(
            freqs, nfft, capture.sample_rate, capture.analysis_bandwidth, axis=0
        )

    if spec.integration_bandwidth is not None:
        freqs = iqwaveform.util.binned_mean(freqs, frequency_bin_averaging, fft=True)

    # only now downconvert. round to a still-large number of digits
    return freqs.astype('float64').round(16)
