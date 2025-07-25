from __future__ import annotations
import decimal
import fractions
import typing

from ..lib import dataarrays, register, specs, util

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
    import iqwaveform.type_stubs
    import pandas as pd
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')


# %% Cellular 5G NR synchronizatino
class Cellular5GNRSyncCorrelationSpec(
    specs.Measurement,
    forbid_unknown_fields=True,
    cache_hash=True,
    kw_only=True,
    frozen=True,
    dict=True,
):
    """
    subcarrier_spacing (float): 3GPP channel subcarrier spacing (Hz)
    sample_rate (float): output sample rate for the resampled synchronization waveform (samples/s)
    discovery_periodicity (float): time period between synchronization blocks (s)
    frequency_offset (float or dict[float, float]):
        center frequency offset (see notes)
    shared_spectrum:
        whether to follow the 3GPP "shared spectrum" synchronizatio block layout
    max_block_count: number of synchronization blocks to evaluate
    trim_cp: whether to trim the cyclic prefix duration from the output
    """

    subcarrier_spacing: float
    sample_rate: float = 15.36e6 / 2
    discovery_periodicity: float = 20e-3
    frequency_offset: typing.Union[float, dict[float, float]] = 0
    shared_spectrum: bool = False
    max_block_count: typing.Optional[int] = 1
    trim_cp: bool = True


class Cellular5GNRSyncCorrelationKeywords(specs.AnalysisKeywords, total=False):
    subcarrier_spacing: float
    sample_rate: float
    discovery_periodicity: float
    frequency_offset: typing.Union[float, dict[float, float]]
    shared_spectrum: bool
    max_block_count: typing.Optional[int]
    trim_cp: bool


@register.coordinate_factory(
    dtype='uint16', attrs={'standard_name': r'Cell Sector ID ($N_{ID}^\text{(2)}$)'}
)
@util.lru_cache()
def cellular_cell_id2(capture: specs.Capture, spec: typing.Any):
    values = np.array([0, 1, 2], dtype='uint16')
    return values


@register.coordinate_factory(dtype='uint16', attrs={'standard_name': 'SSB beam index'})
@util.lru_cache()
def cellular_ssb_beam_index(
    capture: specs.Capture, spec: Cellular5GNRSyncCorrelationSpec
):
    # pss_params and sss_params return the same number of symbol indexes
    params = iqwaveform.ofdm.sss_params(
        sample_rate=spec.sample_rate,
        subcarrier_spacing=spec.subcarrier_spacing,
        discovery_periodicity=spec.discovery_periodicity,
        shared_spectrum=spec.shared_spectrum,
    )

    return list(range(len(params.symbol_indexes)))


@register.coordinate_factory(
    dtype='float32', attrs={'standard_name': 'Time Elapsed', 'units': 's'}
)
@util.lru_cache()
def cellular_ssb_start_time(
    capture: specs.Capture, spec: Cellular5GNRSyncCorrelationSpec
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


@register.coordinate_factory(
    dtype='float32', attrs={'standard_name': 'Symbol lag', 'units': 's'}
)
@util.lru_cache()
def cellular_ssb_lag(capture: specs.Capture, spec: Cellular5GNRSyncCorrelationSpec):
    params = iqwaveform.ofdm.sss_params(
        sample_rate=spec.sample_rate,
        subcarrier_spacing=spec.subcarrier_spacing,
        discovery_periodicity=spec.discovery_periodicity,
        shared_spectrum=spec.shared_spectrum,
    )

    max_len = 2 * round(spec.sample_rate / spec.subcarrier_spacing + params.cp_samples)

    if spec.trim_cp:
        max_len = max_len - round(0.5 * params.cp_samples)

    name = cellular_ssb_lag.__name__
    return pd.RangeIndex(0, max_len, name=name) / spec.sample_rate


def empty_5g_sync_measurement(
    iq,
    *,
    capture: specs.Capture,
    spec: Cellular5GNRSyncCorrelationSpec,
    coord_factories: list[callable],
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
    spec: Cellular5GNRSyncCorrelationSpec,
    params: 'iqwaveform.ofdm.SyncParams',
    cell_id_split: int | None = None,
):
    """correlate the IQ of a synchronization block against a synchronization sequence.

    Arguments:
        ssb_iq: The synchronization block IQ waveform (e.g., from `get_5g_ssb_iq`)
        sync_seq: The reference sequence (e.g. from `iqwaveform.ofdm.pss_5g_nr` or `iqwaveform.ofdm.sss_5g_nr`)
        spec: The measurement specification
        params: The cell synchronization parameters (e.g. from `iqwaveform.ofdm.pss_params` or `iqwaveform.ofdm.sss_params`)
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
    # print(iq_bcast.shape, template_bcast.shape)
    # template_bcast = iqwaveform.util.pad_along_axis(template_bcast, [[0,pad_size]], axis=3)

    cp_samples = round(9 / 128 * spec.sample_rate / spec.subcarrier_spacing)
    offs = round(spec.sample_rate / spec.subcarrier_spacing + 2 * cp_samples)

    R_shape = list(max(a, b) for a, b in zip(iq_bcast.shape, template_bcast.shape))
    R_shape[-1] = iq_bcast.shape[-1] + template_bcast.shape[-1] - 1
    R = xp.empty(tuple(R_shape), dtype='complex64')

    for cell_id in range(template_bcast.shape[1]):
        R[:, cell_id] = iqwaveform.oaconvolve(
            iq_bcast[:, 0], template_bcast[:, cell_id], axes=2, mode='full'
        )
    R = xp.roll(R, -offs, axis=-1)[..., :corr_size]

    # add slot index dimension: -> (port index, cell Nid, sync block index, slot index, IQ sample index)
    excess_cp = round(1 / 128 * spec.sample_rate / spec.subcarrier_spacing)
    R = R.reshape(R.shape[:-1] + (slot_count, -1))[..., 2 * excess_cp :]

    # dims -> (port index, cell Nid, sync block index, symbol pair index, IQ sample index)
    paired_symbol_shape = R.shape[:-2] + (7 * slot_count, -1)
    paired_symbol_indexes = xp.array(params.symbol_indexes, dtype='uint32') // 2
    R = R.reshape(paired_symbol_shape)[..., paired_symbol_indexes, :]

    if spec.trim_cp:
        R = R[..., : -cp_samples // 2]

    return R


ssb_iq_cache = register.KeywordArgumentCache([dataarrays.CAPTURE_DIM, 'spec'])


@ssb_iq_cache.apply
def get_5g_ssb_iq(
    iq: 'iqwaveform.type_stubs.ArrayType',
    capture: specs.Capture,
    spec: Cellular5GNRSyncCorrelationSpec,
    oaresample=False,
) -> 'iqwaveform.type_stubs.ArrayType':
    """return a sync block waveform, which returns IQ that is recentered
    at baseband frequency spec.frequency_offset and downsampled to spec.sample_rate."""

    xp = iqwaveform.util.array_namespace(iq)

    frequency_offset = specs.maybe_lookup_with_capture_key(
        capture,
        spec.frequency_offset,
        capture_attr='center_frequency',
        error_label='frequency_offset',
        default=None,
    )

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


# %% Spectral analysis


class FrequencyAnalysisSpecBase(
    specs.Measurement,
    forbid_unknown_fields=True,
    cache_hash=True,
    kw_only=True,
    frozen=True,
):
    """
    window (specs.WindowType): a window specification, following `scipy.signal.get_window`
    frequency_resolution (float): the STFT resolution (in Hz)
    fractional_overlap (float):
        fraction of each FFT window that overlaps with its neighbor
    window_fill (float):
        fraction of each FFT window that is filled with the window function
        (leaving the rest zeroed)
    video_bandwidth (float): bin bandwidth for RMS averaging in the frequency domain
    trim_stopband (bool):
        whether to trim the frequency axis to capture.analysis_bandwidth
    """

    window: specs.WindowType
    frequency_resolution: float
    fractional_overlap: fractions.Fraction = 0
    window_fill: fractions.Fraction = 1
    video_bandwidth: typing.Optional[float] = None
    trim_stopband: bool = True


class FrequencyAnalysisKeywords(specs.AnalysisKeywords):
    window: typing.NotRequired[specs.WindowType]
    frequency_resolution: typing.NotRequired[float]
    fractional_overlap: typing.NotRequired[float]
    window_fill: typing.NotRequired[float]
    video_bandwidth: typing.NotRequired[typing.Optional[float]]


class SpectrogramSpec(
    FrequencyAnalysisSpecBase,
    forbid_unknown_fields=True,
    cache_hash=True,
    kw_only=True,
    frozen=True,
):
    """
    time_aperture (float):
        if specified, binned RMS averaging is applied along time axis in the
        spectrogram to yield this coarser resolution (s)
    dB (bool): if True, returned power is transformed into dB units
    """

    time_aperture: typing.Optional[float] = None
    dB = True


class SpectrogramKeywords(FrequencyAnalysisKeywords):
    time_aperture: typing.NotRequired[typing.Optional[float]]


@util.lru_cache()
def equivalent_noise_bandwidth(window: specs.WindowType, nfft: int):
    """return the equivalent noise bandwidth (ENBW) of a window, in bins"""
    w = iqwaveform.fourier.get_window(window, nfft)
    return float(len(w) * np.sum(w**2) / np.sum(w) ** 2)


def evaluate_spectrogram(
    iq: 'iqwaveform.util.Array',
    capture: specs.Capture,
    spec: SpectrogramSpec,
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


def truncate_spectrogram_bandwidth(x, nfft, fs, bandwidth, axis=0):
    """trim an array outside of the specified bandwidth on a frequency axis"""
    edges = iqwaveform.fourier._freq_band_edges(
        nfft, 1.0 / fs, cutoff_low=-bandwidth / 2, cutoff_hi=bandwidth / 2
    )
    return iqwaveform.util.axis_slice(x, *edges, axis=axis)


@spectrogram_cache.apply
def _cached_spectrogram(
    iq: 'iqwaveform.util.Array',
    capture: specs.Capture,
    spec: SpectrogramSpec,
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

    if iqwaveform.isroundmod((1 - spec.window_fill) * nfft, 1):
        nzero = round((1 - spec.window_fill) * nfft)
    else:
        raise ValueError(
            '(1-window_fill) * (sample_rate/frequency_resolution) must be a counting number'
        )

    if spec.video_bandwidth is None:
        frequency_bin_averaging = None
    elif iqwaveform.isroundmod(spec.video_bandwidth, spec.frequency_resolution):
        frequency_bin_averaging = round(
            spec.video_bandwidth / spec.frequency_resolution
        )
    else:
        raise ValueError(
            'when specified, video_bandwidth must be a multiple of frequency_resolution'
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

    if time_bin_averaging is not None:
        spg = iqwaveform.util.binned_mean(spg, time_bin_averaging, axis=1, fft=False)

    enbw = spec.frequency_resolution * equivalent_noise_bandwidth(spec.window, nfft)

    attrs = {
        'noise_bandwidth': float(enbw),
        'units': f'dBm/{enbw / 1e3:0.0f} kHz',
    }

    return spg, attrs


def fftfreq(nfft, fs, dtype='float64') -> 'np.ndarray':
    """compute fftfreq for a specified sample rate.

    This is meant to produce higher-precision results for
    rational sample rates in order to avoid rounding errors
    when merging captures with different sample rates.
    """
    # high resolution rational representation of frequency resolution
    fres = decimal.Decimal(fs) / nfft
    span = range(-nfft // 2, -nfft // 2 + nfft)
    if nfft % 2 == 0:
        values = [fres * n for n in span]
    else:
        values = [fres * (n + 1) for n in span]
    return np.array(values, dtype=dtype)


@register.coordinate_factory(
    dtype='float64', attrs={'standard_name': 'Baseband Frequency', 'units': 'Hz'}
)
@util.lru_cache()
def spectrogram_baseband_frequency(
    capture: specs.Capture, spec: SpectrogramSpec, xp=np
) -> dict[str, np.ndarray]:
    if xp is not np:
        return xp.array(spectrogram_baseband_frequency(capture, spec))

    if iqwaveform.isroundmod(capture.sample_rate, spec.frequency_resolution):
        nfft = round(capture.sample_rate / spec.frequency_resolution)
    else:
        raise ValueError('sample_rate/resolution must be a counting number')

    if spec.video_bandwidth is None:
        frequency_bin_averaging = None
    elif iqwaveform.isroundmod(spec.video_bandwidth, spec.frequency_resolution):
        frequency_bin_averaging = round(
            spec.video_bandwidth / spec.frequency_resolution
        )
    else:
        raise ValueError(
            'when specified, video_bandwidth must be a multiple of frequency_resolution'
        )

    # use the iqwaveform.fourier fftfreq for higher precision, which avoids
    # headaches when merging spectra with different sampling parameters due
    # to rounding errors.
    freqs = fftfreq(nfft, capture.sample_rate)

    if spec.trim_stopband and np.isfinite(capture.analysis_bandwidth):
        # stick with python arithmetic here for numpy/cupy consistency
        freqs = truncate_spectrogram_bandwidth(
            freqs, nfft, capture.sample_rate, capture.analysis_bandwidth, axis=0
        )

    if spec.video_bandwidth is not None:
        freqs = iqwaveform.util.binned_mean(freqs, frequency_bin_averaging)
        freqs -= freqs[freqs.size // 2]

    # only now downconvert. round to a still-large number of digits
    return freqs.astype('float64').round(16)
