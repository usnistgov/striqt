"""striqt.analysis.measurements.cellular: the synchronization and cyclic prefix
measurements cellular_5g_pss_correlation, cellular_5g_sss_correlation,
cellular_5g_pss_sync, cellular_5g_sss_sync and cellular_cyclic_autocorrelation, plus
cellular_5g_ssb_spectrogram and cellular_resource_power_histogram.

`tests/waveform/test_waveform_ofdm.py` proves the correlators and the cyclic prefix
kernel numerically. What the measurement layer adds, and what is tested here, is the
(capture, spec) validation that rejects a combination before acquisition: the
resampler's grid and band limits and the 10 ms frame count for the correlators, and
the frame, slot and symbol index bounds for the autocorrelation.

Two frames at 3.84 MS/s is the smallest capture at which a 30 kHz cell search (case
C, 3GPP TS 38.213 Section 4.1) and a two-frame `frame_range` both fit.

The SSB spectrogram is checked against the closed form of a bin-centered tone through
the zero-padded boxcar window; the resource grid histogram against the constant
envelope of a tone, which puts the whole normalized fraction in one power bin.
"""

from __future__ import annotations

import re

import msgspec
import numpy as np
import pytest
from analysis_strategies import (
    POWER_BINS,
    SSB_SPECTROGRAM_FS,
    SSB_SPECTROGRAM_PERIODICITY,
    SSB_SPECTROGRAM_SAMPLE_RATE,
    SSB_SPECTROGRAM_SCS,
    SSB_SPECTROGRAM_SPEC,
    registered_tolerance,
    ssb_spectrogram_capture,
)
from numeric_checks import RTOL_FLOAT64, assert_close, elementwise_rtol, levels

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis import testing
from striqt.analysis.measurements import cellular

FS = 3.84e6
FRAME = round(10e-3 * FS)
DURATION = 2 * FRAME / FS
SCS = 30e3
SSB = {'subcarrier_spacing': SCS, 'sample_rate': FS, 'symbol_indexes': 'c'}

SSB_SPEC_TYPES = [
    sa.specs.Cellular5GNRPSSCorrelator,
    sa.specs.Cellular5GNRSSSCorrelator,
    sa.specs.Cellular5GNPSSSync,
    sa.specs.Cellular5GNSSSSync,
]
SSB_IDS = [cls.__name__ for cls in SSB_SPEC_TYPES]


def capture(samples=2 * FRAME, sample_rate=FS) -> sa.specs.Capture:
    return sa.specs.Capture(duration=samples / sample_rate, sample_rate=sample_rate)


def measure(spec, cap: sa.specs.Capture):
    """run the registered measurement for `spec` the way the registry calls it"""
    info = sa.registry[type(spec)]
    iq = testing.noise(cap.duration, cap.sample_rate, noise_psd=1 / cap.sample_rate)
    return info.func(iq, cap, as_xarray=False, **spec.to_dict())


# %% the 5G NR SSB correlators and synchronizers


@pytest.mark.parametrize('cls', SSB_SPEC_TYPES, ids=SSB_IDS)
def test_ssb_measurements_share_one_validator(cls):
    """`sss_params` reaches every check in `pss_params`, so the four registrations
    resolve their sync layout through one function"""
    assert sa.registry[cls].validate is cellular.validated_5g_ssb_sync_params


SSB_REJECTED = {
    'odd_sample_count': (
        capture(2 * FRAME + 1),
        {},
        'the capture cannot be resampled to the synchronization block: the input length must be even',
    ),
    'frequency_offset_off_the_resampler_grid': (
        capture(),
        # the resampler's frequency step is sample_rate/duration = 50 Hz
        {'frequency_offset': 125.0},
        'frequency_offset must be a counting-number multiple of sample_rate/duration',
    ),
    'frequency_offset_shifts_past_the_band': (
        capture(),
        {'subcarrier_spacing': 15e3, 'sample_rate': FS / 2, 'frequency_offset': FS / 2},
        'the capture cannot be resampled to the synchronization block: shift is too large',
    ),
    'frequency_offset_without_downsampling': (
        capture(),
        # `resample` shifts only while downsampling
        {'frequency_offset': 50.0},
        'the capture cannot be resampled to the synchronization block',
    ),
    'partial_frame': (
        capture(2 * FRAME + FRAME // 2),
        {},
        'duration must hold whole 10 ms frames for the correlator',
    ),
    'block_shorter_than_a_frame': (
        capture(FRAME, sample_rate=2 * FS),
        {'sample_rate': FS / 2, 'subcarrier_spacing': 15e3},
        'duration must hold whole 10 ms frames for the correlator',
    ),
}


@pytest.mark.parametrize('cls', SSB_SPEC_TYPES, ids=SSB_IDS)
@pytest.mark.parametrize('case', list(SSB_REJECTED), ids=list(SSB_REJECTED))
def test_ssb_layout_the_resampler_cannot_produce_is_rejected(cls, case):
    """`fourier.resample` and `correlate_sync_sequence` raise on these once they have
    IQ; the validator rejects them at the measurement path, naming the field"""
    cap, kwargs, message = SSB_REJECTED[case]
    spec = cls(**{**SSB, **kwargs})

    with pytest.raises(msgspec.ValidationError, match=re.escape(message)) as ex:
        measure(spec, cap)

    name = sa.registry[cls].name
    assert str(ex.value).startswith(f'$.{name}: ')


@pytest.mark.parametrize('cls', SSB_SPEC_TYPES, ids=SSB_IDS)
def test_ssb_downsampled_and_shifted_block_runs(cls):
    """the accepted counterpart of the rejections above: a 2:1 downsample with a
    frequency offset on the 50 Hz resampler grid reaches the correlator"""
    spec = cls(subcarrier_spacing=15e3, sample_rate=FS / 2, frequency_offset=100 * 50.0)
    data, _ = measure(spec, capture())
    assert np.all(np.isfinite(data))


# %% cellular_cyclic_autocorrelation

AUTOCORRELATION_REJECTED = {
    'symbol_range_past_the_slot': (
        {'symbol_range': (14, 15)},
        'symbol_range must index the 14 symbols of a slot, within [-14, 13]',
    ),
    'symbol_range_below_the_slot': (
        {'symbol_range': (-15, 0)},
        'symbol_range must index the 14 symbols of a slot, within [-14, 13]',
    ),
    'empty_symbol_range': (
        {'symbol_range': (0, 0)},
        'symbol_range must select at least one symbol',
    ),
    'no_downlink_slot': (
        {'frame_slots': 'u'},
        "frame_slots must include at least one downlink slot 'd'",
    ),
    'frame_range_past_the_capture': (
        {'frame_range': (1, 3)},
        'frame_range must index whole 10 ms frames inside the capture',
    ),
    'negative_frame': (
        {'frame_range': (-1, 1)},
        'frame_range must index whole 10 ms frames inside the capture',
    ),
    'empty_frame_range': (
        {'frame_range': (0, 0)},
        'frame_range must select at least one frame',
    ),
}


@pytest.mark.parametrize(
    'case', list(AUTOCORRELATION_REJECTED), ids=list(AUTOCORRELATION_REJECTED)
)
def test_autocorrelation_index_out_of_the_capture_is_rejected(case):
    """`Phy3GPP.index_cyclic_prefix` bounds the symbol axis but not the frame axis,
    whose indices past the capture the correlation kernel reads as zeros, and an
    all-uplink frame leaves the downlink correlation with no indices at all"""
    kwargs, message = AUTOCORRELATION_REJECTED[case]
    spec = sa.specs.CellularCyclicAutocorrelator(subcarrier_spacings=SCS, **kwargs)

    with pytest.raises(msgspec.ValidationError, match=re.escape(message)) as ex:
        measure(spec, capture())

    assert str(ex.value).startswith('$.cellular_cyclic_autocorrelation: ')


@pytest.mark.parametrize(
    'symbol_range', [(0, None), (0, 14), (13, 14), (-14, 0), 3], ids=str
)
def test_autocorrelation_symbol_range_within_the_slot_runs(symbol_range):
    """the accepted counterpart of the index bounds: negative indices count back from
    the end of the slot as `cp_start_idx` does"""
    spec = sa.specs.CellularCyclicAutocorrelator(
        subcarrier_spacings=SCS, frame_range=(0, 2), symbol_range=symbol_range
    )
    data, _ = measure(spec, capture())
    downlink = data[0, 0, 0]
    assert np.all(np.isfinite(downlink))


def test_autocorrelation_guards_a_waveform_shorter_than_its_capture():
    """the wrapper validates the capture, not the array, so a direct caller of the
    body with fewer samples than the capture declares is stopped before the kernel
    reads past the end"""
    body = sa.measurements.cellular_cyclic_autocorrelation.__wrapped__
    spec = sa.specs.CellularCyclicAutocorrelator(
        subcarrier_spacings=SCS, frame_range=(0, 2)
    )
    short_iq = testing.noise(DURATION / 2, FS, noise_psd=1 / FS)

    with pytest.raises(ValueError, match=r'lies past the .* samples of the waveform'):
        body(short_iq, capture(), **spec.to_dict())


# %% cellular_5g_ssb_spectrogram

SSB_SYMBOLS = round(28 * SSB_SPECTROGRAM_SCS / 15e3)


def ssb_of(duration, frequency=None, **kwargs):
    if frequency is None:
        iq = testing.noise(
            duration, SSB_SPECTROGRAM_FS, noise_psd=1 / SSB_SPECTROGRAM_FS
        )
    else:
        iq = testing.tone(duration, SSB_SPECTROGRAM_FS, frequency=frequency)
    return sa.measurements.cellular_5g_ssb_spectrogram(
        iq,
        ssb_spectrogram_capture(duration),
        as_xarray=True,
        **(SSB_SPECTROGRAM_SPEC.to_dict() | kwargs),
    )


def ssb_tone_level_dB() -> float:
    """level of a tone centered on a 15 kHz bin after integration to one 30 kHz bin.

    The STFT window is a boxcar of `window_fill` * nfft samples zero-padded to nfft,
    whose ENBW is nfft/L bins, so the tone bin alone reads 10*log10(L/nfft). Its
    Dirichlet kernel is not zero on the neighboring bins, and the integration sums
    the tone bin with one neighbor holding sinc(L/nfft)**2 of the tone's power.
    """
    _, window_fill = sa.measurements.cellular.cellular_stft_window_fractions('normal')
    nfft = round(2 * SSB_SPECTROGRAM_FS / SSB_SPECTROGRAM_SCS)
    L = round(window_fill * nfft)
    return 10 * np.log10(L / nfft * (1 + np.sinc(L / nfft) ** 2))


class TestCellular5GSSBSpectrogram:
    @pytest.mark.parametrize('blocks', [1, 2], ids='blocks{}'.format)
    def test_axis_shapes(self, blocks):
        da = ssb_of(blocks * SSB_SPECTROGRAM_PERIODICITY)

        assert da.sizes['cellular_ssb_index'] == blocks
        assert da.sizes['cellular_ssb_symbol_index'] == SSB_SYMBOLS
        assert list(da.cellular_ssb_symbol_index.values) == list(range(SSB_SYMBOLS))

        # the frequency axis is binned to one bin per subcarrier spacing and then
        # truncated to the SSB sample rate around frequency_offset
        freqs = da.cellular_ssb_baseband_frequency.values
        assert freqs.size == round(SSB_SPECTROGRAM_SAMPLE_RATE / SSB_SPECTROGRAM_SCS)
        assert_close(np.diff(freqs), SSB_SPECTROGRAM_SCS, rtol=RTOL_FLOAT64)
        assert np.all(np.abs(freqs) <= SSB_SPECTROGRAM_SAMPLE_RATE / 2)

    def test_tone_lands_in_its_labeled_frequency_bin_in_every_block(self):
        tone_frequency = SSB_SPECTROGRAM_SCS
        da = ssb_of(2 * SSB_SPECTROGRAM_PERIODICITY, frequency=tone_frequency)

        freqs = da.cellular_ssb_baseband_frequency.values
        expected = int(np.argmin(np.abs(freqs - tone_frequency)))
        spg = levels(da)
        assert np.all(np.argmax(spg, axis=-1) == expected)

        # a stationary tone reads the same in every symbol of every block
        peak = spg[..., expected]
        assert peak.max() == peak.min()

    def test_tone_level_sums_the_tone_bin_with_its_leaking_neighbor(self):
        duration = 2 * SSB_SPECTROGRAM_PERIODICITY
        da = ssb_of(duration, frequency=SSB_SPECTROGRAM_SCS)

        freqs = da.cellular_ssb_baseband_frequency.values
        index = int(np.argmin(np.abs(freqs - SSB_SPECTROGRAM_SCS)))
        tol = registered_tolerance(
            ssb_spectrogram_capture(duration), SSB_SPECTROGRAM_SPEC
        )
        assert_close(
            levels(da)[..., index],
            ssb_tone_level_dB(),
            rtol=tol.rtol,
            atol=tol.on_peak.peak,
        )

    @pytest.mark.parametrize(
        'duration,kwargs,message',
        [
            (
                SSB_SPECTROGRAM_PERIODICITY,
                {'frequency_offset': SSB_SPECTROGRAM_SCS / 2},
                'sample_rate must select a band of whole bins centered at frequency_offset',
            ),
            (
                SSB_SPECTROGRAM_PERIODICITY,
                {'sample_rate': 2 * SSB_SPECTROGRAM_FS},
                'sample_rate must select a band of whole bins',
            ),
            (
                # half a burst set (one slot) past the discovery period
                SSB_SPECTROGRAM_PERIODICITY + sw.ofdm.slot_period(SSB_SPECTROGRAM_SCS),
                {},
                'duration must end on a whole discovery_periodicity or after a complete burst set of 56 symbols',
            ),
        ],
        ids=[
            'frequency_offset_off_the_subcarrier_grid',
            'sample_rate_above_capture',
            'partial_burst_set',
        ],
    )
    def test_incompatible_burst_layout_is_rejected(self, duration, kwargs, message):
        """the frequency cut and the per-block reshape at the end of the measurement
        would fail on these; the validator names the field first"""
        with pytest.raises(msgspec.ValidationError, match=re.escape(message)) as ex:
            ssb_of(duration, **kwargs)

        assert str(ex.value).startswith('$.cellular_5g_ssb_spectrogram: ')


# %% cellular_resource_power_histogram

# the cellular resource grid needs sample_rate/(subcarrier_spacing/2) divisible by 28,
# because its spectrogram overlaps 13/28 of the window and fills 15/28 of it: at 15 kHz
# subcarriers 210 kHz gives the smallest such FFT (28 bins, hopping 1 symbol of 15
# samples). 2 ms covers 2 of the 10 slots in the frame, which is all the mask needs.
RESOURCE_GRID_FS = 210e3
RESOURCE_GRID_DURATION = 2e-3
RESOURCE_GRID_SCS = 15e3
RESOURCE_GRID_SLOTS = 10
RESOURCE_GRID_CAPTURE = sa.specs.Capture(
    duration=RESOURCE_GRID_DURATION,
    sample_rate=RESOURCE_GRID_FS,
    analysis_bandwidth=150e3,
)
RESOURCE_GRID_KWS = {
    'window': 'hamming',
    'subcarrier_spacing': RESOURCE_GRID_SCS,
    **POWER_BINS,
}

RTOL = elementwise_rtol(np.float32)


def resource_histogram(*, frame_slots, **kwargs):
    iq = sa.testing.tone(RESOURCE_GRID_DURATION, RESOURCE_GRID_FS)
    return sa.measurements.cellular_resource_power_histogram(
        iq,
        RESOURCE_GRID_CAPTURE,
        frame_slots=frame_slots,
        as_xarray=True,
        **RESOURCE_GRID_KWS,
        **kwargs,
    )


def test_cellular_resource_power_histogram_link_direction_coordinate():
    da = resource_histogram(frame_slots='d' * RESOURCE_GRID_SLOTS)

    assert tuple(da.link_direction.values) == ('downlink', 'uplink')


def test_cellular_resource_power_histogram_all_downlink_frame():
    """an all-downlink frame masks every uplink resource element with nan, which the
    histogram drops: the uplink row is exactly zero rather than nan, and the downlink
    row carries the whole fraction"""
    da = resource_histogram(frame_slots='d' * RESOURCE_GRID_SLOTS)

    downlink, uplink = da.values[0]
    assert (uplink == 0).all()
    assert_close(downlink.sum(), 1.0, rtol=RTOL)


def test_cellular_resource_power_histogram_splits_link_directions():
    """the normalization sums over link direction as well as power bin, so the two
    rows of a mixed frame share the fraction of 1 instead of each summing to 1"""
    da = resource_histogram(frame_slots='d' + 'u' * (RESOURCE_GRID_SLOTS - 1))

    downlink, uplink = da.values[0]
    assert downlink.sum() > 0 and uplink.sum() > 0
    assert downlink.sum() < 1 and uplink.sum() < 1
    assert_close(da.values[0].sum(), 1.0, rtol=RTOL)


def test_cellular_resource_power_histogram_guard_bandwidths():
    """a guard band masks the frequency bins at the edges of the analysis band with
    nan, which drops them from the histogram.

    A constant-envelope input makes every symbol of the grid identical, so each
    retained frequency bin contributes the same count and the smallest non-zero
    fraction is the reciprocal of the number of retained bins.
    """
    retained = []
    for guard_bandwidths in [(0, 0), (15e3, 15e3), (30e3, 30e3)]:
        da = resource_histogram(
            frame_slots='d' * RESOURCE_GRID_SLOTS, guard_bandwidths=guard_bandwidths
        )
        downlink = da.values[0, 0]
        retained.append(round(1 / downlink[downlink > 0].min()))

    assert retained[0] > retained[1] > retained[2]


@pytest.mark.parametrize(
    'capture,kwargs,message',
    [
        (
            RESOURCE_GRID_CAPTURE.replace(analysis_bandwidth=1.5 * RESOURCE_GRID_FS),
            {},
            'analysis_bandwidth must select a band of whole bins',
        ),
        (
            # a whole resource block spans 24 half-subcarrier bins, but the 105 kHz
            # analysis band keeps only 14 of the 28
            RESOURCE_GRID_CAPTURE.replace(analysis_bandwidth=RESOURCE_GRID_FS / 2),
            {'average_rbs': True},
            'integration_bandwidth must not exceed the analyzed bandwidth',
        ),
        (
            # 13 symbols: one short of the slot that average_slots reduces over
            RESOURCE_GRID_CAPTURE.replace(duration=1e-3),
            {'average_slots': True},
            'duration must span at least one slot to average across slots',
        ),
    ],
    ids=[
        'analysis_bandwidth_above_sample_rate',
        'resource_block_wider_than_analysis_band',
        'capture_shorter_than_a_slot',
    ],
)
def test_cellular_resource_power_histogram_rejects_undersized_grids(
    capture, kwargs, message
):
    """`truncate_freqs` and both `binned_mean` calls fail on these grids with
    array-shape errors; the validator rejects them first at the measurement path"""
    iq = sa.testing.tone(capture.duration, RESOURCE_GRID_FS)
    with pytest.raises(msgspec.ValidationError, match=message) as ex:
        sa.measurements.cellular_resource_power_histogram(
            iq,
            capture,
            frame_slots='d' * RESOURCE_GRID_SLOTS,
            **RESOURCE_GRID_KWS,
            **kwargs,
        )

    assert str(ex.value).startswith('$.cellular_resource_power_histogram: ')
