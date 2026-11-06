"""striqt.analysis.measurements.cellular: the 5G NR synchronization, cyclic prefix, and
resource grid measurements.

`tests/waveform/test_waveform_comms_sync_5gnr.py` and
`tests/waveform/test_waveform_comms_cp.py` prove the correlators and the cyclic prefix
kernel numerically. These tests cover the (capture, spec) validation the measurement
layer adds, and the coordinates and levels of the spectrogram products. Two frames at
3.84 MS/s is the smallest capture in which a 30 kHz cell search (case C, 3GPP TS 38.213
Section 4.1) and a 2-frame `frame_range` both fit.
"""

from __future__ import annotations

import math
import re

import msgspec
import numpy as np
import pytest
from analysis_strategies import (
    CELL_FRAME,
    CELL_FS,
    POWER_BINS,
    RTOL,
    SSB_SPECTROGRAM_FS,
    SSB_SPECTROGRAM_PERIODICITY,
    SSB_SPECTROGRAM_SAMPLE_RATE,
    SSB_SPECTROGRAM_SCS,
    SSB_SPECTROGRAM_SPEC,
    SSB_WINDOW_FILL,
    capture_of,
    registered_tolerance,
    ssb_spectrogram_capture,
)
from numeric_checks import RTOL_FLOAT64, assert_close, levels
from numpy.testing import assert_array_equal

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis import testing
from striqt.analysis.measurements import cellular

DURATION = 2 * CELL_FRAME / CELL_FS
SCS = 30e3
SSB = {'subcarrier_spacing': SCS, 'sample_rate': CELL_FS, 'symbol_indexes': 'c'}


def capture(
    samples: int = 2 * CELL_FRAME, sample_rate: float = CELL_FS
) -> sa.specs.Capture:
    return capture_of(samples, sample_rate)


def measure(spec, cap: sa.specs.Capture):
    """run the measurement registered for `spec` on unit-power white noise over `cap`,
    with ``as_xarray=False``."""
    info = sa.registry[type(spec)]
    iq = testing.circular_awgn(cap.duration, cap.sample_rate, power=1.0)
    return info.func(iq, cap, as_xarray=False, **spec.to_dict())


# %% the 5G NR SSB correlators and synchronizers

# the PSS and SSS correlators and synchronizers share one validator, so the PSS
# correlator carries every case
SSB_REJECTED = {
    'odd_sample_count': (
        capture(2 * CELL_FRAME + 1),
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
        {
            'subcarrier_spacing': 15e3,
            'sample_rate': CELL_FS / 2,
            'frequency_offset': CELL_FS / 2,
        },
        'the capture cannot be resampled to the synchronization block: shift is too large',
    ),
    'frequency_offset_without_downsampling': (
        capture(),
        # `resample` shifts only while downsampling
        {'frequency_offset': 50.0},
        'the capture cannot be resampled to the synchronization block',
    ),
    'partial_frame': (
        capture(2 * CELL_FRAME + CELL_FRAME // 2),
        {},
        'duration must hold whole 10 ms frames for the correlator',
    ),
    'block_shorter_than_a_frame': (
        capture(CELL_FRAME, sample_rate=2 * CELL_FS),
        {'sample_rate': CELL_FS / 2, 'subcarrier_spacing': 15e3},
        'duration must hold whole 10 ms frames for the correlator',
    ),
}


@pytest.mark.parametrize('case', list(SSB_REJECTED), ids=list(SSB_REJECTED))
def test_ssb_layout_the_resampler_cannot_produce_is_rejected(case):
    """`fourier.resample` and `correlate_sync_sequence` raise on these layouts once
    given IQ. The validator rejects each layout at the measurement path and names the
    field."""
    cap, kwargs, message = SSB_REJECTED[case]
    spec = sa.specs.Cellular5GNRPSSCorrelator(**{**SSB, **kwargs})

    with pytest.raises(msgspec.ValidationError, match=re.escape(message)) as ex:
        measure(spec, cap)

    assert str(ex.value).startswith('$.cellular_5g_pss_correlation: ')


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
    """`Phy3GPP.index_cyclic_prefix` bounds the symbol axis but not the frame axis.
    The correlation kernel reads frame indices past the capture as zeros. An
    all-uplink frame leaves the downlink correlation with no indices."""
    kwargs, message = AUTOCORRELATION_REJECTED[case]
    spec = sa.specs.CellularCyclicAutocorrelator(subcarrier_spacings=SCS, **kwargs)

    with pytest.raises(msgspec.ValidationError, match=re.escape(message)) as ex:
        measure(spec, capture())

    assert str(ex.value).startswith('$.cellular_cyclic_autocorrelation: ')


def test_autocorrelation_guards_a_waveform_shorter_than_its_capture():
    """the wrapper validates the capture, not the array, so a direct call of the body
    with fewer samples than the capture declares raises before the kernel reads past
    the end"""
    body = sa.measurements.cellular_cyclic_autocorrelation.__wrapped__
    spec = sa.specs.CellularCyclicAutocorrelator(
        subcarrier_spacings=SCS, frame_range=(0, 2)
    )
    short_iq = testing.circular_awgn(DURATION / 2, CELL_FS, power=1.0)

    with pytest.raises(ValueError, match=r'lies past the .* samples of the waveform'):
        body(short_iq, capture(), **spec.to_dict())


# %% cellular_5g_ssb_spectrogram

SSB_SYMBOLS = round(28 * SSB_SPECTROGRAM_SCS / 15e3)


def ssb_of(duration, frequency=None, **kwargs):
    if frequency is None:
        iq = testing.circular_awgn(duration, SSB_SPECTROGRAM_FS, power=1.0)
    else:
        iq = testing.tone(duration, SSB_SPECTROGRAM_FS, frequency=frequency)
    return sa.measurements.cellular_5g_ssb_spectrogram(
        iq,
        ssb_spectrogram_capture(duration),
        as_xarray=True,
        **(SSB_SPECTROGRAM_SPEC.to_dict() | kwargs),
    )


def ssb_tone_level_dB() -> float:
    """return the level (dB) of a tone centered on a 15 kHz bin after integration to
    one 30 kHz bin.

    The STFT window is a boxcar of L = `SSB_WINDOW_FILL` * nfft samples zero-padded
    to nfft. The ENBW of the window is nfft/L bins, so the tone bin alone is at
    10*log10(L/nfft). The Dirichlet kernel of the window is not zero on the
    neighboring bins. The integration sums the tone bin with one neighbor at
    sinc(L/nfft)**2 of the tone power.
    """
    nfft = round(2 * SSB_SPECTROGRAM_FS / SSB_SPECTROGRAM_SCS)
    L = round(SSB_WINDOW_FILL * nfft)
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

        # a stationary tone has the same level in every symbol of every block
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
                SSB_SPECTROGRAM_PERIODICITY
                + sw.comms_5gnr.slot_period(SSB_SPECTROGRAM_SCS),
                {},
                'duration must end on a whole discovery_periodicity or after a complete burst set of 56 symbols',
            ),
            (
                # a discovery period of one slot is shorter than the 2 ms burst set
                SSB_SPECTROGRAM_PERIODICITY,
                {
                    'discovery_periodicity': sw.comms_5gnr.slot_period(
                        SSB_SPECTROGRAM_SCS
                    )
                },
                'discovery_periodicity must span the burst set of 56 symbols',
            ),
        ],
        ids=[
            'frequency_offset_off_the_subcarrier_grid',
            'sample_rate_above_capture',
            'partial_burst_set',
            'discovery_period_shorter_than_the_burst_set',
        ],
    )
    def test_incompatible_burst_layout_is_rejected(self, duration, kwargs, message):
        """the frequency cut and the per-block reshape at the end of the measurement
        fail on these layouts. The validator rejects each layout first and names the
        field."""
        with pytest.raises(msgspec.ValidationError, match=re.escape(message)) as ex:
            ssb_of(duration, **kwargs)

        assert str(ex.value).startswith('$.cellular_5g_ssb_spectrogram: ')


# %% noise bandwidth of the SSB measurements


def test_pss_correlation_noise_bandwidth_is_one_subcarrier():
    """the unit-energy PSS template passes one subcarrier of white noise per lag, so
    |R|^2 of white noise averages N0 * subcarrier_spacing. The `noise_bandwidth` attr
    equals `subcarrier_spacing`."""
    cap = capture()
    spec = sa.specs.Cellular5GNRPSSCorrelator(**SSB)
    R, metadata = measure(spec, cap)

    assert metadata['units'] == sa.units.AMPLITUDE
    assert metadata['noise_bandwidth'] == SCS

    # the awgn has unit power over CELL_FS. The last lags of each block are zero
    psd = 1.0 / CELL_FS
    power = np.abs(np.asarray(R)) ** 2
    lags = power.shape[-1]
    mean = float(power[..., : lags - 32].mean())
    sigma = 1 / math.sqrt(power[..., : lags - 32].size)
    assert_close(
        10 * np.log10(mean),
        10 * np.log10(psd * SCS),
        atol=sw.power_analysis.level_tolerance_dB(5 * sigma, power=True),
    )


def test_ssb_spectrogram_white_noise_reads_one_subcarrier():
    """for white noise, the sum of 2 half-subcarrier bins has a mean of
    N0 * subcarrier_spacing. The `resolution_bandwidth` attr is that of the boxcar
    over the `window_fill` samples."""
    periods = 10
    da = ssb_of(periods * SSB_SPECTROGRAM_PERIODICITY)

    assert da.attrs['noise_bandwidth'] == SSB_SPECTROGRAM_SCS
    nfft = round(2 * SSB_SPECTROGRAM_FS / SSB_SPECTROGRAM_SCS)
    nwindow = round(SSB_WINDOW_FILL * nfft)
    assert da.attrs['resolution_bandwidth'] == pytest.approx(
        SSB_SPECTROGRAM_FS / nwindow
    )

    psd = 1.0 / SSB_SPECTROGRAM_FS
    linear = 10 ** (da.values.astype('float64') / 10)
    sigma = math.sqrt(2 / linear.size)
    assert_close(
        10 * np.log10(np.nanmean(linear)),
        10 * np.log10(psd * SSB_SPECTROGRAM_SCS),
        atol=sw.power_analysis.level_tolerance_dB(5 * sigma, power=True),
    )


# %% cellular_resource_power_histogram

# the cellular resource grid requires sample_rate/(subcarrier_spacing/2) divisible by
# 28, because the spectrogram overlaps 13/28 of the window and fills 15/28 of the
# window. At 15 kHz subcarriers, 210 kHz gives the smallest such FFT (28 bins, hopping
# 1 symbol of 15 samples). 2 ms covers 2 of the 10 slots in a frame, and the slot mask
# needs no more.
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


def resource_histogram(*, frame_slots, **kwargs):
    x = sa.testing.tone(RESOURCE_GRID_DURATION, RESOURCE_GRID_FS)
    iq = sa.AcquiredIQ(x, RESOURCE_GRID_CAPTURE, power_units='mW')
    return sa.measurements.cellular_resource_power_histogram(
        iq,
        RESOURCE_GRID_CAPTURE,
        frame_slots=frame_slots,
        as_xarray=True,
        **RESOURCE_GRID_KWS,
        **kwargs,
    )


def test_apply_mask_nulls_the_guard_band_and_keeps_downlink_symbols():
    rng = np.random.default_rng(0)
    spg = rng.uniform(1e-12, 1e-9, (1, 28, 14)).astype('float32')
    freqs = np.fft.fftshift(np.fft.fftfreq(14, 1 / RESOURCE_GRID_FS))

    out = cellular.apply_mask(
        spg,
        freqs,
        subcarrier_spacing=RESOURCE_GRID_SCS,
        link_direction=('downlink', 'uplink'),
        channel_bandwidth=150e3,
        frame_slots='dddddddddd',
        special_symbols=None,
        guard_left=RESOURCE_GRID_SCS,
        guard_right=RESOURCE_GRID_SCS,
        xp=np,
    )

    assert np.isnan(out[..., 0]).all(), 'the low guard band is nulled in the output'
    assert np.isfinite(out[0, 0, :, 7]).all(), 'downlink symbols at 0 Hz are kept'


def test_cellular_resource_power_histogram_leaves_the_cached_spectrogram_intact():
    """a `spectrogram` measurement with the sizing's spectrogram spec reads the same
    cache entry after the histogram, so that entry must stay free of guard-band NaN"""
    x = sa.testing.tone(RESOURCE_GRID_DURATION, RESOURCE_GRID_FS)
    spec = sa.specs.CellularResourcePowerHistogram(
        frame_slots='dddddddddd',
        guard_bandwidths=(RESOURCE_GRID_SCS, RESOURCE_GRID_SCS),
        **RESOURCE_GRID_KWS,
    )
    sizing = cellular.validated_resource_grid_sizing(RESOURCE_GRID_CAPTURE, spec)

    with cellular.spectrum.spectrogram_cache:
        cached, _ = cellular.spectrum.evaluate_spectrogram(
            x, RESOURCE_GRID_CAPTURE, sizing.spectrogram, dtype='float32', dB=False
        )
        before = cached.copy()
        sa.measurements.cellular_resource_power_histogram(
            x, RESOURCE_GRID_CAPTURE, **spec.to_dict()
        )
        after, _ = cellular.spectrum.evaluate_spectrogram(
            x, RESOURCE_GRID_CAPTURE, sizing.spectrogram, dtype='float32', dB=False
        )

    assert after is cached, 'the measurement and this test share one cache entry'
    assert_array_equal(after, before)


@pytest.mark.parametrize(
    'average_rbs,resource_blocks', [(False, 1 / 12), ('half', 1 / 2)]
)
def test_cellular_resource_power_histogram_bin_coordinate_bandwidth(
    average_rbs, resource_blocks
):
    """the bin coordinate and the histogram have the noise bandwidth of one bin:
    a subcarrier, or the 6 subcarriers averaged per half resource block. A full block
    exceeds the 150 kHz analysis bandwidth of the test capture."""
    da = resource_histogram(frame_slots='dddddddddd', average_rbs=average_rbs)

    bandwidth = 12 * RESOURCE_GRID_SCS * resource_blocks
    coord_attrs = da.cellular_resource_power_bin.attrs
    assert coord_attrs['units'] == 'dBm'
    assert coord_attrs['noise_bandwidth'] == pytest.approx(bandwidth)
    assert da.attrs['noise_bandwidth'] == pytest.approx(bandwidth)
    assert da.attrs['resolution_bandwidth'] == coord_attrs['resolution_bandwidth']
    assert 'units' not in da.attrs


# the 2 ms capture spans 2 slots of 14 symbols, but the STFT (28-sample window hopping
# 15 samples over 420 samples) yields 27 windows, so the second slot has 13 symbols
RESOURCE_GRID_SYMBOLS = 27


@pytest.mark.parametrize(
    ('frame_slots', 'expected'),
    [
        ('d' * RESOURCE_GRID_SLOTS, (1.0, 0.0)),
        (
            'd' + 'u' * (RESOURCE_GRID_SLOTS - 1),
            (14 / RESOURCE_GRID_SYMBOLS, 13 / RESOURCE_GRID_SYMBOLS),
        ),
    ],
    ids=['all_downlink', 'downlink_then_uplink'],
)
def test_cellular_resource_power_histogram_splits_the_fraction_by_link_direction(
    frame_slots, expected
):
    """the normalization sums over link direction as well as power bin, so the
    (downlink, uplink) rows share the fraction of 1 in the ratio of the symbols that
    `frame_slots` assigns to each. A link direction without symbols is exactly 0."""
    da = resource_histogram(frame_slots=frame_slots)

    assert tuple(da.link_direction.values) == ('downlink', 'uplink')
    assert_close(da.values[0].sum(axis=-1), expected, rtol=RTOL)


# the 150 kHz analysis band spans 10 subcarriers, and `binned_mean` keeps the odd
# count of 9 whole bins centered on DC: -60 kHz to 60 kHz. A guard nulls the bins
# within its width of each band edge, and the retained band is half-open at the upper
# edge: [-60, 60) kHz keeps 8 subcarriers with 15 kHz guards and [-45, 45) kHz keeps 6
# with 30 kHz guards
RETAINED_SUBCARRIERS = {(0, 0): 9, (15e3, 15e3): 8, (30e3, 30e3): 6}


@pytest.mark.parametrize(
    'guard_bandwidths',
    list(RETAINED_SUBCARRIERS),
    ids=['no_guard', 'guard15kHz', 'guard30kHz'],
)
def test_cellular_resource_power_histogram_guard_bandwidths(guard_bandwidths):
    """a guard band masks the frequency bins at the edges of the analysis band with
    nan, and the histogram drops nan.

    A constant-envelope input makes every symbol of the grid identical, so each
    retained frequency bin contributes the same count and the smallest non-zero
    fraction is the reciprocal of the retained bin count.
    """
    da = resource_histogram(
        frame_slots='d' * RESOURCE_GRID_SLOTS, guard_bandwidths=guard_bandwidths
    )

    downlink = da.values[0, 0]
    smallest = downlink[downlink > 0].min()
    assert_close(smallest, 1 / RETAINED_SUBCARRIERS[guard_bandwidths], rtol=RTOL)


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
            # 13 symbols: one short of the slot over which average_slots reduces
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
    array-shape errors. The validator rejects each grid first, at the measurement
    path."""
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
