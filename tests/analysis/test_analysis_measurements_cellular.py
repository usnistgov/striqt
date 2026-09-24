"""the cellular synchronization and cyclic prefix measurements:
cellular_5g_pss_correlation, cellular_5g_sss_correlation, cellular_5g_pss_sync,
cellular_5g_sss_sync and cellular_cyclic_autocorrelation.

`tests/waveform/test_waveform_ofdm.py` proves the correlators and the cyclic prefix
kernel numerically. What the measurement layer adds, and what is tested here, is the
(capture, spec) validation that rejects a combination before acquisition: the
resampler's grid and band limits and the 10 ms frame count for the correlators, and
the frame, slot and symbol index bounds for the autocorrelation.

Two frames at 3.84 MS/s is the smallest capture at which a 30 kHz cell search (case
C, 3GPP TS 38.213 Section 4.1) and a two-frame `frame_range` both fit.
"""

from __future__ import annotations

import re

import msgspec
import numpy as np
import pytest

import striqt.analysis as sa
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
