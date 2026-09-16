"""striqt.sensor.lib.compute.corrections: sizing of the extra acquisition overlap that
the resampler and the analysis filter consume"""

from __future__ import annotations

import pytest

import striqt.sensor as ss
from striqt.sensor.lib.compute import corrections

MCR = 125e6
SOURCE = ss.specs.SoapySource(master_clock_rate=MCR)


def _capture(**kws):
    kws = {
        'port': 0,
        'center_frequency': 1e9,
        'gain': 0,
        'duration': 10e-3,
        'sample_rate': MCR,
        **kws,
    }
    return ss.specs.SoapyCapture(**kws)


# %% get_correction_overlaps


def _assert_valid_overlaps(capture):
    low, high = corrections.get_correction_overlaps(capture, SOURCE)
    filter_pad = corrections._get_filter_overlap(capture)
    assert low >= filter_pad and high >= filter_pad
    assert low > 0 and high > 0


@pytest.mark.parametrize('sample_rate', [MCR, 62.5e6, 15.36e6])
def test_infinite_bandwidth_overlaps(sample_rate):
    _assert_valid_overlaps(_capture(sample_rate=sample_rate))


def test_finite_bandwidth_without_host_resampling():
    _assert_valid_overlaps(_capture(analysis_bandwidth=40e6, host_resample=False))


def test_finite_bandwidth_with_a_large_resampler_fft():
    # fs_sdr 15.625 MS/s -> 15.36 MS/s designs a 6250-point FFT, larger than the
    # filter overlap, so the block sizing works out regardless of the ceildiv order
    _assert_valid_overlaps(_capture(sample_rate=15.36e6, analysis_bandwidth=10e6))


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='_get_resample_overlap swaps the ceildiv arguments, so a small resampler '
    'FFT gives a block size of one FFT rather than enough to cover the filter '
    'overlap; the pad assertions in _get_resample_overlap/_get_resampler_overlaps '
    'then fail for finite analysis_bandwidth with host_resample',
)
@pytest.mark.parametrize(
    'sample_rate, analysis_bandwidth',
    [(MCR, 40e6), (1e6, 0.5e6), (10e6, 8e6)],
)
def test_finite_bandwidth_with_a_small_resampler_fft(sample_rate, analysis_bandwidth):
    _assert_valid_overlaps(
        _capture(sample_rate=sample_rate, analysis_bandwidth=analysis_bandwidth)
    )
