"""striqt.figures.util: placing the calibrated system noise on power outputs"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from figures_datasets import (
    CORRELATOR_BANDWIDTH,
    HISTOGRAM_BANDWIDTH,
    HISTOGRAM_OFFSET_DB,
    POWER_BINS,
    capture_bandwidth,
    make_dataset,
    occupied_bins,
    predicted_levels_dBm,
)

from striqt.figures import util

# %% noise_bandwidth and get_system_noise


@pytest.mark.parametrize(
    ('var_name', 'attr_missing', 'bandwidth', 'long_name'),
    [
        ('spectrogram_histogram', False, HISTOGRAM_BANDWIDTH, 'dBm/10 kHz'),
        ('spectrogram_power_bin', False, HISTOGRAM_BANDWIDTH, 'dBm/10 kHz'),
        ('spectrogram_power_bin', True, HISTOGRAM_BANDWIDTH, 'dBm/10 kHz'),
        ('cellular_5g_pss_correlation', False, CORRELATOR_BANDWIDTH, 'dBm/30 kHz'),
    ],
    ids=['variable_attr', 'coordinate_attr', 'coordinate_without_attr', 'sqrt_mW_attr'],
)
def test_system_noise_in_the_noise_bandwidth_attr(
    var_name: str, attr_missing: bool, bandwidth: float, long_name: str
) -> None:
    """the level is the system noise in the `noise_bandwidth` of the variable.

    A bin coordinate without the attr and with ``dBm/<bandwidth>`` units resolves to
    the `noise_bandwidth` of the histogram variable on the same dimension.
    """
    ds = make_dataset(attr_missing=attr_missing)

    noise = util.get_system_noise(ds, var_name)

    assert noise is not None
    assert noise.dims == ('capture',)
    np.testing.assert_allclose(noise.values, predicted_levels_dBm(bandwidth), atol=1e-4)
    assert noise.attrs['units'] == 'dBm'
    assert noise.attrs['noise_bandwidth'] == bandwidth
    assert noise.attrs['long_name'] == f'System noise ({long_name})'


@pytest.mark.parametrize('var_name', ['channel_power_time_series', 'channel_power_bin'])
def test_dBm_variable_without_the_attr_in_the_capture_bandwidth(var_name: str) -> None:
    """a dBm output without `noise_bandwidth` resolves to the analysis bandwidth of
    each capture, or to the sample rate where the analysis bandwidth is infinite"""
    ds = make_dataset()

    noise = util.get_system_noise(ds, var_name)

    assert noise is not None
    np.testing.assert_allclose(
        noise.values, predicted_levels_dBm(capture_bandwidth()), atol=1e-4
    )
    assert noise.attrs['units'] == 'dBm'
    assert 'noise_bandwidth' not in noise.attrs


@pytest.mark.parametrize(
    'var_name', ['spectrogram_ratio_histogram', 'spectrogram_ratio_power_bin']
)
def test_ratios_have_no_system_noise(var_name: str) -> None:
    assert util.noise_bandwidth(make_dataset(), var_name) is None
    assert util.get_system_noise(make_dataset(), var_name) is None


def test_no_calibration_gives_none() -> None:
    ds = make_dataset(system_noise=False)
    assert util.get_system_noise(ds, 'spectrogram_histogram') is None
    assert util.system_noise_floor(ds, 'spectrogram_histogram', 6) is None


@pytest.mark.parametrize(
    'func',
    [util.system_noise_floor, util.get_system_noise],
    ids=['system_noise_floor', 'get_system_noise'],
)
def test_margin_returns_the_minimum_over_captures(func: Callable) -> None:
    ds = make_dataset()
    floor = func(ds, 'spectrogram_power_bin', 12)
    assert floor == pytest.approx(predicted_levels_dBm().min() - 12, abs=1e-4)


# %% histogram_bin_limits and select_histogram_bins


def test_histogram_bin_limits_are_per_capture() -> None:
    """the lower limit is the noise level of each capture minus `pad_low`, and the
    upper limit is the highest occupied bin of any capture plus `pad_hi`"""
    ds = make_dataset()
    hist = ds.spectrogram_histogram.values

    low, high = util.histogram_bin_limits(
        ds, 'spectrogram_histogram', 'spectrogram_power_bin', pad_low=12, pad_hi=3
    )

    assert low is not None
    np.testing.assert_allclose(low.values, predicted_levels_dBm() - 12, atol=1e-4)
    top = max(occupied_bins(fractions).max() for fractions in hist)
    assert high == pytest.approx(top + 3, abs=1e-6)


def test_select_histogram_bins_keeps_every_capture_in_range() -> None:
    """the selected bins contain the occupied bins of every capture. The selection
    extends 12 dB below the lowest noise level and 3 dB above the highest occupied
    bin, and excludes the catch-all bins."""
    ds = make_dataset()
    hist = ds.spectrogram_histogram.values

    sub = util.select_histogram_bins(
        ds, 'spectrogram_histogram', 'spectrogram_power_bin'
    )

    bins = sub.spectrogram_power_bin.values
    assert np.all(np.isfinite(bins))
    for fractions in hist:
        assert np.isin(occupied_bins(fractions), bins).all()
    lowest = predicted_levels_dBm().min() - 12
    top = max(occupied_bins(fractions).max() for fractions in hist) + 3
    assert bins.min() == POWER_BINS[POWER_BINS >= lowest].min()
    assert bins.max() == POWER_BINS[POWER_BINS <= top].max()


# %% check_noise_floor


@pytest.mark.parametrize('attr_missing', [False, True], ids=['attr', 'attr_missing'])
def test_check_noise_floor_offset_equals_the_mode_shift(attr_missing: bool) -> None:
    """`offset_dB` is the shift of the histogram mode above the predicted level"""
    ds = make_dataset(attr_missing=attr_missing)

    check = util.check_noise_floor(ds, 'spectrogram_histogram')

    assert set(check.data_vars) == {'observed_dBm', 'predicted_dBm', 'offset_dB'}
    assert check.offset_dB.dims == ('capture',)
    np.testing.assert_allclose(
        check.predicted_dBm.values, predicted_levels_dBm(), atol=1e-4
    )
    np.testing.assert_allclose(check.offset_dB.values, HISTOGRAM_OFFSET_DB, atol=0.1)
    assert np.all(np.isfinite(check.observed_dBm.values))


def test_check_noise_floor_predicts_from_the_bin_coordinate() -> None:
    """the prediction for a channel power histogram without attrs is the system noise
    in the capture bandwidth of the dBm bin coordinate"""
    ds = make_dataset()

    check = util.check_noise_floor(ds, 'channel_power_histogram')

    np.testing.assert_allclose(
        check.predicted_dBm.values, predicted_levels_dBm(capture_bandwidth()), atol=1e-4
    )


def test_check_noise_floor_rejects_variables_without_a_floor() -> None:
    ds = make_dataset()
    with pytest.raises(ValueError, match='power_bin'):
        util.check_noise_floor(ds, 'channel_power_time_series')
    with pytest.raises(ValueError, match='predict the floor'):
        util.check_noise_floor(ds, 'spectrogram_ratio_histogram')
