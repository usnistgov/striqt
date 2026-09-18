"""the power-level-selective measurements: channel power, spectrogram, spectrogram
ratio and cellular resource grid histograms.

The inputs are constant-envelope tones from `striqt.analysis.testing`, whose mean
power of 1 reads exactly 0 dBm, scaled by ``10**(level/20)`` to move the reading to a
known level. A constant envelope puts every reading of a measurement into one power
bin, which makes the expected fraction exactly 1 rather than a tolerance.

The last cell holds the cross-measurement contract shared by all ten in-scope
measurements; the spectral and power test modules do not repeat it.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from numeric_checks import assert_close, elementwise_rtol

import striqt.analysis as sa
from striqt.analysis.measurements._channel_power_histogram import make_power_bins

FS = 1e6
DURATION = 1e-3
DETECTOR_PERIOD = Fraction(1, 10000)
FREQUENCY_RESOLUTION = 50e3
NFFT = round(FS / FREQUENCY_RESOLUTION)

# a 1 dB grid on [-40, 10] dBm, so a whole-number level in dB sits on a bin center
POWER_BINS = {'power_low': -40.0, 'power_high': 10.0, 'power_resolution': 1.0}

CAPTURE = sa.specs.Capture(duration=DURATION, sample_rate=FS)

RTOL = elementwise_rtol(np.float32)

# the cellular resource grid needs sample_rate/(subcarrier_spacing/2) divisible by 28,
# because its spectrogram overlaps 13/28 of the window and fills 15/28 of it: at 15 kHz
# subcarriers 210 kHz gives the smallest such FFT (28 bins, hopping 1 symbol of 15
# samples). 2 ms covers 2 of the 10 slots in the frame, which is all the mask needs.
CELL_FS = 210e3
CELL_DURATION = 2e-3
CELL_SUBCARRIER_SPACING = 15e3
CELL_SLOTS = 10
CELL_CAPTURE = sa.specs.Capture(
    duration=CELL_DURATION, sample_rate=CELL_FS, analysis_bandwidth=150e3
)
CELL_KWS = {
    'window': 'hamming',
    'subcarrier_spacing': CELL_SUBCARRIER_SPACING,
    **POWER_BINS,
}


def populated(fractions, bins):
    """the (bin, fraction) pairs of the non-zero entries of one histogram row"""
    index = np.nonzero(np.asarray(fractions))
    return list(zip(bins[index], np.asarray(fractions)[index]))


# %% channel_power_histogram


@pytest.mark.parametrize('level', [-13.0, 0.0, 7.0], ids='level{:g}dB'.format)
def test_channel_power_histogram_constant_level_fills_one_bin(level):
    """every detector reading of a constant-envelope tone is the same level, so the
    whole normalized fraction lands in the bin centered on it"""
    iq = sa.testing.tone(DURATION, FS) * 10 ** (level / 20)

    da = sa.measurements.channel_power_histogram(
        iq, CAPTURE, detector_period=DETECTOR_PERIOD, as_xarray=True, **POWER_BINS
    )

    bins = da.channel_power_bin.values
    for detector in range(da.sizes['power_detector']):
        assert populated(da.values[0, detector], bins) == [(level, 1.0)]


def test_channel_power_histogram_fractions_sum_to_one():
    """the counts are normalized so that each (port, detector) row sums to 1, even
    though the sum in the source runs over both detectors at once"""
    # one ramp across the whole capture, so each detector period reads a level of its
    # own rather than repeating one bin
    iq = sa.testing.sawtooth(DURATION, FS, period=DURATION, ports=2)

    da = sa.measurements.channel_power_histogram(
        iq, CAPTURE, detector_period=DETECTOR_PERIOD, as_xarray=True, **POWER_BINS
    )

    assert (da.values > 0).sum(axis=-1).min() > 1, 'expected a spread of bins'
    assert_close(da.values.sum(axis=-1), np.ones((2, 2)), rtol=RTOL)


@pytest.mark.parametrize(
    'level,expected_bin',
    [(20.0, float('inf')), (-60.0, float('-inf'))],
    ids=['above_power_high', 'below_power_low'],
)
def test_channel_power_histogram_catches_out_of_range_levels(level, expected_bin):
    iq = sa.testing.tone(DURATION, FS) * 10 ** (level / 20)

    da = sa.measurements.channel_power_histogram(
        iq, CAPTURE, detector_period=DETECTOR_PERIOD, as_xarray=True, **POWER_BINS
    )

    for detector in range(da.sizes['power_detector']):
        assert populated(da.values[0, detector], da.channel_power_bin.values) == [
            (expected_bin, 1.0)
        ]


@pytest.mark.parametrize(
    'power_low,power_high,power_resolution',
    [(-40.0, 10.0, 1.0), (-40.0, 10.0, 3.0), (-20.0, 0.0, 0.5)],
    ids=['step1', 'step3_off_grid_high', 'step0.5'],
)
def test_channel_power_histogram_bin_coordinate(
    power_low, power_high, power_resolution
):
    iq = sa.testing.tone(DURATION, FS)
    bins = {
        'power_low': power_low,
        'power_high': power_high,
        'power_resolution': power_resolution,
    }

    da = sa.measurements.channel_power_histogram(
        iq, CAPTURE, detector_period=DETECTOR_PERIOD, as_xarray=True, **bins
    )

    expected = make_power_bins(power_low, power_high, power_resolution)
    assert_close(da.channel_power_bin.values, expected, rtol=RTOL)
    assert da.channel_power_bin.values[0] == float('-inf')
    assert da.channel_power_bin.values[-1] == float('inf')


# %% spectrogram_histogram


def test_spectrogram_histogram_fractions_sum_to_one():
    """both ports are normalized by the count of port 0, which is the same count"""
    iq = sa.testing.noise(DURATION, FS, noise_psd=1e-6, ports=2)

    da = sa.measurements.spectrogram_histogram(
        iq,
        CAPTURE,
        window='hamming',
        frequency_resolution=FREQUENCY_RESOLUTION,
        as_xarray=True,
        **POWER_BINS,
    )

    assert (da.values > 0).sum(axis=-1).min() > 1, 'expected a spread of bins'
    assert_close(da.values.sum(axis=-1), np.ones(2), rtol=RTOL)


def test_spectrogram_histogram_concentrates_bin_centered_tone():
    """a tone on an FFT bin center, taken through a rectangular window, puts all of
    its power in 1 of the `NFFT` bins of every STFT window and exactly 0 in the rest.

    The populated bin is therefore 0 dBm -- the whole power of the tone -- rather than
    a level referred to the 50 kHz noise bandwidth that the units attr reports, and
    the empty bins fall in the -inf catch-all.

    The fractions are 19/20 and 1/20 (`NFFT` is 20, not a power of two, so these are
    not dyadic), computed here in float32 to match the measurement's registered
    dtype, so they are compared with a float32-appropriate tolerance rather than
    exact equality.
    """
    iq = sa.testing.tone(DURATION, FS, frequency=2 * FREQUENCY_RESOLUTION)

    da = sa.measurements.spectrogram_histogram(
        iq,
        CAPTURE,
        window='boxcar',
        frequency_resolution=FREQUENCY_RESOLUTION,
        as_xarray=True,
        **POWER_BINS,
    )

    assert da.attrs['noise_bandwidth'] == FREQUENCY_RESOLUTION
    bins, fractions = zip(*populated(da.values[0], da.spectrogram_power_bin.values))
    assert bins == (float('-inf'), 0.0)
    assert_close(fractions, ((NFFT - 1) / NFFT, 1 / NFFT), rtol=RTOL)


@pytest.mark.parametrize(
    'integration_bandwidth,units',
    [(None, 'dBm/50 kHz'), (100e3, 'dBm/100 kHz')],
    ids=['no_integration', 'integrate_100kHz'],
)
def test_spectrogram_histogram_bin_coordinate(integration_bandwidth, units):
    """the bin coordinate is the shared power grid, labeled with the equivalent noise
    bandwidth that the readings are referred to"""
    iq = sa.testing.tone(DURATION, FS)

    da = sa.measurements.spectrogram_histogram(
        iq,
        CAPTURE,
        window='hamming',
        frequency_resolution=FREQUENCY_RESOLUTION,
        integration_bandwidth=integration_bandwidth,
        as_xarray=True,
        **POWER_BINS,
    )

    expected = make_power_bins(**POWER_BINS)
    assert_close(da.spectrogram_power_bin.values, expected, rtol=RTOL)
    assert da.spectrogram_power_bin.attrs['units'] == units


# %% spectrogram_ratio_histogram


def ratio_histogram(iq, **kwargs):
    return sa.measurements.spectrogram_ratio_histogram(
        iq,
        CAPTURE,
        window='hamming',
        frequency_resolution=FREQUENCY_RESOLUTION,
        as_xarray=True,
        **POWER_BINS,
        **kwargs,
    )


def two_ports_offset_by(offset_dB):
    """noise repeated on 2 ports, with port 1 scaled up by `offset_dB`.

    Noise rather than a tone so that every spectrogram bin carries power well above
    the float32 roundoff floor: in the near-empty bins of a tone's spectrogram the
    cross-port ratio is roundoff noise rather than the applied offset.
    """
    iq = sa.testing.noise(DURATION, FS, noise_psd=1e-6, ports=2)
    iq[1] = iq[0] * 10 ** (offset_dB / 20)
    return iq


def test_spectrogram_ratio_histogram_identical_ports():
    da = ratio_histogram(two_ports_offset_by(0.0))

    bins = da.spectrogram_ratio_power_bin.values
    for port in range(2):
        assert populated(da.values[port], bins) == [(0.0, 1.0)]


@pytest.mark.parametrize('offset_dB', [6.0, -7.0], ids='offset{:g}dB'.format)
def test_spectrogram_ratio_histogram_offset_ports(offset_dB):
    """row 0 holds spg[0]-spg[1] and row 1 holds spg[1]-spg[0], so a level offset
    between the ports puts the two rows at opposite signs of it"""
    da = ratio_histogram(two_ports_offset_by(offset_dB))

    bins = da.spectrogram_ratio_power_bin.values
    assert populated(da.values[0], bins) == [(-offset_dB, 1.0)]
    assert populated(da.values[1], bins) == [(offset_dB, 1.0)]


@pytest.mark.parametrize('ports', [1, 3], ids='ports{}'.format)
def test_spectrogram_ratio_histogram_requires_two_ports(ports):
    iq = sa.testing.tone(DURATION, FS, ports=ports)

    with pytest.raises(ValueError, match='only supported for 2-channel measurements'):
        ratio_histogram(iq)


def test_spectrogram_ratio_histogram_bin_units_are_ratios():
    """the bins hold a cross-port ratio, so their units are dB rather than the dBm of
    the absolute spectrogram histogram"""
    da = ratio_histogram(two_ports_offset_by(0.0))

    units = da.spectrogram_ratio_power_bin.attrs['units']
    assert units == 'dB/50 kHz'


# %% cellular_resource_power_histogram


def resource_histogram(*, frame_slots, **kwargs):
    iq = sa.testing.tone(CELL_DURATION, CELL_FS)
    return sa.measurements.cellular_resource_power_histogram(
        iq, CELL_CAPTURE, frame_slots=frame_slots, as_xarray=True, **CELL_KWS, **kwargs
    )


def test_cellular_resource_power_histogram_link_direction_coordinate():
    da = resource_histogram(frame_slots='d' * CELL_SLOTS)

    assert tuple(da.link_direction.values) == ('downlink', 'uplink')


def test_cellular_resource_power_histogram_all_downlink_frame():
    """an all-downlink frame masks every uplink resource element with nan, which the
    histogram drops: the uplink row is exactly zero rather than nan, and the downlink
    row carries the whole fraction"""
    da = resource_histogram(frame_slots='d' * CELL_SLOTS)

    downlink, uplink = da.values[0]
    assert (uplink == 0).all()
    assert_close(downlink.sum(), 1.0, rtol=RTOL)


def test_cellular_resource_power_histogram_splits_link_directions():
    """the normalization sums over link direction as well as power bin, so the two
    rows of a mixed frame share the fraction of 1 instead of each summing to 1"""
    da = resource_histogram(frame_slots='d' + 'u' * (CELL_SLOTS - 1))

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
            frame_slots='d' * CELL_SLOTS, guard_bandwidths=guard_bandwidths
        )
        downlink = da.values[0, 0]
        retained.append(round(1 / downlink[downlink > 0].min()))

    assert retained[0] > retained[1] > retained[2]


# %% the contract shared by every measurement

CONTRACT_CAPTURE = sa.specs.Capture(
    duration=4e-3, sample_rate=CELL_FS, analysis_bandwidth=150e3
)

# 4 ms at 210 kHz is 1 discovery period of 56 symbols at 15 kHz subcarriers, whose
# first 28 are the SSB burst set, and a whole number of detector and cyclic periods
CONTRACT_SPECS = (
    sa.specs.Spectrogram(window='hamming', frequency_resolution=7.5e3),
    sa.specs.PowerSpectralDensity(window='hamming', frequency_resolution=7.5e3),
    sa.specs.Cellular5GNRSSBSpectrogram(
        subcarrier_spacing=CELL_SUBCARRIER_SPACING,
        sample_rate=CELL_FS / 2,
        discovery_periodicity=CONTRACT_CAPTURE.duration,
    ),
    sa.specs.ChannelPowerTimeSeries(detector_period=DETECTOR_PERIOD),
    sa.specs.CyclicChannelPower(cyclic_period=1e-3, detector_period=DETECTOR_PERIOD),
    sa.specs.IQWaveform(),
    sa.specs.ChannelPowerHistogram(detector_period=DETECTOR_PERIOD, **POWER_BINS),
    sa.specs.SpectrogramHistogram(
        window='hamming', frequency_resolution=7.5e3, **POWER_BINS
    ),
    sa.specs.SpectrogramHistogramRatio(
        window='hamming', frequency_resolution=7.5e3, **POWER_BINS
    ),
    sa.specs.CellularResourcePowerHistogram(**CELL_KWS),
)

SPEC_IDS = [type(spec).__name__ for spec in CONTRACT_SPECS]


def measure(spec, as_xarray):
    """run the measurement registered for `type(spec)` the way the registry calls it"""
    info = sa.registry[type(spec)]
    ports = 2 if isinstance(spec, sa.specs.SpectrogramHistogramRatio) else 1
    iq = sa.testing.tone(
        CONTRACT_CAPTURE.duration, CONTRACT_CAPTURE.sample_rate, ports=ports
    )
    return info, info.func(iq, CONTRACT_CAPTURE, as_xarray=as_xarray, **spec.to_dict())


def registered_dims(info):
    dims = ['port']
    for factory in info.coord_factories:
        for dim in sa.registry.coordinates[factory].dims:
            if dim not in dims:
                dims.append(dim)
    return tuple(dims)


@pytest.mark.parametrize('spec', CONTRACT_SPECS, ids=SPEC_IDS)
def test_measurement_dtype_and_dims(spec):
    info, da = measure(spec, as_xarray=True)

    assert da.dtype == info.dtype
    if info.dims is None:
        assert da.dims == registered_dims(info)
    else:
        assert da.dims == ('port',) + info.dims
    assert set(da.coords) == set(da.dims) - {'port'}


@pytest.mark.parametrize('spec', CONTRACT_SPECS, ids=SPEC_IDS)
def test_measurement_attrs(spec):
    """the registered attrs and every spec field reach the DataArray, which is what
    names them in a saved zarr store"""
    info, da = measure(spec, as_xarray=True)

    for name, value in info.attrs.items():
        assert da.attrs[name] == value

    # the wrapper decodes the keywords into a spec before calling, so the attrs hold
    # the validated field values rather than the ones passed in
    for name, value in spec.validate().to_dict().items():
        assert da.attrs[name] == value


@pytest.mark.parametrize('spec', CONTRACT_SPECS, ids=SPEC_IDS)
def test_measurement_without_xarray(spec):
    _, (data, attrs) = measure(spec, as_xarray=False)
    _, da = measure(spec, as_xarray=True)

    assert isinstance(attrs, dict)
    data = np.asarray(data)
    assert data.shape == da.shape
    # the raw array is not held to the registered dtype: power_spectral_density
    # returns float16 where it registers float32
    assert np.array_equal(data.astype(da.dtype), da.values, equal_nan=True)
