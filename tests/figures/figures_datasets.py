"""synthetic datasets in the shape of a saved sweep store for the figures tests.

Each dataset has 3 captures with a calibrated `system_noise` (dBm/Hz). The power
histograms are the dB density of a Gamma-distributed bin power, with the mode of each
capture at the predicted noise level plus `HISTOGRAM_OFFSET_DB`. The third capture has
a higher noise figure and a mode 5 dB above the prediction, as with a signal present.
Fractions below `MIN_FRACTION` are zero.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

SYSTEM_NOISE_DBM_HZ = np.array([-170.0, -170.0, -160.0])
ANALYSIS_BANDWIDTH = np.array([1e6, np.inf, 1e6])
SAMPLE_RATE = 2e6
CENTER_FREQUENCY = np.array([1e9, 1e9, 2e9])
HISTOGRAM_BANDWIDTH = 1e4
CORRELATOR_BANDWIDTH = 30e3
HISTOGRAM_OFFSET_DB = np.array([0.0, 0.0, 5.0])
GAMMA_SHAPE = 24
POWER_DETECTORS = ['rms', 'peak']
CYCLIC_STATISTICS = ['min', 'mean', 'max']
CYCLIC_LAG_COUNT = 4
DETECTOR_PERIOD = 1e-3
PEAK_OFFSET_DB = 10.0
CYCLIC_BOUND_DB = 3.0

# one count in 1e6, above the float32 resolution of a cumulative fraction near 1, so
# every nonzero bin registers in a float32 cumsum
MIN_FRACTION = 1e-6

POWER_BINS = np.concatenate([[-np.inf], np.arange(-150.0, 0.0001, 0.1), [np.inf]])


def predicted_levels_dBm(
    bandwidth: float | np.ndarray = HISTOGRAM_BANDWIDTH,
) -> np.ndarray:
    """return the system noise (dBm) of each capture in `bandwidth` (Hz)."""
    return SYSTEM_NOISE_DBM_HZ + 10 * np.log10(bandwidth)


def capture_bandwidth() -> np.ndarray:
    """return the noise bandwidth (Hz) of each capture for a dBm output without an attr.

    The bandwidth is the analysis bandwidth, or the sample rate where the analysis
    bandwidth is infinite.
    """
    return np.where(np.isfinite(ANALYSIS_BANDWIDTH), ANALYSIS_BANDWIDTH, SAMPLE_RATE)


def cyclic_mean_dBm() -> np.ndarray:
    """return the mean cyclic channel power (dBm) of each capture and detector.

    The ``rms`` detector is at the system noise in the capture bandwidth, and the
    ``peak`` detector `PEAK_OFFSET_DB` above. The array has shape
    ``(capture, power_detector)``.
    """
    offsets = np.array([
        PEAK_OFFSET_DB if d == 'peak' else 0.0 for d in POWER_DETECTORS
    ])
    return predicted_levels_dBm(capture_bandwidth())[:, np.newaxis] + offsets


def gamma_dB_histogram(level_dBm: float) -> np.ndarray:
    """return the `POWER_BINS` fractions of a Gamma(`GAMMA_SHAPE`) power histogram.

    The mean power is `level_dBm`, and the density of the dB value peaks at the dB of
    the mean. Fractions below `MIN_FRACTION` are zero, and the remainder sum to 1.
    """
    finite = np.isfinite(POWER_BINS)
    u = (POWER_BINS[finite] - level_dBm) * np.log(10) / 10
    density = np.exp(GAMMA_SHAPE * (u - np.exp(u)))
    fractions = np.zeros(POWER_BINS.size)
    fractions[finite] = density / density.sum()
    fractions[fractions < MIN_FRACTION] = 0.0
    return fractions / fractions.sum()


def occupied_bins(fractions: np.ndarray) -> np.ndarray:
    """return the `POWER_BINS` values (dBm) with nonzero fraction in one histogram."""
    return POWER_BINS[fractions > 0]


def make_dataset(attr_missing: bool = False, system_noise: bool = True) -> xr.Dataset:
    """return a dataset in the shape of a saved store.

    Args:
        attr_missing: True gives the histogram bin coordinate ``dBm/<bandwidth>``
            units and no `noise_bandwidth` attr
        system_noise: False omits the `system_noise` variable
    """
    levels = predicted_levels_dBm() + HISTOGRAM_OFFSET_DB
    hist = np.stack([gamma_dB_histogram(level) for level in levels])

    stat_offsets = np.array([-CYCLIC_BOUND_DB, 0.0, CYCLIC_BOUND_DB])
    cyclic = (
        cyclic_mean_dBm()[:, :, np.newaxis, np.newaxis]
        + stat_offsets[np.newaxis, np.newaxis, :, np.newaxis]
        + np.zeros(CYCLIC_LAG_COUNT)
    )

    if attr_missing:
        bin_attrs = {'standard_name': 'Spectrogram bin power', 'units': 'dBm/10 kHz'}
    else:
        bin_attrs = {'units': 'dBm', 'noise_bandwidth': HISTOGRAM_BANDWIDTH}

    ds = xr.Dataset(
        {
            'spectrogram_histogram': (
                ('capture', 'spectrogram_power_bin'),
                hist.astype('float32'),
                {
                    'standard_name': 'Fraction of counts',
                    'noise_bandwidth': HISTOGRAM_BANDWIDTH,
                },
            ),
            'channel_power_histogram': (
                ('capture', 'channel_power_bin'),
                hist.astype('float32'),
                {'standard_name': 'Fraction of channel power readings'},
            ),
            'channel_power_time_series': (
                ('capture', 'time_elapsed'),
                np.zeros((3, 4), 'float32'),
                {'standard_name': 'Channel power', 'units': 'dBm'},
            ),
            'spectrogram_ratio_histogram': (
                ('capture', 'spectrogram_ratio_power_bin'),
                np.zeros((3, 5), 'float32'),
                {'standard_name': 'Fraction of counts'},
            ),
            'cyclic_channel_power': (
                ('capture', 'power_detector', 'cyclic_statistic', 'cyclic_lag'),
                cyclic.astype('float32'),
                {'standard_name': 'Cyclic channel power', 'units': 'dBm'},
            ),
            'cellular_5g_pss_correlation': (
                ('capture', 'cellular_ssb_lag'),
                np.zeros((3, 4), 'complex64'),
                {
                    'standard_name': 'PSS Cross-Covariance',
                    'units': '√mW',
                    'noise_bandwidth': CORRELATOR_BANDWIDTH,
                },
            ),
        },
        coords={
            'spectrogram_power_bin': ('spectrogram_power_bin', POWER_BINS, bin_attrs),
            'channel_power_bin': (
                'channel_power_bin',
                POWER_BINS,
                {'standard_name': 'Channel power', 'units': 'dBm'},
            ),
            'spectrogram_ratio_power_bin': (
                'spectrogram_ratio_power_bin',
                np.arange(5.0),
                {
                    'standard_name': 'Spectrogram cross-channel power ratio',
                    'units': 'dB',
                },
            ),
            'power_detector': (
                'power_detector',
                np.array(POWER_DETECTORS, dtype=object),
            ),
            'cyclic_statistic': (
                'cyclic_statistic',
                np.array(CYCLIC_STATISTICS, dtype=object),
            ),
            'cyclic_lag': (
                'cyclic_lag',
                np.arange(CYCLIC_LAG_COUNT) * DETECTOR_PERIOD,
                {'standard_name': 'Cyclic lag', 'units': 's'},
            ),
            'analysis_bandwidth': ('capture', ANALYSIS_BANDWIDTH),
            'sample_rate': ('capture', np.full(3, SAMPLE_RATE)),
            'center_frequency': ('capture', CENTER_FREQUENCY),
        },
    )
    if system_noise:
        ds['system_noise'] = (
            ('capture',),
            SYSTEM_NOISE_DBM_HZ.astype('float32'),
            {'standard_name': 'System noise', 'units': 'dBm/Hz'},
        )
    return ds


def cyclic_statistics(
    ds: xr.Dataset | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """return the (min, mean, max) cyclic channel power of `ds`, or of `make_dataset()`"""
    da = (make_dataset() if ds is None else ds)['cyclic_channel_power']
    return tuple(da.sel(cyclic_statistic=s) for s in ('min', 'mean', 'max'))
