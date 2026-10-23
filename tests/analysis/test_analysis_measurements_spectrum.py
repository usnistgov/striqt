"""striqt.analysis.measurements.spectrum: the spectrogram-derived measurements.

Level assertions are anchored on a bin-centered unit tone: a boxcar window reads
exactly ``0.0`` dB in the peak bin, and any other window reads
``-10*log10(enbw_bins)``. The spectrogram is normalized to a power spectral density
labeled with ``noise_bandwidth = frequency_resolution`` while the tone's power spreads
over the window's equivalent noise bandwidth.
"""

from __future__ import annotations

import fractions
import math
import re

import msgspec
import numpy as np
import pytest
from analysis_strategies import (
    POWER_BINS,
    SSB_SPECTROGRAM_PERIODICITY,
    SSB_SPECTROGRAM_SPEC,
    registered_tolerance,
    ssb_spectrogram_capture,
)
from numeric_checks import (
    ATOL,
    RTOL_FLOAT64,
    assert_close,
    elementwise_rtol,
    levels,
    populated,
)

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis import testing
from striqt.waveform.fourier import fft_tolerance_rms, off_peak_floor_dBc

# a 64-point FFT over 8 non-overlapping windows: 512 samples, 500 us
FS = 1.024e6
RES = 16e3
NFFT = round(FS / RES)
NWINDOW = 8
DURATION = NWINDOW * NFFT / FS

CAPTURE = sa.specs.Capture(duration=DURATION, sample_rate=FS)

# roundoff of the one FFT that separates the IQ from the reported spectrum
FFT_SIGMA = fft_tolerance_rms(np.complex64, [NFFT])


def bin_centered_tone_level_dB(window, nfft=NFFT):
    """dB level of the peak bin of a bin-centered unit tone, per the module docstring"""
    enbw_bins = sw.equivalent_noise_bandwidth(window, nfft)
    return -10 * np.log10(float(enbw_bins))


def centered_bin_count(size, count):
    """number of `count`-wide bins that `sw.binned_mean(..., fft=True)` keeps.

    It centers one bin on DC and keeps an odd number of them, so the partial bins at
    both edges of the frequency axis are dropped rather than the axis being divided
    evenly.
    """
    return 2 * ((size // 2 - count // 2) // count) + 1


# %% spectrogram

WINDOWS = ['boxcar', 'hamming', 'hann', 'blackmanharris']


def spg_spec(**kwargs) -> sa.specs.Spectrogram:
    kwargs.setdefault('window', 'boxcar')
    kwargs.setdefault('frequency_resolution', RES)
    return sa.specs.Spectrogram(**kwargs)


def spg_of(iq, capture=CAPTURE, **kwargs):
    kwargs.setdefault('window', 'boxcar')
    kwargs.setdefault('frequency_resolution', RES)
    return sa.measurements.spectrogram(iq, capture, **kwargs)


class TestSpectrogram:
    @pytest.mark.parametrize('window', WINDOWS, ids=WINDOWS)
    def test_bin_centered_tone_level(self, window):
        tone_bin = 5
        iq = testing.tone(DURATION, FS, frequency=tone_bin * RES)
        arr, metadata = spg_of(iq, window=window, as_xarray=False)

        spg = levels(arr)
        assert metadata['noise_bandwidth'] == RES
        expected = bin_centered_tone_level_dB(window)
        peak_index = NFFT // 2 + tone_bin

        tol = registered_tolerance(CAPTURE, spg_spec(window=window))
        assert_close(
            spg[:, :, peak_index],
            expected,
            rtol=tol.rtol,
            atol=tol.on_peak.peak,
            err_msg=f'{window} peak bin level',
        )
        assert np.argmax(spg, axis=-1).tolist() == [[peak_index] * NWINDOW]

    def test_off_peak_bins_hold_only_roundoff(self):
        tone_bin = 5
        iq = testing.tone(DURATION, FS, frequency=tone_bin * RES)
        arr, _ = spg_of(iq, as_xarray=False)

        spg = levels(arr)
        peak_index = NFFT // 2 + tone_bin
        others = np.delete(spg, peak_index, axis=-1)

        # a boxcar window puts the whole tone in its on-peak bin, so its immediate
        # neighbors read only FFT roundoff referred to the 0 dB tone, like every
        # other off-peak bin
        off_peak_floor = off_peak_floor_dBc(FFT_SIGMA, NFFT, size=others.shape[-1])
        assert spg[:, :, [peak_index - 1, peak_index + 1]].max() < off_peak_floor
        assert others.max() < off_peak_floor

    def test_frequency_coordinate_is_the_fftfreq_grid(self):
        iq = testing.tone(DURATION, FS)
        da = spg_of(iq, as_xarray=True)
        freqs = da.spectrogram_baseband_frequency.values
        assert_close(freqs, sw.fftfreq(NFFT, FS), rtol=RTOL_FLOAT64, atol=ATOL)

    @pytest.mark.parametrize('trim_stopband', [True, False], ids='trim{}'.format)
    def test_trim_stopband_selects_the_analysis_bandwidth(self, trim_stopband):
        bandwidth = FS / 2
        capture = CAPTURE.replace(analysis_bandwidth=bandwidth)
        iq = testing.tone(DURATION, FS)
        da = spg_of(iq, capture, trim_stopband=trim_stopband, as_xarray=True)

        freqs = da.spectrogram_baseband_frequency.values
        if trim_stopband:
            assert freqs.size == round(NFFT * bandwidth / FS)
            assert np.all(freqs >= -bandwidth / 2) and np.all(freqs < bandwidth / 2)
        else:
            assert freqs.size == NFFT
        assert da.shape[-1] == freqs.size

    @pytest.mark.parametrize('lo_bandstop', [RES, 4 * RES], ids='bw{:g}'.format)
    def test_lo_bandstop_nulls_only_the_bins_at_dc(self, lo_bandstop):
        iq = testing.circular_awgn(DURATION, FS, power=1.0)
        da = spg_of(iq, lo_bandstop=lo_bandstop, as_xarray=True)

        freqs = da.spectrogram_baseband_frequency.values
        # sw.fourier.null_lo nulls the half-open band [-bw/2, +bw/2)
        expected = (freqs >= -lo_bandstop / 2) & (freqs < lo_bandstop / 2)
        assert expected.sum() == round(NFFT * lo_bandstop / FS)

        isnan = np.isnan(levels(da))
        assert np.array_equal(isnan, np.broadcast_to(expected, isnan.shape))

    def test_binning_reduces_both_axes_and_sums_over_frequency(self):
        tone_bin = 5
        frequency_bins = 4
        time_bins = 2
        iq = testing.tone(DURATION, FS, frequency=tone_bin * RES)

        binning = {
            'integration_bandwidth': frequency_bins * RES,
            'time_aperture': time_bins * NFFT / FS,
        }
        reference, ref_metadata = spg_of(iq, as_xarray=False)
        da = spg_of(iq, as_xarray=True, **binning)

        assert da.sizes['spectrogram_time'] == NWINDOW // time_bins
        assert da.sizes['spectrogram_baseband_frequency'] == centered_bin_count(
            NFFT, frequency_bins
        )

        freqs = da.spectrogram_baseband_frequency.values
        assert_close(np.diff(freqs), frequency_bins * RES, rtol=RTOL_FLOAT64)

        # the frequency bins are summed, not averaged, so a tone that fits inside one
        # of them keeps the level it had at the finer resolution while the bandwidth
        # it is referred to grows
        assert ref_metadata['noise_bandwidth'] == RES
        assert ref_metadata['units'] == 'dBm'
        assert ref_metadata['long_name'] == 'Power spectral density (dBm/16 kHz)'
        assert da.attrs['noise_bandwidth'] == frequency_bins * RES
        assert da.attrs['units'] == 'dBm'
        assert da.attrs['long_name'] == 'Power spectral density (dBm/64 kHz)'

        # each side has its own quantization and roundoff budget
        atol = (
            registered_tolerance(CAPTURE, spg_spec()).on_peak.peak
            + registered_tolerance(CAPTURE, spg_spec(**binning)).on_peak.peak
        )
        assert_close(levels(da).max(), levels(reference).max(), atol=atol)

    @pytest.mark.parametrize(
        'kwargs,message,quantities',
        [
            (
                {'time_aperture': 1.5 * NFFT / FS},
                (
                    'time_aperture must be a multiple of '
                    '(1-fractional_overlap)/frequency_resolution'
                ),
                # no overlap, so one hop is the whole window
                [
                    f'time_aperture: {1.5 * NFFT / FS}',
                    'fractional_overlap: 0',
                    f'frequency_resolution: {RES}',
                    f'hop_period: {NFFT / FS}',
                ],
            ),
            (
                {'integration_bandwidth': 1.5 * RES},
                'integration_bandwidth must be a multiple of frequency_resolution',
                [
                    f'integration_bandwidth: {1.5 * RES}',
                    f'frequency_resolution: {RES}',
                ],
            ),
            (
                {'frequency_resolution': 1.5 * RES},
                'sample_rate/resolution must be a counting number',
                [f'sample_rate: {FS}', f'frequency_resolution: {1.5 * RES}'],
            ),
            (
                {'window_fill': fractions.Fraction(1, 3)},
                (
                    '(1-window_fill) * sample_rate must be a counting-number '
                    'multiple of frequency_resolution'
                ),
                [
                    'window_fill: 1/3',
                    f'sample_rate: {FS}',
                    f'frequency_resolution: {RES}',
                    f'(1-window_fill)*nfft: {fractions.Fraction(2, 3) * NFFT}',
                ],
            ),
            (
                {'frequency_resolution': FS / 7, 'lo_bandstop': FS / 7},
                'lo_bandstop must select a band of whole bins centered at baseband DC',
                # an odd nfft puts DC between two bins
                [f'lo_bandstop: {FS / 7}', f'sample_rate: {FS}', '7-bin'],
            ),
            (
                {'integration_bandwidth': 2 * FS},
                'integration_bandwidth must not exceed the analyzed bandwidth',
                [f'integration_bandwidth: {2 * FS}', f'frequency bins: {NFFT}'],
            ),
            (
                {'time_aperture': 2 * DURATION},
                'duration must span at least one time_aperture of STFT windows',
                [f'time_aperture: {2 * DURATION}', f'STFT windows: {NWINDOW}'],
            ),
        ],
        ids=[
            'time_aperture',
            'integration_bandwidth',
            'frequency_resolution',
            'window_fill',
            'lo_bandstop_on_odd_nfft',
            'integration_bandwidth_above_sample_rate',
            'time_aperture_longer_than_capture',
        ],
    )
    def test_non_integer_binning_raises(self, kwargs, message, quantities):
        """the sizing validator rejects these before any IQ is touched, so the error
        is a msgspec.ValidationError that names the measurement's field path"""
        iq = testing.tone(DURATION, FS)
        with pytest.raises(
            msgspec.ValidationError, match=re.escape(message)
        ) as excinfo:
            spg_of(iq, as_xarray=False, **kwargs)

        assert str(excinfo.value).startswith('$.spectrogram: ')

        # the validator itself raises the bare error; the field path is attached by
        # whichever caller has the surrounding context
        spec_fields = {'window': 'boxcar', 'frequency_resolution': RES, **kwargs}
        with pytest.raises(ValueError, match=re.escape(message)) as valueinfo:
            sa.measurements.spectrum.validated_spectrogram_sizing(
                CAPTURE, sa.specs.Spectrogram(**spec_fields)
            )

        # the validator is the only layer that knows which quantities the rule
        # compared, so it names them in a parenthetical after the rule text
        for quantity in quantities:
            assert quantity in str(valueinfo.value)

    @pytest.mark.parametrize(
        'capture,message',
        [
            (
                sa.specs.Capture(duration=(NFFT - 1) / FS, sample_rate=FS),
                'duration must span at least one FFT window',
            ),
            (
                CAPTURE.replace(analysis_bandwidth=1.5 * FS),
                'analysis_bandwidth must select a band of whole bins',
            ),
        ],
        ids=['shorter_than_nfft', 'analysis_bandwidth_above_sample_rate'],
    )
    def test_incompatible_capture_raises(self, capture, message):
        """`sw.stft` and `sw.fourier.truncate_freqs` reject these with array-shape
        messages once IQ is in hand; the validator names the capture field"""
        iq = testing.tone(capture.duration, FS)
        with pytest.raises(msgspec.ValidationError, match=message) as excinfo:
            spg_of(iq, capture, as_xarray=False)

        assert str(excinfo.value).startswith('$.spectrogram: ')

    def test_odd_nfft_cannot_be_trimmed_to_the_analysis_bandwidth(self):
        """with 7 bins, DC falls between 2 of them, so the half-open band that
        `trim_stopband` keeps has no whole-bin edges"""
        capture = CAPTURE.replace(analysis_bandwidth=FS / 2)
        iq = testing.tone(DURATION, FS)
        with pytest.raises(msgspec.ValidationError, match='analysis_bandwidth'):
            spg_of(iq, capture, frequency_resolution=FS / 7, as_xarray=False)

        # the same nfft passes once the band is not cut
        da = spg_of(iq, capture, frequency_resolution=FS / 7, trim_stopband=False)
        assert da.sizes['spectrogram_baseband_frequency'] == 7


# %% noise bandwidth and resolution bandwidth

# 128 windows of white noise: enough exponential variates that a 5-sigma band on the
# mean bin power is a few tenths of a dB, well inside the 2.7 dB shift that a
# window_fill of 1/2 would produce if the zero padding entered the normalization
NOISE_DURATION = 128 * NFFT / FS
NOISE_CAPTURE = sa.specs.Capture(duration=NOISE_DURATION, sample_rate=FS)
NOISE_PSD = 1e-6


def noise_level_tolerance_dB(window, bins, windows, window_fill=1.0) -> float:
    """5-sigma bound (in dB) on the mean of `windows` x `bins` noise bin powers.

    Adjacent bins are correlated over the window's ENBW, which multiplies the
    variance of the mean over bins by the ENBW in bins.
    """
    nwindow = round(window_fill * NFFT)
    enbw_bins = float(sw.equivalent_noise_bandwidth(window, nwindow))
    sigma = math.sqrt(enbw_bins / (bins * windows))
    return sw.power_analysis.level_tolerance_dB(5 * sigma, power=True)


def mean_linear_power(da) -> float:
    return float(np.nanmean(10 ** (da.values.astype('float64') / 10)))


@pytest.mark.parametrize('window', WINDOWS, ids=WINDOWS)
@pytest.mark.parametrize(
    'window_fill', [1, fractions.Fraction(1, 2)], ids=['fill1', 'fill1/2']
)
def test_spectrogram_white_noise_reads_noise_bandwidth(window, window_fill):
    """white noise with power spectral density N0 has a mean bin power of
    N0 * noise_bandwidth for every window and window fill, and the stored attr equals
    frequency_resolution"""
    iq = testing.circular_awgn(NOISE_DURATION, FS, power=NOISE_PSD * FS)
    da = spg_of(
        iq,
        capture=NOISE_CAPTURE,
        window=window,
        window_fill=window_fill,
        as_xarray=True,
    )

    assert da.attrs['noise_bandwidth'] == RES
    assert da.attrs['units'] == 'dBm'
    windows, bins = (
        da.sizes['spectrogram_time'],
        da.sizes['spectrogram_baseband_frequency'],
    )
    assert_close(
        10 * np.log10(mean_linear_power(da)),
        10 * np.log10(NOISE_PSD * RES),
        atol=noise_level_tolerance_dB(window, bins, windows, float(window_fill)),
    )


def test_integrated_spectrogram_white_noise_reads_integration_bandwidth():
    """the sum of 4 bins has 4 times the noise power, and noise_bandwidth is 4 times
    the resolution"""
    frequency_bins = 4
    iq = testing.circular_awgn(NOISE_DURATION, FS, power=NOISE_PSD * FS)
    da = spg_of(
        iq,
        capture=NOISE_CAPTURE,
        window='hann',
        integration_bandwidth=frequency_bins * RES,
        as_xarray=True,
    )

    assert da.attrs['noise_bandwidth'] == frequency_bins * RES
    windows, bins = (
        da.sizes['spectrogram_time'],
        da.sizes['spectrogram_baseband_frequency'],
    )
    assert_close(
        10 * np.log10(mean_linear_power(da)),
        10 * np.log10(NOISE_PSD * frequency_bins * RES),
        atol=noise_level_tolerance_dB('hann', bins * frequency_bins, windows),
    )


def test_power_spectral_density_white_noise_reads_noise_bandwidth():
    """the Welch mean of white noise is N0 * noise_bandwidth per bin"""
    iq = testing.circular_awgn(NOISE_DURATION, FS, power=NOISE_PSD * FS)
    da = psd_of(iq, capture=NOISE_CAPTURE, window='hamming', time_statistic=['mean'])

    assert da.attrs['noise_bandwidth'] == RES
    assert da.attrs['units'] == 'dBm'
    assert da.attrs['long_name'] == 'Power spectral density (dBm/16 kHz)'
    windows = round(NOISE_DURATION * FS / NFFT)
    assert_close(
        10 * np.log10(mean_linear_power(da)),
        10 * np.log10(NOISE_PSD * RES),
        atol=noise_level_tolerance_dB('hamming', NFFT, windows),
    )


@pytest.mark.parametrize('window', WINDOWS, ids=WINDOWS)
@pytest.mark.parametrize(
    'window_fill', [1, fractions.Fraction(1, 2)], ids=['fill1', 'fill1/2']
)
def test_resolution_bandwidth_sets_the_tone_level(window, window_fill):
    """a bin-centered tone has frequency_resolution / resolution_bandwidth of the tone
    power in the tone bin, where the attr is the ENBW in Hz of the nonzero part of the
    window"""
    nwindow = round(window_fill * NFFT)
    enbw_bins = float(sw.equivalent_noise_bandwidth(window, nwindow))
    expected_rbw = enbw_bins * FS / nwindow

    tone_bin = 5
    iq = testing.tone(DURATION, FS, frequency=tone_bin * RES)
    da = spg_of(iq, window=window, window_fill=window_fill, as_xarray=True)

    assert da.attrs['resolution_bandwidth'] == pytest.approx(expected_rbw)
    assert da.spectrogram_baseband_frequency.attrs.get('resolution_bandwidth') is None

    peak_dB = float(da.max())
    tol = registered_tolerance(
        CAPTURE, spg_spec(window=window, window_fill=window_fill)
    )
    assert_close(
        peak_dB,
        10 * np.log10(RES / da.attrs['resolution_bandwidth']),
        atol=tol.on_peak.peak,
    )


# %% power_spectral_density

PSD_STATISTICS = ('min', 0.5, 'mean', 'max')
# 16 windows so that the quantile has something to interpolate between
PSD_DURATION = 16 * NFFT / FS
PSD_CAPTURE = sa.specs.Capture(duration=PSD_DURATION, sample_rate=FS)


def psd_spec(**kwargs) -> sa.specs.PowerSpectralDensity:
    kwargs.setdefault('window', 'boxcar')
    kwargs.setdefault('frequency_resolution', RES)
    return sa.specs.PowerSpectralDensity(**kwargs)


def psd_of(iq, capture=PSD_CAPTURE, **kwargs):
    kwargs.setdefault('window', 'boxcar')
    kwargs.setdefault('frequency_resolution', RES)
    return sa.measurements.power_spectral_density(iq, capture, as_xarray=True, **kwargs)


class TestPowerSpectralDensity:
    def test_time_statistic_coordinate_and_ordering(self):
        iq = testing.single_tone(PSD_DURATION, FS, frequency_offset=5 * RES, snr=20)
        da = psd_of(iq, time_statistic=PSD_STATISTICS)

        names = list(da.time_statistic.values)
        assert names == [str(s) for s in PSD_STATISTICS]

        sel = {name: levels(da.sel(time_statistic=name)) for name in names}
        slack = registered_tolerance(
            PSD_CAPTURE, psd_spec(time_statistic=PSD_STATISTICS)
        ).on_peak.peak
        assert np.all(sel['0.5'] >= sel['min'] - slack)
        assert np.all(sel['0.5'] <= sel['max'] + slack)
        assert np.all(sel['mean'] >= sel['min'] - slack)
        assert np.all(sel['mean'] <= sel['max'] + slack)

    def test_max_matches_the_spectrogram_it_derives_from(self):
        iq = testing.single_tone(PSD_DURATION, FS, frequency_offset=5 * RES, snr=20)
        da = psd_of(iq, time_statistic=('max',))
        spg, _ = spg_of(iq, PSD_CAPTURE, as_xarray=False)

        expected = levels(spg).max(axis=1)
        # the 2 paths quantize independently: `spectrogram` rounds to 2 decimals
        # before its float16 cast, and `power_spectral_density` only casts
        atol = (
            registered_tolerance(PSD_CAPTURE, spg_spec()).on_peak.peak
            + registered_tolerance(
                PSD_CAPTURE, psd_spec(time_statistic=('max',))
            ).on_peak.peak
        )
        assert_close(levels(da)[:, 0], expected, atol=atol)

    def test_two_tones_keep_their_own_bins_and_levels(self):
        bins = (-9, 5)
        levels_dB = (0.0, -20.0)
        iq = sum(
            10 ** (level / 20) * testing.tone(PSD_DURATION, FS, frequency=b * RES)
            for b, level in zip(bins, levels_dB)
        )
        da = psd_of(iq, time_statistic=('mean',))

        psd = levels(da)[0, 0]
        freqs = da.baseband_frequency.values
        tol = registered_tolerance(PSD_CAPTURE, psd_spec(time_statistic=('mean',)))
        atol = sa.util.elementwise_atol(tol, levels_dB)
        for b, level, tone_atol in zip(bins, levels_dB, atol):
            index = int(np.argmin(np.abs(freqs - b * RES)))
            assert_close(
                psd[index],
                level,
                rtol=tol.rtol,
                atol=tone_atol,
                err_msg=f'tone at bin {b}',
            )

        others = np.delete(psd, [NFFT // 2 + b for b in bins])
        assert others.max() < off_peak_floor_dBc(FFT_SIGMA, NFFT, size=others.size)

    def test_values_are_not_quantized_to_float16(self):
        iq = testing.single_tone(PSD_DURATION, FS, frequency_offset=5 * RES, snr=20)
        da = psd_of(iq, time_statistic=('mean',))

        values = np.asarray(da.values, dtype='float32')
        assert len(np.unique(values)) > 16
        assert not np.array_equal(values, values.astype('float16').astype('float32'))

    @pytest.mark.parametrize(
        'statistic', ['bogus', 1.5, -0.1], ids=['unknown_name', 'above_one', 'negative']
    )
    def test_unsupported_time_statistic_is_rejected(self, statistic):
        """`stat_ufunc_from_shorthand` and `xp.quantile` would raise on these mid-
        measurement; the validator rejects them at the field before any IQ"""
        iq = testing.tone(PSD_DURATION, FS)
        with pytest.raises(
            msgspec.ValidationError, match=f'time_statistic entry {statistic!r}'
        ) as excinfo:
            psd_of(iq, time_statistic=('mean', statistic))

        assert str(excinfo.value).startswith('$.power_spectral_density: ')


# %% spectrogram_histogram

HIST_FS = 1e6
HIST_DURATION = 1e-3
HIST_FREQUENCY_RESOLUTION = 50e3
HIST_NFFT = round(HIST_FS / HIST_FREQUENCY_RESOLUTION)
HIST_CAPTURE = sa.specs.Capture(duration=HIST_DURATION, sample_rate=HIST_FS)

RTOL = elementwise_rtol(np.float32)


def spectrogram_histogram_of(iq, window='hamming', **kwargs):
    return sa.measurements.spectrogram_histogram(
        iq,
        HIST_CAPTURE,
        window=window,
        frequency_resolution=HIST_FREQUENCY_RESOLUTION,
        as_xarray=True,
        **POWER_BINS,
        **kwargs,
    )


def test_spectrogram_histogram_fractions_sum_to_one():
    """both ports are normalized by the count of port 0, which is the same count"""
    iq = sa.testing.circular_awgn(HIST_DURATION, HIST_FS, power=1e-6 * HIST_FS, ports=2)

    da = spectrogram_histogram_of(iq)

    assert (da.values > 0).sum(axis=-1).min() > 1, 'expected a spread of bins'
    assert_close(da.values.sum(axis=-1), np.ones(2), rtol=RTOL)


def test_spectrogram_histogram_concentrates_bin_centered_tone():
    """a tone on an FFT bin center, taken through a rectangular window, puts all of
    its power in 1 of the `HIST_NFFT` bins of every STFT window and exactly 0 in the rest.

    The populated bin therefore reads 0 dBm rather than a level referred to the 50 kHz
    noise bandwidth that the units attr reports. The empty bins fall in the -inf
    catch-all.

    The fractions are 19/20 and 1/20. `HIST_NFFT` is 20, not a power of 2, so neither
    fraction is dyadic. They are computed here in float32 to match the measurement's
    registered dtype and compared with a float32 tolerance rather than exact equality.
    """
    iq = sa.testing.tone(
        HIST_DURATION, HIST_FS, frequency=2 * HIST_FREQUENCY_RESOLUTION
    )

    da = spectrogram_histogram_of(iq, window='boxcar')

    assert da.attrs['noise_bandwidth'] == HIST_FREQUENCY_RESOLUTION
    bins, fractions = zip(*populated(da.values[0], da.spectrogram_power_bin.values))
    assert bins == (float('-inf'), 0.0)
    assert_close(fractions, ((HIST_NFFT - 1) / HIST_NFFT, 1 / HIST_NFFT), rtol=RTOL)


@pytest.mark.parametrize(
    'integration_bandwidth,noise_bandwidth,long_name',
    [
        (None, 50e3, 'Spectrogram bin power (dBm/50 kHz)'),
        (100e3, 100e3, 'Spectrogram bin power (dBm/100 kHz)'),
    ],
    ids=['no_integration', 'integrate_100kHz'],
)
def test_spectrogram_histogram_bin_coordinate(
    integration_bandwidth, noise_bandwidth, long_name
):
    """the bin coordinate is the shared power grid with the noise bandwidth of the
    readings, the same attr as on the histogram variable"""
    iq = sa.testing.tone(HIST_DURATION, HIST_FS)

    da = spectrogram_histogram_of(iq, integration_bandwidth=integration_bandwidth)

    coord_attrs = da.spectrogram_power_bin.attrs
    assert coord_attrs['units'] == 'dBm'
    assert coord_attrs['noise_bandwidth'] == noise_bandwidth
    assert coord_attrs['long_name'] == long_name
    assert da.attrs['noise_bandwidth'] == noise_bandwidth
    assert 'long_name' not in da.attrs


# a histogram of 1e5 noise pixels: the dB-of-the-mean bin is populated 10 % more than
# the bins 2 dB away, against a count noise of about 1 %
HIST_NOISE_DURATION = 0.1
HIST_NOISE_LEVEL_DBM = -10.0


def test_spectrogram_histogram_white_noise_peaks_at_the_noise_bandwidth_level():
    """the mode of the dB histogram of exponentially distributed bin powers is at the
    dB of the mean, N0 * noise_bandwidth, and the fraction-weighted linear mean equals
    that power"""
    noise_mW = 10 ** (HIST_NOISE_LEVEL_DBM / 10)
    psd = noise_mW / HIST_FREQUENCY_RESOLUTION
    iq = sa.testing.circular_awgn(HIST_NOISE_DURATION, HIST_FS, power=psd * HIST_FS)
    capture = sa.specs.Capture(duration=HIST_NOISE_DURATION, sample_rate=HIST_FS)

    da = sa.measurements.spectrogram_histogram(
        iq,
        capture,
        window='hamming',
        frequency_resolution=HIST_FREQUENCY_RESOLUTION,
        as_xarray=True,
        **POWER_BINS,
    )

    bins = da.spectrogram_power_bin.values
    fractions_ = da.values[0]
    finite = np.isfinite(bins)
    mode = bins[finite][np.argmax(fractions_[finite])]
    assert abs(mode - HIST_NOISE_LEVEL_DBM) <= 2 * POWER_BINS['power_resolution']

    # the 1 dB bin centers quantize the linear mean by under 0.1 dB, and the hamming
    # window correlates adjacent bins over the 1.36-bin hamming ENBW
    linear_mean = np.sum(fractions_[finite] * 10 ** (bins[finite] / 10))
    pixels = round(HIST_NOISE_DURATION * HIST_FS)
    enbw_bins = float(sw.equivalent_noise_bandwidth('hamming', HIST_NFFT))
    sigma = math.sqrt(enbw_bins / pixels)
    assert_close(
        10 * np.log10(linear_mean),
        HIST_NOISE_LEVEL_DBM,
        atol=sw.power_analysis.level_tolerance_dB(5 * sigma, power=True) + 0.1,
    )


# %% spectrogram_ratio_histogram


def ratio_histogram(iq, **kwargs):
    return sa.measurements.spectrogram_ratio_histogram(
        iq,
        HIST_CAPTURE,
        window='hamming',
        frequency_resolution=HIST_FREQUENCY_RESOLUTION,
        as_xarray=True,
        **POWER_BINS,
        **kwargs,
    )


def two_ports_offset_by(offset_dB):
    """noise repeated on 2 ports, with port 1 scaled up by `offset_dB`.

    Noise rather than a tone keeps every spectrogram bin well above the float32
    roundoff floor. In the near-empty bins of a tone's spectrogram the cross-port
    ratio is roundoff noise rather than the applied offset.
    """
    iq = sa.testing.circular_awgn(HIST_DURATION, HIST_FS, power=1e-6 * HIST_FS, ports=2)
    iq[1] = iq[0] * 10 ** (offset_dB / 20)
    return iq


@pytest.mark.parametrize('offset_dB', [0.0, 6.0, -7.0], ids='offset{:g}dB'.format)
def test_spectrogram_ratio_histogram_offset_ports(offset_dB):
    """row 0 is spg[0]-spg[1] and row 1 is spg[1]-spg[0], so a level offset between
    the ports puts the 2 rows at opposite signs of it"""
    da = ratio_histogram(two_ports_offset_by(offset_dB))

    bins = da.spectrogram_ratio_power_bin.values
    assert populated(da.values[0], bins) == [(-offset_dB, 1.0)]
    assert populated(da.values[1], bins) == [(offset_dB, 1.0)]


@pytest.mark.parametrize('ports', [1, 3], ids='ports{}'.format)
def test_spectrogram_ratio_histogram_requires_two_ports(ports):
    iq = sa.testing.tone(HIST_DURATION, HIST_FS, ports=ports)

    with pytest.raises(ValueError, match='only supported for 2-channel measurements'):
        ratio_histogram(iq)


def test_spectrogram_ratio_histogram_bin_units_are_ratios():
    """the bin coordinate is a cross-port ratio, so its units are dB rather than the
    dBm of the absolute spectrogram histogram, and neither the coordinate nor the
    histogram has a noise bandwidth"""
    da = ratio_histogram(two_ports_offset_by(0.0))

    assert da.spectrogram_ratio_power_bin.attrs['units'] == 'dB'
    assert 'noise_bandwidth' not in da.spectrogram_ratio_power_bin.attrs
    assert 'noise_bandwidth' not in da.attrs


# %% tolerance

TOLERANCE_CASES = [
    (CAPTURE, spg_spec()),
    (PSD_CAPTURE, psd_spec()),
    (ssb_spectrogram_capture(SSB_SPECTROGRAM_PERIODICITY), SSB_SPECTROGRAM_SPEC),
]
TOLERANCE_IDS = ['spectrogram', 'power_spectral_density', 'cellular_5g_ssb_spectrogram']


@pytest.mark.parametrize('capture,spec', TOLERANCE_CASES, ids=TOLERANCE_IDS)
def test_tolerance_is_looser_for_cupy_than_numpy(capture, spec):
    numpy_tol = registered_tolerance(capture, spec, array_backend='numpy')
    cupy_tol = registered_tolerance(capture, spec, array_backend='cupy')
    assert cupy_tol.on_peak.rms > numpy_tol.on_peak.rms
    assert cupy_tol.on_peak.peak > numpy_tol.on_peak.peak
