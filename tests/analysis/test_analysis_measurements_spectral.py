"""striqt.analysis.measurements.spectrogram, power_spectral_density and
cellular_5g_ssb_spectrogram: levels, axes and the shared spectrogram machinery.

The level assertions are anchored on the closed form of a bin-centered unit tone. For
a boxcar window the peak bin reads exactly ``0.0`` dB, and for any other window it
reads ``-10*log10(enbw_bins)``, where ``enbw_bins`` is the window's equivalent noise
bandwidth in FFT bins: `shared._cached_spectrogram` normalizes the STFT to a power
spectral density and labels the result with ``noise_bandwidth = frequency_resolution``,
so a tone whose power all falls in one bin is reported low by the ratio of the window's
ENBW to that labeled bandwidth.
"""

from __future__ import annotations

import fractions
import re

import msgspec
import numpy as np
import pytest
from numeric_checks import ATOL, RTOL_FLOAT64, assert_close

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis import testing
from striqt.waveform.lib.fourier import fft_tolerance_rms, off_peak_floor_dBc

# a 64-point FFT over 8 non-overlapping windows: 512 samples, 500 us
FS = 1.024e6
RES = 16e3
NFFT = round(FS / RES)
NWINDOW = 8
DURATION = NWINDOW * NFFT / FS

CAPTURE = sa.specs.Capture(duration=DURATION, sample_rate=FS)

# roundoff of the one FFT that separates the IQ from the reported spectrum
FFT_SIGMA = fft_tolerance_rms(np.complex64, [NFFT])

REGISTRY = sa.register.registry


def levels(result) -> np.ndarray:
    """the dB values of a measurement result, widened to float64 for comparison"""
    values = result.values if hasattr(result, 'values') else result
    return np.asarray(values, dtype='float64')


def registered_tolerance(capture, spec, **kwargs) -> sa.specs.Tolerance:
    """the tolerance registered for the measurement that `spec` selects, fetched
    through the registry walk a consumer would use"""
    name = REGISTRY[type(spec)].name
    group = REGISTRY.tospec()(**{name: spec})
    return REGISTRY.tolerances(capture, group, **kwargs)[name]


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
        # neighbors hold only FFT roundoff referred to the 0 dB tone, like every
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
        iq = testing.noise(DURATION, FS, noise_psd=1 / FS)
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
        assert ref_metadata['units'] == 'dBm/16 kHz'
        assert da.attrs['noise_bandwidth'] == frequency_bins * RES
        assert da.attrs['units'] == 'dBm/64 kHz'

        # each side carries its own quantization and roundoff budget
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
        ],
        ids=[
            'time_aperture',
            'integration_bandwidth',
            'frequency_resolution',
            'window_fill',
        ],
    )
    def test_non_integer_binning_raises(self, kwargs, message, quantities):
        """the sizing validator rejects these before any IQ is touched, so the error
        is a msgspec.ValidationError carrying the measurement's field path"""
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
            sa.measurements.shared.validated_spectrogram_sizing(
                CAPTURE, sa.specs.Spectrogram(**spec_fields)
            )

        # the validator is the only layer that knows which quantities the rule
        # compared, so it names them in a parenthetical after the rule text
        for quantity in quantities:
            assert quantity in str(valueinfo.value)


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
        # the two paths quantize independently: `spectrogram` rounds to 2 decimals
        # before its float16 cast, `power_spectral_density` only casts
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


# %% cellular_5g_ssb_spectrogram

SSB_SCS = 30e3
# the STFT nfft is sample_rate/(subcarrier_spacing/2) and has to be a multiple of 28
# for the 13/28 overlap and 15/28 window fill to land on whole samples; 420 kHz is the
# smallest rate that satisfies it, which keeps this the cheapest capture that still
# spans a whole 20 ms discovery period
SSB_FS = 420e3
SSB_SAMPLE_RATE = 120e3
SSB_PERIODICITY = 20e-3
SSB_SYMBOLS = round(28 * SSB_SCS / 15e3)


SSB_SPEC = sa.specs.Cellular5GNRSSBSpectrogram(
    subcarrier_spacing=SSB_SCS,
    sample_rate=SSB_SAMPLE_RATE,
    discovery_periodicity=SSB_PERIODICITY,
    window='boxcar',
)


def ssb_capture(duration) -> sa.specs.Capture:
    return sa.specs.Capture(duration=duration, sample_rate=SSB_FS)


def ssb_of(duration, frequency=None, **kwargs):
    if frequency is None:
        iq = testing.noise(duration, SSB_FS, noise_psd=1 / SSB_FS)
    else:
        iq = testing.tone(duration, SSB_FS, frequency=frequency)
    return sa.measurements.cellular_5g_ssb_spectrogram(
        iq, ssb_capture(duration), as_xarray=True, **(SSB_SPEC.to_dict() | kwargs)
    )


def ssb_tone_level_dB() -> float:
    """level of a tone centered on a 15 kHz bin after integration to one 30 kHz bin.

    The STFT window is a boxcar of `window_fill` * nfft samples zero-padded to nfft,
    whose ENBW is nfft/L bins, so the tone bin alone reads 10*log10(L/nfft). Its
    Dirichlet kernel is not zero on the neighboring bins, and the integration sums
    the tone bin with one neighbor holding sinc(L/nfft)**2 of the tone's power.
    """
    _, window_fill = sa.measurements.shared.cellular_stft_window_fractions('normal')
    nfft = round(2 * SSB_FS / SSB_SCS)
    L = round(window_fill * nfft)
    return 10 * np.log10(L / nfft * (1 + np.sinc(L / nfft) ** 2))


class TestCellular5GSSBSpectrogram:
    @pytest.mark.parametrize('blocks', [1, 2], ids='blocks{}'.format)
    def test_axis_shapes(self, blocks):
        da = ssb_of(blocks * SSB_PERIODICITY)

        assert da.sizes['cellular_ssb_index'] == blocks
        assert da.sizes['cellular_ssb_symbol_index'] == SSB_SYMBOLS
        assert list(da.cellular_ssb_symbol_index.values) == list(range(SSB_SYMBOLS))

        # the frequency axis is binned to one bin per subcarrier spacing and then
        # truncated to the SSB sample rate around frequency_offset
        freqs = da.cellular_ssb_baseband_frequency.values
        assert freqs.size == round(SSB_SAMPLE_RATE / SSB_SCS)
        assert_close(np.diff(freqs), SSB_SCS, rtol=RTOL_FLOAT64)
        assert np.all(np.abs(freqs) <= SSB_SAMPLE_RATE / 2)

    def test_tone_lands_in_its_labeled_frequency_bin_in_every_block(self):
        tone_frequency = SSB_SCS
        da = ssb_of(2 * SSB_PERIODICITY, frequency=tone_frequency)

        freqs = da.cellular_ssb_baseband_frequency.values
        expected = int(np.argmin(np.abs(freqs - tone_frequency)))
        spg = levels(da)
        assert np.all(np.argmax(spg, axis=-1) == expected)

        # a stationary tone reads the same in every symbol of every block
        peak = spg[..., expected]
        assert peak.max() == peak.min()

    def test_tone_level_sums_the_tone_bin_with_its_leaking_neighbor(self):
        duration = 2 * SSB_PERIODICITY
        da = ssb_of(duration, frequency=SSB_SCS)

        freqs = da.cellular_ssb_baseband_frequency.values
        index = int(np.argmin(np.abs(freqs - SSB_SCS)))
        tol = registered_tolerance(ssb_capture(duration), SSB_SPEC)
        assert_close(
            levels(da)[..., index],
            ssb_tone_level_dB(),
            rtol=tol.rtol,
            atol=tol.on_peak.peak,
        )


# %% tolerance

TOLERANCE_CASES = [
    (CAPTURE, spg_spec()),
    (PSD_CAPTURE, psd_spec()),
    (ssb_capture(SSB_PERIODICITY), SSB_SPEC),
]
TOLERANCE_IDS = ['spectrogram', 'power_spectral_density', 'cellular_5g_ssb_spectrogram']


@pytest.mark.parametrize('capture,spec', TOLERANCE_CASES, ids=TOLERANCE_IDS)
def test_tolerance_is_a_dB_budget_with_peak_above_rms(capture, spec):
    tol = registered_tolerance(capture, spec)
    assert isinstance(tol, sa.specs.Tolerance)
    assert tol.units == 'dB'
    assert tol.on_peak.peak >= tol.on_peak.rms > 0
    assert tol.off_peak_dBc.rms < tol.off_peak_dBc.peak < 0


@pytest.mark.parametrize('capture,spec', TOLERANCE_CASES, ids=TOLERANCE_IDS)
def test_tolerance_grows_with_the_input_error(capture, spec):
    exact = registered_tolerance(capture, spec, input_error=0.0)
    perturbed = registered_tolerance(capture, spec, input_error=1e-4)
    assert perturbed.on_peak.rms > exact.on_peak.rms
    assert perturbed.on_peak.peak > exact.on_peak.peak


@pytest.mark.parametrize('capture,spec', TOLERANCE_CASES, ids=TOLERANCE_IDS)
def test_tolerance_is_looser_for_cupy_than_numpy(capture, spec):
    numpy_tol = registered_tolerance(capture, spec, array_backend='numpy')
    cupy_tol = registered_tolerance(capture, spec, array_backend='cupy')
    assert cupy_tol.on_peak.rms > numpy_tol.on_peak.rms
    assert cupy_tol.on_peak.peak > numpy_tol.on_peak.peak
