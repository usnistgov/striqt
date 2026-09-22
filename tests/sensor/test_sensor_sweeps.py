"""striqt.sensor sweeps end to end over the synthetic sources: the CLI entry point on
the YAML files under sweeps/, and in-memory sweeps checked against the
striqt.analysis.testing generators that the sources themselves call.

The PSD oracle is the window's own spectrum: a unit tone at fine bin k0 puts
``|W(k - k0)|**2 / (nfft * sum(w**2))`` of its power in fine bin k (Parseval), and a
reported bin integrates the fine bins it spans. For a bin-centred tone with no
integration this is the ``-10*log10(enbw_bins)`` of tests/analysis; with
quick.yaml's 24-bin integration it is the in-block fraction of the kaiser window,
about -0.002 dB, with -33 dB spilling into the neighbouring block.
"""

from __future__ import annotations

import math
import os
from fractions import Fraction

import numpy as np
import pytest
from conftest import SITE_DIR, SWEEP_DIR
from numeric_checks import (
    COLA_LEVEL_DB,
    COLA_RIPPLE_DB,
    FIR_LEAKAGE,
    assert_close,
    cross_backend_peak_roundoff,
    cross_backend_rms,
    far_bin_floor_dBc,
    level_atol_dB,
    level_tolerance_dB,
    log_conversion_tol,
    single_backend_rms,
)
from site_strategies import RADIO_ID
from sweep_strategies import SOURCE
from synthetic_sources import (
    ANALYSIS,
    DETECTOR_PERIOD,
    FILTER_ONLY,
    FILTER_SIZE,
    RESAMPLE_FILTER,
    expected_corrected,
    fs_sdr,
    make_capture,
    make_sweep,
    resampler_nffts,
    run_in_memory,
)

import striqt.analysis as sa
import striqt.sensor as ss
import striqt.waveform as sw
from striqt.cli import sensor_sweep

# the ports of a capture in row order
ports_of = ss.specs.helpers.ensure_tuple

# %% oracles


def correction_sigma(capture, extra_nffts=()) -> float:
    """rms roundoff bound of correct_iq relative to the signal: the resampler's
    forward and inverse FFTs plus the FIR's overlap-add pair, sized at the next power
    of two above the filter"""
    fir_nfft = 2 ** math.ceil(math.log2(2 * FILTER_SIZE))
    nffts = [*resampler_nffts(capture), fir_nfft, fir_nfft, *extra_nffts]
    return single_backend_rms(np.complex64, nffts)


def tone_bin_powers(window, nfft, nzero, tone_bin, bin_centers, half_width):
    """the fraction of a unit tone's power that each reported PSD bin receives.

    `tone_bin`, `bin_centers` and `half_width` are in units of fine (nfft) bins; a
    reported bin integrates the fine bins within `half_width` of its center. The
    window is `nfft - nzero` samples long and zero-padded to `nfft`, as
    `sw.spectrogram` does for window_fill < 1.
    """
    w = np.zeros(nfft)
    w[: nfft - nzero] = np.asarray(sw.get_window(window, nfft - nzero), dtype='float64')
    spectrum = np.abs(np.fft.fft(w)) ** 2 / (nfft * np.sum(w**2))
    fine_bins = np.fft.fftfreq(nfft, 1 / nfft).round().astype(int)
    powers = []
    for center in bin_centers:
        members = np.abs(fine_bins + tone_bin - center) < half_width
        powers.append(spectrum[members].sum())
    return np.asarray(powers)


def exponential_mean_bound(windows, count, p=1e-3):
    """x such that the mean of `windows` unit exponentials exceeds x in any of
    `count` bins with probability below `p`, from the Chernoff bound
    ``P(mean > x) <= (x*exp(1-x))**windows``"""
    lo, hi = 1.0, 100.0
    for _ in range(60):
        x = (lo + hi) / 2
        if count * (x * math.exp(1 - x)) ** windows > p:
            lo = x
        else:
            hi = x
    return hi


def window_count(attrs, capture) -> int:
    nfft = round(capture.sample_rate / attrs['frequency_resolution'])
    hop = nfft - round(Fraction(str(attrs['fractional_overlap'])) * nfft)
    return (round(capture.duration * capture.sample_rate) - nfft) // hop + 1


def check_tone_psd(psd, freqs, attrs, capture, frequency_offset, snr, err_msg=''):
    """assert the mean PSD of one port holds a unit tone at `frequency_offset` at the
    modelled level, and nothing else above the noise and roundoff floors"""
    res = attrs['frequency_resolution']
    nfft = round(capture.sample_rate / res)
    nzero = round((1 - Fraction(str(attrs['window_fill']))) * nfft)
    window = attrs['window']
    if isinstance(window, list):
        window = tuple(window)
    assert sw.isroundmod(frequency_offset, res), 'the test tone must be bin-centred'

    tone = tone_bin_powers(
        window,
        nfft,
        nzero,
        tone_bin=round(frequency_offset / res),
        bin_centers=freqs / res,
        half_width=attrs['noise_bandwidth'] / res / 2,
    )
    if snr is None:
        noise_bin = 0.0
    else:
        # the source draws its noise at the source sample rate
        noise_bin = 10 ** (-snr / 10) / fs_sdr(capture) * attrs['noise_bandwidth']

    k0 = int(np.argmin(np.abs(freqs - frequency_offset)))
    assert int(np.argmax(psd)) == k0, f'{err_msg} tone bin'
    assert_close(
        psd[k0],
        10 * np.log10(tone[k0] + noise_bin),
        atol=COLA_LEVEL_DB,
        err_msg=f'{err_msg} tone level',
    )

    others = np.arange(psd.size) != k0
    tail = exponential_mean_bound(window_count(attrs, capture), others.sum())
    roundoff_dB = far_bin_floor_dBc(
        correction_sigma(capture, [nfft]), nfft, size=others.sum()
    )
    with np.errstate(divide='ignore'):
        bound = 10 * np.log10(tone + noise_bin * tail) + COLA_LEVEL_DB
    bound = np.maximum(bound, roundoff_dB)
    excess = psd[others] - bound[others]
    assert excess.max() <= 0, (
        f'{err_msg} other bins exceed the model by {excess.max():.2f} dB'
    )


def mean_psd(ds, row):
    """the 'mean' PSD of one row and its frequency axis, without the NaN padding
    that concatenating captures of different bandwidths adds"""
    psd = ds.power_spectral_density.sel(time_statistic='mean').isel(capture=row)
    psd = psd.dropna('baseband_frequency')
    return psd.values.astype('float64'), psd.baseband_frequency.values


def check_site(ds):
    # sites/global.yaml and sites/radio02.yaml keyed on the extension module's RADIO_ID
    assert set(ds.source_id.values) == {RADIO_ID}
    assert set(ds.site_name.values) == {'WAPA-north'}
    assert set(ds.radio_name.values) == {'radio02'}
    assert set(ds.channel_name.values) == {'3750 MHz'}
    assert ds.antenna_name.values.tolist() == ['Omni', '1x32']
    assert ds.antenna_polarization.values.tolist() == ['Vertical', 'Linear +45']
    assert ds.antenna_model.values.tolist() == ['OmniModel', 'PanelModel']
    assert ds.antenna_index.values.tolist() == [0, 1]
    assert set(ds.gain.values) == {0.0}
    assert set(ds.mast_height.values) == {1.7}


# %% the CLI entry point

CLI_SWEEPS = {
    'synthetic': SWEEP_DIR / 'synthetic.yaml',
    'site': SITE_DIR / 'site-cpu.yaml',
}


@pytest.mark.parametrize('name', list(CLI_SWEEPS), ids=list(CLI_SWEEPS))
def test_cli_run(name, tmp_path, monkeypatch, subtests):
    path = CLI_SWEEPS[name]
    # open_resources chdirs into the spec directory and does not change back
    monkeypatch.chdir(os.getcwd())
    out_path = tmp_path / 'out.zarr.zip'

    sensor_sweep.run(str(path), output_path=str(out_path))

    spec = ss.read_yaml_spec(path, output_path=str(out_path))
    assert ss.read_zarr_spec(out_path, extension_root=path.parent) == spec
    ds = sa.load(out_path)

    source_id = str(ds.source_id.values[0])
    captures = ss.specs.helpers.loop_captures(spec, source_id=source_id)
    repeat = spec.loops[0].count if isinstance(spec.loops[0], ss.specs.Repeat) else 1
    rows = [(i, p) for i, c in enumerate(captures) for p in ports_of(c.port)]
    assert ds.capture_index.values.tolist() == [i for i, _ in rows] * repeat
    assert ds.port.values.tolist() == [p for _, p in rows] * repeat
    assert ds.sweep_index.values.tolist() == sorted([*range(repeat)] * len(rows))
    assert set(spec.analysis.to_dict()) <= set(ds.data_vars)

    attrs = ds.power_spectral_density.attrs
    for row, (i, port) in enumerate(rows * repeat):
        with subtests.test('tone in the PSD', row=row, port=port):
            psd, freqs = mean_psd(ds, row)
            capture = captures[i]
            check_tone_psd(
                psd, freqs, attrs, capture, capture.frequency_offset, capture.snr
            )

    if name == 'site':
        check_site(ds)


# %% in-memory sweeps against the generators

FS = RESAMPLE_FILTER['sample_rate']
SAMPLES_PER_DETECTOR_BIN = round(FS * DETECTOR_PERIOD)
DETECTOR_BINS = round(RESAMPLE_FILTER['duration'] / DETECTOR_PERIOD)
# 100 samples past the start of detector bin 8, so that neither the impulse nor the
# FIR ringing around it touches a bin edge
IMPULSE_BIN = 8
IMPULSE_TIME = (IMPULSE_BIN * SAMPLES_PER_DETECTOR_BIN + 100) / FS
FILTER_ONLY_FS = FILTER_ONLY['sample_rate']
FILTER_ONLY_IMPULSE_TIME = (
    IMPULSE_BIN * round(FILTER_ONLY_FS * DETECTOR_PERIOD) + 100
) / FILTER_ONLY_FS

# -1e6 and 1.6e6 Hz sit on the 8 kHz PSD grid and on the 1 kHz grid of the
# resampler's 6250-point FFT at 6.25 MS/s, so the tone passes without COLA ripple
FIDELITY_CAPTURES = {
    'single_tone': (
        make_capture('single_tone', frequency_offset=-1e6, snr=None),
        make_capture('single_tone', frequency_offset=1.6e6, snr=None),
    ),
    'noise': (make_capture('noise', noise_psd=1e-17),),
    'sawtooth': (
        make_capture('sawtooth', period=RESAMPLE_FILTER['duration'], power=0.0),
    ),
    'dirac_delta': (
        make_capture('dirac_delta', time=IMPULSE_TIME, power=0.0),
        make_capture(
            'dirac_delta', **FILTER_ONLY, time=FILTER_ONLY_IMPULSE_TIME, power=-3.0
        ),
    ),
}


def detector(ds, kind):
    return ds.channel_power_time_series.sel(power_detector=kind).values.astype(
        'float64'
    )


def check_single_tone(ds, capture, subtests):
    sigma = correction_sigma(capture)
    with subtests.test('iq_waveform reproduces the generator'):
        # the tone is continuous through the source pre-roll, so the whole capture
        # is steady state
        assert_close(
            ds.iq_waveform.values,
            expected_corrected('single_tone', capture),
            sigma=sigma,
        )
    with subtests.test('rms detector reads 0 dBm'):
        assert_close(detector(ds, 'rms'), 0.0, atol=level_tolerance_dB(sigma))
    attrs = ds.power_spectral_density.attrs
    for row in range(ds.sizes['capture']):
        with subtests.test('tone in the PSD', row=row):
            psd, freqs = mean_psd(ds, row)
            check_tone_psd(psd, freqs, attrs, capture, capture.frequency_offset, None)


def check_noise(ds, capture, subtests):
    """only the noise power survives resampling: the PSD is flat at
    noise_psd * noise_bandwidth per bin, and its mean over ports, windows and bins
    averages that many exponential variates (hann bins are correlated over enbw)"""
    attrs = ds.power_spectral_density.attrs
    nfft = round(capture.sample_rate / attrs['frequency_resolution'])
    mean_dB = ds.power_spectral_density.sel(time_statistic='mean')
    linear = 10 ** (mean_dB.values.astype('float64') / 10)
    expected = capture.noise_psd * attrs['noise_bandwidth']
    count = linear.size * window_count(attrs, capture)
    sigma = math.sqrt(
        float(sw.equivalent_noise_bandwidth(attrs['window'], nfft)) / count
    )
    with subtests.test('PSD level'):
        assert_close(
            10 * np.log10(linear.mean()),
            10 * np.log10(expected),
            atol=level_tolerance_dB(5 * sigma, power=True) + COLA_LEVEL_DB,
        )


def sawtooth_rms_model_dB(capture, bins):
    """rms power of the ramp ``amplitude * t / period`` over detector bin `i` of N
    samples: the mean over n of ``((i*N + n) / (period*fs))**2`` in closed form"""
    N = round(capture.sample_rate * DETECTOR_PERIOD)
    i = np.asarray(bins)
    mean_sq = i**2 * N**2 + i * N * (N - 1) + (N - 1) * (2 * N - 1) / 6
    amplitude = 10 ** (capture.power / 20)
    return 10 * np.log10(
        mean_sq * (amplitude / (capture.period * capture.sample_rate)) ** 2
    )


def check_sawtooth(ds, capture, subtests):
    """a single ramp over the capture. The source pre-roll ends at full scale, so the
    ramp start is a step whose FIR ringing spans FILTER_SIZE//2 samples at each end;
    a linear ramp passes a symmetric unit-gain filter unchanged elsewhere."""
    sigma = correction_sigma(capture)
    pad = FILTER_SIZE // 2
    with subtests.test('rms detector follows the ramp'):
        skip = math.ceil(pad / SAMPLES_PER_DETECTOR_BIN)
        bins = np.arange(skip, DETECTOR_BINS - skip)
        rms_dB = detector(ds, 'rms')[:, bins]
        assert np.all(np.diff(rms_dB, axis=1) > 0)
        model = np.broadcast_to(sawtooth_rms_model_dB(capture, bins), rms_dB.shape)
        assert_close(rms_dB, model, atol=level_atol_dB(model, sigma))


def check_dirac_delta(ds, capture, subtests):
    amplitude = 10 ** (capture.power / 20)
    peak = detector(ds, 'peak')
    with subtests.test('peak detector bin'):
        assert np.argmax(peak, axis=1).tolist() == [IMPULSE_BIN, IMPULSE_BIN]
    # the FIR's central tap is bw/fs (a symmetric transition around bw/2), and the
    # resampler's own passband can only reduce the peak further
    filter_peak_dB = 20 * np.log10(
        amplitude * capture.analysis_bandwidth / fs_sdr(capture)
    )
    if capture.host_resample:
        with subtests.test('peak level bounded by the filter'):
            assert peak[:, IMPULSE_BIN].max() <= filter_peak_dB + COLA_RIPPLE_DB
    else:
        with subtests.test('peak level is the FIR central tap'):
            assert_close(
                peak[:, IMPULSE_BIN],
                filter_peak_dB,
                atol=level_tolerance_dB(FIR_LEAKAGE),
            )


CHECKS = {
    'single_tone': check_single_tone,
    'noise': check_noise,
    'sawtooth': check_sawtooth,
    'dirac_delta': check_dirac_delta,
}


@pytest.mark.parametrize('binding', list(CHECKS), ids=list(CHECKS))
def test_in_memory_fidelity(binding, subtests):
    captures = FIDELITY_CAPTURES[binding]
    datasets = run_in_memory(make_sweep(binding, captures))
    assert len(datasets) == len(captures)
    for i, (capture, ds) in enumerate(zip(captures, datasets)):
        assert ds.sizes['capture'] == len(ports_of(capture.port))
        assert set(ANALYSIS.to_dict()) <= set(ds.data_vars)
        with subtests.test(capture=i):
            CHECKS[binding](ds, capture, subtests)


@pytest.mark.namespaces('cupy')
@pytest.mark.parametrize('binding', list(CHECKS), ids=list(CHECKS))
def test_in_memory_fidelity_cupy(binding, array_backend, subtests):
    """the cupy signal path agrees with numpy to within the roundoff of both"""
    captures = FIDELITY_CAPTURES[binding]
    reference = run_in_memory(make_sweep(binding, captures))
    source = SOURCE.replace(array_backend=array_backend)
    datasets = run_in_memory(make_sweep(binding, captures, source=source))
    # each backend rounds its own log10 and stores the result as float32 dB
    level_tol = log_conversion_tol(np.float32, 10, complex_input=True, n_impl=2)
    for i, (capture, ds, ref) in enumerate(zip(captures, datasets, reference)):
        sigma = cross_backend_rms(np.complex64, resampler_nffts(capture))
        with subtests.test('iq_waveform', capture=i):
            expected = ref.iq_waveform.values
            assert_close(
                ds.iq_waveform.values,
                expected,
                sigma=sigma,
                # the resample concentrates its roundoff on the samples that carry the
                # peak, which an rms-referenced sigma understates by the crest factor -
                # near 100 for the impulse captures
                atol=cross_backend_peak_roundoff(np.complex64) * np.abs(expected).max(),
            )
        for name in ('power_spectral_density', 'channel_power_time_series'):
            with subtests.test(name, capture=i):
                expected = ref[name].values
                assert_close(
                    ds[name].values,
                    expected,
                    rtol=level_tol['rtol'],
                    atol=level_tol['atol'] + level_atol_dB(expected, sigma),
                )
