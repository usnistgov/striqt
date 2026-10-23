"""striqt.sensor sweeps end to end over the synthetic sources: the CLI on the YAML files
under sweeps/, and in-memory sweeps checked against the generators the sources call.

The PSD oracle is the window's own spectrum: a unit tone at fine bin k0 puts
``|W(k - k0)|**2 / (nfft * sum(w**2))`` of its power in fine bin k (Parseval), and a
reported bin integrates the fine bins it spans. For a bin-centered tone with no
integration this is the ``-10*log10(enbw_bins)`` of tests/analysis; with quick.yaml's
24-bin integration it is the in-block fraction of the kaiser window, about -0.002 dB.
The remaining -33 dB spills into the neighboring block.
"""

from __future__ import annotations

import ast
import math
import os
import re
from fractions import Fraction

import numpy as np
import pytest
from conftest import FAKE_SOAPY_CELLULAR_SPEC, SITE_CELLULAR_SPEC, SITE_DIR, SWEEP_DIR
from numeric_checks import COLA_LEVEL_DB, COLA_RIPPLE_DB, assert_close, cross_backend
from site_strategies import RADIO_ID
from sweep_strategies import SOURCE
from synthetic_sources import (
    ANALYSIS,
    DETECTOR_PERIOD,
    FILTER_ONLY,
    FILTER_SIZE,
    ONE_PORT,
    RESAMPLE_FILTER,
    fs_sdr,
    preset_capture,
    preset_sweep,
    run_in_memory,
    tone_capture,
)

import striqt.analysis as sa
import striqt.sensor as ss
import striqt.waveform as sw
from striqt.cli import check_sweep, sensor_sweep
from striqt.sensor.lib.compute import capture_tolerances, correction_error
from striqt.waveform import fourier
from striqt.waveform.fourier import off_peak_floor_dBc, on_peak_roundoff

# the ports of a capture in row order
ports_of = ss.specs.captures.ensure_tuple

# %% oracles


def tone_bin_powers(window, nfft, nzero, tone_bin, bin_centers, half_width):
    """the fraction of a unit tone's power that each reported PSD bin receives.

    `tone_bin`, `bin_centers`, and `half_width` are in units of fine (nfft) bins; a
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
    """assert the mean PSD of one port contains a unit tone at `frequency_offset` at the
    modeled level, and nothing else above the noise and roundoff floors"""
    res = attrs['frequency_resolution']
    nfft = round(capture.sample_rate / res)
    nzero = round((1 - Fraction(str(attrs['window_fill']))) * nfft)
    window = attrs['window']
    if isinstance(window, list):
        window = tuple(window)
    assert sw.isroundmod(frequency_offset, res), 'the test tone must be bin-centered'

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
    sigma = math.hypot(
        correction_error(capture, SOURCE, ANALYSIS),
        fourier.fft_tolerance_rms(np.complex64, [nfft]),
    )
    roundoff_dB = off_peak_floor_dBc(sigma, nfft, size=others.sum())
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
    captures = ss.specs.sequencing.loop_captures(spec, source_id=source_id)
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


def test_check_sweep_prints_the_error_budget(monkeypatch, capsys):
    path = SWEEP_DIR / 'synthetic.yaml'
    # open_resources chdirs into the spec directory and does not change back
    monkeypatch.chdir(os.getcwd())

    check_sweep.run(str(path))

    out = capsys.readouterr().out
    assert 'Numerical tolerance' in out
    # the block is the title, a rule, the column header, and one row per product
    block = out.split('Numerical tolerance')[1].split('\n\n')[0]
    groups, header, *rows = block.splitlines()[2:]
    assert 'On-peak tolerance (dB)' in groups and 'Off-peak (dBc)' in groups
    assert header.split() == ['rms', 'peak', 'rms', 'peak']
    table = {row.split()[0]: row.split()[1:] for row in rows}

    spec = ss.read_yaml_spec(path)
    budgeted = {
        name
        for name in spec.analysis.to_dict()
        if sa.registry[type(getattr(spec.analysis, name))].tolerance is not None
    }
    assert budgeted == {
        'power_spectral_density',
        'channel_power_time_series',
        'spectrogram',
        'iq_waveform',
    }
    assert set(table) == budgeted
    for name, fields in table.items():
        # rtol, then the on-peak and off-peak pairs
        assert len(fields) == 5 and all(math.isfinite(float(f)) for f in fields), name


def test_check_sweep_formats_the_default_sink_path(write_yaml, monkeypatch, capsys):
    """a spec with no `sink:` block takes the `Sink.path` default, whose format
    fields must all be ones that `get_path_fields` supplies"""
    path = write_yaml(
        'no-sink-block.yaml',
        """
        sensor_binding: single_tone
        source:
          master_clock_rate: 125e6
          num_rx_ports: 1
          array_backend: numpy
        captures:
          - port: 0
            sample_rate: 6.144e6
            duration: 1e-3
        analysis:
          iq_waveform: {}
        """,
    )
    # open_resources chdirs into the spec directory and does not change back
    monkeypatch.chdir(os.getcwd())

    check_sweep.run(str(path))

    out = capsys.readouterr().out
    block = re.search(
        r'^sink\.path:\n  Input: +(.*)\n  Formatted: +(.*)$', out, re.MULTILINE
    )
    assert block is not None, out
    template, formatted = (ast.literal_eval(v) for v in block.groups())
    assert set(ss.specs.paths.get_format_fields(template)) <= {
        'spec_name',
        'start_time',
    }
    assert formatted.startswith(path.stem + '-')
    assert '{' not in formatted


def test_check_sweep_lists_the_looped_field_over_its_adjustment(
    write_yaml, monkeypatch, capsys
):
    """the coordinates block follows `loop_captures`: a field that is both looped and
    adjusted lists the loop's values in loop order, while an adjusted field that no
    loop touches lists the adjusted value"""
    path = write_yaml(
        'looped-and-adjusted.yaml',
        """
        sensor_binding: single_tone
        source:
          master_clock_rate: 125e6
          num_rx_ports: 1
          array_backend: numpy
        captures:
          - port: 0
            sample_rate: 6.144e6
            duration: 1e-3
            analysis_bandwidth: 5e6
        loops:
          - kind: list
            field: host_resample
            values: [true, false]
        adjust_captures:
          defaults:
            host_resample: false
            snr: 5.0
        analysis:
          iq_waveform: {}
        sink:
          path: out.zarr.zip
        """,
    )
    # open_resources chdirs into the spec directory and does not change back
    monkeypatch.chdir(os.getcwd())

    check_sweep.run(str(path))

    out = capsys.readouterr().out
    id_line = re.search(r'^source_id: (.*)$', out, re.MULTILINE)
    source_id = ast.literal_eval(id_line[1])
    # the title, then a rule, then the entries of a pformat dict without its braces
    _, _, *entries = out.split('Unique capture field coordinates in output:')[1].split(
        '\n'
    )
    labels = ast.literal_eval('{' + '\n'.join(entries) + '}')

    spec = ss.read_yaml_spec(path)
    captures = ss.specs.sequencing.loop_captures(spec, source_id=source_id)
    looped = tuple(dict.fromkeys(c.host_resample for c in captures))
    assert looped == (True, False)
    assert labels == {
        'host_resample': looped,
        'snr': (spec.adjust_captures['defaults']['snr'],),
    }


# %% cellular sweeps: the Sweep that the runner derives from a Cellular5GNRSweep

CELLULAR_SWEEPS = {
    'fake_soapy': FAKE_SOAPY_CELLULAR_SPEC,
    'site': SITE_CELLULAR_SPEC,
}


def derived_sample_rates(spec) -> list[float]:
    """the sample rate of each row of the output, one per port of each capture, from
    the channel each capture names: the smallest multiple of 896 subcarriers at or
    above the channel bandwidth plus the 250 kHz transition band"""
    rates = []
    for capture in spec.captures:
        (channel,) = [
            c
            for c in spec.cell_channels.values()
            if c.center_frequency == capture.center_frequency
        ]
        step = 896 * channel.subcarrier_spacing
        rate = math.ceil((channel.channel_bandwidth + 250e3) / step) * step
        rates += [rate] * len(ports_of(capture.port))
    return rates


@pytest.mark.parametrize('name', list(CELLULAR_SWEEPS), ids=list(CELLULAR_SWEEPS))
def test_check_sweep_budgets_the_derived_cellular_analysis(
    name, fake_soapy, monkeypatch, capsys
):
    """check-sweep opens a cellular spec through its runner. The printed sink path is
    formatted, and the tolerance table lists the measurements of the derived `Sweep`
    analysis rather than the fields of the document's `analysis:` block."""
    path = CELLULAR_SWEEPS[name]
    # open_resources chdirs into the spec directory and does not change back
    monkeypatch.chdir(os.getcwd())

    check_sweep.run(str(path))

    out = capsys.readouterr().out
    assert f'for {ss.read_yaml_spec(path).__class__.__name__!r} bindings' in out
    block = re.search(
        r'^sink\.path:\n  Input: +(.*)\n  Formatted: +(.*)$', out, re.MULTILINE
    )
    assert block is not None, out
    template, formatted = (ast.literal_eval(v) for v in block.groups())
    assert formatted.startswith(template.split('{')[0])
    assert path.stem in formatted
    assert '{' not in formatted

    spec = ss.read_yaml_spec(path)
    sweep = ss.lib.actions.cellular.build_cellular_sweep(spec)
    budgeted = {
        name
        for name in sweep.analysis.to_dict()
        if sa.registry[type(getattr(sweep.analysis, name))].tolerance is not None
    }
    assert budgeted, 'the fixture must enable at least one budgeted measurement'
    enabled = {k for k, v in spec.analysis.to_dict().items() if v is not None}
    block = out.split('Numerical tolerance')[1].split('\n\n')[0]
    rows = block.splitlines()[4:]
    table = {row.split()[0] for row in rows}
    assert table == budgeted
    assert table.isdisjoint(enabled - set(sweep.analysis.to_dict()))


def test_cli_run_cellular(fake_soapy, tmp_path, monkeypatch):
    """the fake-soapy cellular sweep writes one store whose rows carry the sample
    rate and analysis bandwidth derived from each capture's channel, the derived
    measurements, the document's `cell_channels` as an attr, and attrs that rebuild
    the document"""
    path = FAKE_SOAPY_CELLULAR_SPEC
    # open_resources chdirs into the spec directory and does not change back
    monkeypatch.chdir(os.getcwd())
    out_path = tmp_path / 'out.zarr.zip'

    sensor_sweep.run(str(path), output_path=str(out_path))

    spec = ss.read_yaml_spec(path, output_path=str(out_path))
    assert isinstance(spec, ss.specs.Cellular5GNRSweep)
    assert ss.read_zarr_spec(out_path, extension_root=path.parent) == spec
    ds = sa.load(out_path)

    rates = derived_sample_rates(spec)
    assert sorted(set(rates)) == [53.76e6, 107.52e6]
    assert ds.sample_rate.values.tolist() == rates
    assert ds.analysis_bandwidth.values.tolist() == [100e6, 100e6, 40e6]
    assert ds.port.values.tolist() == [0, 1, 0]
    assert set(ds.attrs['cell_channels']) == {'3750 MHz', '3960 MHz'}

    sweep = ss.lib.actions.cellular.build_cellular_sweep(spec)
    assert set(sweep.analysis.to_dict()) <= set(ds.data_vars)
    assert 'cellular_5g_pss_correlation' in ds.data_vars
    # the default detector period is one OFDM symbol: 28000 per second at 30 kHz
    # subcarrier spacing with a normal cyclic prefix
    (duration,) = {c.duration for c in spec.captures}
    assert ds.sizes['time_elapsed'] == round(duration * 28000)


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
        preset_capture('single_tone', frequency_offset=-1e6, snr=None),
        preset_capture('single_tone', frequency_offset=1.6e6, snr=None),
    ),
    'noise': (preset_capture('noise', noise_psd=1e-17),),
    'sawtooth': (
        preset_capture('sawtooth', period=RESAMPLE_FILTER['duration'], power=0.0),
    ),
    'dirac_delta': (
        preset_capture('dirac_delta', time=IMPULSE_TIME, power=0.0),
        preset_capture(
            'dirac_delta', **FILTER_ONLY, time=FILTER_ONLY_IMPULSE_TIME, power=-3.0
        ),
    ),
}


def detector(ds, kind):
    return ds.channel_power_time_series.sel(power_detector=kind).values.astype(
        'float64'
    )


def check_single_tone(ds, capture, subtests):
    with subtests.test('rms detector reads 0 dBm'):
        tol = capture_tolerances(capture, SOURCE, ANALYSIS)['channel_power_time_series']
        assert_close(detector(ds, 'rms'), 0.0, rtol=tol.rtol, atol=tol.on_peak.peak)
    attrs = ds.power_spectral_density.attrs
    for row in range(ds.sizes['capture']):
        with subtests.test('tone in the PSD', row=row):
            psd, freqs = mean_psd(ds, row)
            check_tone_psd(psd, freqs, attrs, capture, capture.frequency_offset, None)


def check_noise(ds, capture, subtests):
    """only the noise power survives resampling: the PSD is flat at
    noise_psd * noise_bandwidth per bin, and its mean over ports, windows, and bins
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
            atol=sw.power_analysis.level_tolerance_dB(5 * sigma, power=True)
            + COLA_LEVEL_DB,
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
    tol = capture_tolerances(capture, SOURCE, ANALYSIS)['channel_power_time_series']
    pad = FILTER_SIZE // 2
    with subtests.test('rms detector follows the ramp'):
        skip = math.ceil(pad / SAMPLES_PER_DETECTOR_BIN)
        bins = np.arange(skip, DETECTOR_BINS - skip)
        rms_dB = detector(ds, 'rms')[:, bins]
        assert np.all(np.diff(rms_dB, axis=1) > 0)
        model = np.broadcast_to(sawtooth_rms_model_dB(capture, bins), rms_dB.shape)
        assert_close(
            rms_dB, model, rtol=tol.rtol, atol=sa.util.elementwise_atol(tol, model)
        )


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


CHECKS = {
    'single_tone': check_single_tone,
    'noise': check_noise,
    'sawtooth': check_sawtooth,
    'dirac_delta': check_dirac_delta,
}


@pytest.mark.parametrize('binding', list(CHECKS), ids=list(CHECKS))
def test_in_memory_fidelity(binding, subtests):
    captures = FIDELITY_CAPTURES[binding]
    datasets = run_in_memory(preset_sweep(binding, captures))
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
    reference = run_in_memory(preset_sweep(binding, captures))
    source = SOURCE.replace(array_backend=array_backend)
    datasets = run_in_memory(preset_sweep(binding, captures, source=source))
    backends = ('numpy', 'cupy')
    for i, (capture, ds, ref) in enumerate(zip(captures, datasets, reference)):
        budget = {
            b: capture_tolerances(capture, SOURCE, ANALYSIS, array_backend=b)
            for b in backends
        }
        err = {
            b: correction_error(capture, SOURCE, ANALYSIS, array_backend=b)
            for b in backends
        }
        with subtests.test('iq_waveform', capture=i):
            expected = ref.iq_waveform.values
            peak_roundoff = on_peak_roundoff(np.complex64)
            assert_close(
                ds.iq_waveform.values,
                expected,
                sigma=cross_backend(err['numpy'], err['cupy']),
                # the resample concentrates its roundoff on the samples at the peak,
                # which an rms-referenced sigma understates by the crest factor (near
                # 100 for the impulse captures)
                atol=cross_backend(peak_roundoff, peak_roundoff)
                * np.abs(expected).max(),
            )
        for name in ('power_spectral_density', 'channel_power_time_series'):
            with subtests.test(name, capture=i):
                expected = ref[name].values
                tol = cross_backend(budget['numpy'][name], budget['cupy'][name])
                assert_close(
                    ds[name].values,
                    expected,
                    rtol=tol.rtol,
                    atol=sa.util.elementwise_atol(tol, expected),
                )


# %% bandwidth attrs of a store written from more than one analysis bandwidth

CAPTURE_BANDWIDTH_OUTPUTS = ss.specs.BundledAnalysis.from_dict({
    'iq_waveform': {},
    'channel_power_time_series': {'detector_period': DETECTOR_PERIOD},
    'channel_power_histogram': {
        'detector_period': DETECTOR_PERIOD,
        'power_low': -120,
        'power_high': 0,
        'power_resolution': 1,
    },
})


def test_mixed_analysis_bandwidth_store_resolves_bandwidth_per_capture(
    tmp_path, monkeypatch
):
    """channel power and IQ outputs written from captures at two analysis bandwidths
    carry no `noise_bandwidth` attr, and `striqt.figures` resolves the bandwidth of
    each capture from the coordinates: the finite `analysis_bandwidth`, else the
    `sample_rate`"""
    from striqt.figures import util as figures_util

    # open_resources chdirs into the spec directory and does not change back
    monkeypatch.chdir(os.getcwd())
    captures = (tone_capture(ONE_PORT), tone_capture(ONE_PORT, analysis_bandwidth=5e6))
    path = tmp_path / 'out.zarr'
    sweep = preset_sweep(
        'single_tone',
        captures,
        analysis=CAPTURE_BANDWIDTH_OUTPUTS,
        sink=ss.specs.ZarrSink(path=str(path)),
    )

    with ss.open_resources(sweep) as resources:
        for _ in ss.iterate_sweep(resources):
            pass
    ds = sa.load(path)

    assert ds.analysis_bandwidth.values.tolist() == [float('inf'), 5e6]
    expected = [ONE_PORT['sample_rate'], 5e6]
    outputs = (
        'iq_waveform',
        'channel_power_time_series',
        'channel_power_histogram',
        'channel_power_bin',
    )
    for name in outputs:
        assert 'noise_bandwidth' not in ds[name].attrs, name
    # a histogram resolves through its bin coordinate, as check_noise_floor does
    for name in ('iq_waveform', 'channel_power_time_series', 'channel_power_bin'):
        bandwidth = figures_util.noise_bandwidth(ds, name)
        assert bandwidth is not None, name
        assert bandwidth.values.tolist() == expected, name
