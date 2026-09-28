"""striqt.sensor.specs.sweep: validate_sweep and the located errors it raises from
Sweep.__post_init__
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import msgspec
import pytest
from conftest import SWEEP_DIR
from sweep_strategies import RESOLUTION_MSG, SOURCE, SPG, make_capture, make_sweep

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.analysis.specs.helpers import SpecValidationError

Remap = ss.specs.CaptureRemap


# %% validate_sweep

BAD_RESOLUTION = Remap(
    key='frequency_offset',
    lookup={0.0: {'frequency_resolution': 1e4}, 1e5: {'frequency_resolution': 3e4}},
)
IN_TREE_SWEEPS = sorted([
    SWEEP_DIR / 'synthetic.yaml',
    SWEEP_DIR / 'air7101b.yaml',
    *SWEEP_DIR.glob('fake_soapy-*.yaml'),
    *SWEEP_DIR.glob('site/*.yaml'),
])
DOWNSTREAM_SWEEP = (
    Path(__file__).parents[2]
    / '_training_material'
    / 'downstream test acquisition'
    / 'basic.yaml'
)


def test_validate_sweep_accepts_the_synthetic_sweep(synthetic_sweep):
    assert ss.specs.sweep.validate_sweep(synthetic_sweep) is None


def test_validate_sweep_reports_a_per_source_override():
    """`source_id` selects an `adjust_captures` block that `Sweep.__post_init__` cannot
    reach, since it resolves only the 'defaults' block"""
    sweep = make_sweep(
        captures=(make_capture(),),
        loops=(ss.specs.List(field='frequency_offset', values=(0.0, 1e5)),),
        adjust_captures={'ab12': {'adjust_analysis': BAD_RESOLUTION}},
        analysis=SPG,
    )

    assert ss.specs.sweep.validate_sweep(sweep) is None

    with pytest.raises(msgspec.ValidationError) as excinfo:
        ss.specs.sweep.validate_sweep(sweep, 'ab12')

    message = str(excinfo.value)
    # the measurement that rejected it, the values the failed rule compared, then the
    # place in the sweep that produced them: the loop point and the `captures:` entry
    assert message.startswith('$.analysis.spectrogram: ')
    assert RESOLUTION_MSG in message
    assert '(sample_rate: 1000000.0, frequency_resolution: 30000.0)' in message
    assert message.endswith(
        " - at $.loops: {'frequency_offset': 100000.0} on $.captures[0]"
    )


def test_validate_sweep_ignores_loops_over_sensor_only_fields(monkeypatch):
    """`snr` is invisible to the analysis layer, so both captures project onto one
    `AnalysisCapture` and the pair is validated once"""
    # build before patching: Sweep.__post_init__ validates too, and would be counted
    sweep = make_sweep(
        captures=(make_capture(),),
        loops=(ss.specs.List(field='snr', values=(10.0, 20.0)),),
        analysis=SPG,
    )
    assert len(ss.specs.sequencing.loop_captures(sweep)) == 2

    seen = []
    monkeypatch.setattr(
        sa.registry, 'validate', lambda capture, analysis: seen.append(capture)
    )

    ss.specs.sweep.validate_sweep(sweep)

    assert len(seen) == 1


def test_validate_sweep_skips_an_empty_analysis(monkeypatch):
    sweep = make_sweep(captures=(make_capture(),))

    seen = []
    monkeypatch.setattr(
        sa.registry, 'validate', lambda capture, analysis: seen.append(capture)
    )

    ss.specs.sweep.validate_sweep(sweep)

    assert seen == []


def test_fir_band_errors_name_the_analysis_bandwidth():
    capture = make_capture(sample_rate=1e6, analysis_bandwidth=0.9e6)
    match = re.escape('$.captures[0].analysis_bandwidth: analysis_bandwidth')
    with pytest.raises(SpecValidationError, match=match):
        make_sweep(captures=(capture,))


def test_design_resampler_errors_name_the_sample_rate():
    capture = make_capture(
        host_resample=False, sample_rate=2 * SOURCE.master_clock_rate
    )
    match = re.escape('$.captures[0].sample_rate: upsampling requires host_resample')
    with pytest.raises(SpecValidationError, match=match):
        make_sweep(captures=(capture,))


def test_a_looped_capture_is_located_by_its_loop_point():
    loops = (ss.specs.List(field='lo_shift', values=('none', 'left')),)
    capture = make_capture(analysis_bandwidth=math.inf)
    match = (
        re.escape('frequency shifting may only be applied')
        + '.*'
        + re.escape("at $.loops: {'lo_shift': 'left'} on $.captures[0]")
    )
    with pytest.raises(SpecValidationError, match=match):
        make_sweep(captures=(capture,), loops=loops)


def test_a_trigger_without_its_measurement_is_rejected_at_the_source():
    source = SOURCE.replace(signal_trigger='cellular_5g_pss_sync')
    match = re.escape('$.source.signal_trigger: signal_trigger')
    with pytest.raises(SpecValidationError, match=match):
        make_sweep(captures=(make_capture(),), source=source)


def test_file_sources_skip_the_resampler_design_but_not_the_filter():
    """the file, not `master_clock_rate`, sets the rate the design would run from"""
    source = ss.specs.MATSource(path='absent.mat', master_clock_rate=1e6)
    capture = ss.specs.FileCapture(
        port=0, sample_rate=2e6, duration=1e-3, host_resample=False
    )
    with pytest.raises(ValueError, match='upsampling requires host_resample'):
        ss.lib.compute.design_resampler(capture, source.master_clock_rate)

    cls = ss.bindings.mat_file.sensor.sweep_spec_cls
    assert cls(source=source, captures=(capture,)).captures == (capture,)

    filtered = capture.replace(analysis_bandwidth=capture.sample_rate - 1e3)
    with pytest.raises(SpecValidationError, match='analysis_bandwidth'):
        cls(source=source, captures=(filtered,))


@pytest.mark.parametrize('path', IN_TREE_SWEEPS, ids=lambda p: p.name)
def test_in_tree_sweeps_still_load(path):
    sweep = ss.read_yaml_spec(path)
    assert len(ss.specs.sequencing.loop_captures(sweep)) > 0


@pytest.mark.skipif(not DOWNSTREAM_SWEEP.exists(), reason='not checked in')
def test_downstream_sweep_still_loads():
    sweep = ss.read_yaml_spec(DOWNSTREAM_SWEEP)
    assert len(ss.specs.sequencing.loop_captures(sweep)) > 0
