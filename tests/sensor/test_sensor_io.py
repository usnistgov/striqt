"""striqt.sensor.lib.io: read_yaml_spec and read_json_spec, including the `extensions:`
block and YAML files shaped like the downstream sensor and fragment files"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import msgspec
import pytest
from conftest import SWEEP_DIR
from site_strategies import EXT, SiteCalCaptureCls, SiteCaptureCls
from sweep_strategies import CalSweepCls, loop_fields

import striqt.analysis as sa
import striqt.sensor as ss

FRAGMENTS = SWEEP_DIR / 'fragments'

INLINE_SPEC = """
sensor_binding: single_tone
extensions:
  import_path: {import_path}
  import_name: {import_name}
source:
  master_clock_rate: 125e6
  num_rx_ports: 2
captures:
  - port: 0
    duration: 1e-3
    sample_rate: 1e6
"""

INLINE_SITE_SPEC = f"""
sensor_binding: site_single_tone
extensions:
  import_path: {SWEEP_DIR / 'src'}
  import_name: extensions
source:
  master_clock_rate: 125e6
  num_rx_ports: 2
captures:
  - port: 0
    center_frequency: 3750e6
    gain: 0
    duration: 1e-3
    sample_rate: 1e6
"""


def _write_spec(write_yaml, relpath, import_path, import_name='extensions'):
    return write_yaml(
        relpath, INLINE_SPEC.format(import_path=import_path, import_name=import_name)
    )


# %% built-in bindings


def test_mock_source_swaps_the_source_but_keeps_the_captures(write_yaml):
    spec = write_yaml('spec.yaml', INLINE_SITE_SPEC + 'mock_source: warmup\n')
    sweep = ss.read_yaml_spec(spec)
    assert type(sweep).__name__ == 'mock_warmup_site_single_tone'
    assert isinstance(sweep.source, ss.specs.NoSource)
    assert isinstance(sweep.captures[0], SiteCaptureCls)


def test_sink_including_a_source_fragment_is_rejected(write_yaml):
    write_yaml('frag.yaml', 'master_clock_rate: 125e6\n')
    spec = write_yaml('spec.yaml', INLINE_SITE_SPEC + 'sink: !include frag.yaml\n')
    with pytest.raises(msgspec.ValidationError, match=r'master_clock_rate.*\$\.sink'):
        ss.read_yaml_spec(spec)


# %% the extensions: block


def test_legacy_extensions_keys_are_rejected_before_import(write_yaml):
    text = INLINE_SITE_SPEC.replace(
        '  import_name: extensions\n', '  import_name: extensions\n  sweep_struct: x\n'
    )
    spec = write_yaml('spec.yaml', text)
    with pytest.raises(msgspec.ValidationError, match='sweep_struct'):
        ss.read_yaml_spec(spec)


def test_import_path_is_relative_to_the_spec_directory(
    write_yaml, isolated_extension_import, caplog, tmp_path
):
    write_yaml('ext/extensions.py', "MARKER = 'tmp'\n")
    spec = _write_spec(write_yaml, 'sub/spec.yaml', '../ext')

    ss.read_yaml_spec(spec)

    assert sys.modules['extensions'].MARKER == 'tmp'
    assert Path(sys.path[0]).resolve() == (tmp_path / 'ext').resolve()
    assert 'did not bind a sensor' in caplog.text


def test_import_name_null_only_extends_sys_path(write_yaml, isolated_extension_import):
    write_yaml('ext/placeholder.txt', '')
    spec = _write_spec(write_yaml, 'sub/spec.yaml', '../ext', import_name='null')

    ss.read_yaml_spec(spec)

    assert Path(sys.path[0]).resolve() == (spec.parent / '../ext').resolve()
    assert 'extensions' not in sys.modules


def test_rereading_a_spec_warns_that_nothing_new_was_bound(site_spec_path, caplog):
    # documented as current: the binding count is compared before and after every
    # import, so the second read of any spec that shares a module logs this warning
    ss.read_yaml_spec(site_spec_path)
    assert 'did not bind a sensor' in caplog.text


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason='_import_extensions_from_spec checks that the root directory exists but '
    'not that root/import_path does',
)
def test_missing_import_path_raises_file_not_found(
    write_yaml, isolated_extension_import
):
    spec = _write_spec(write_yaml, 'sub/spec.yaml', 'nope')
    with pytest.raises(FileNotFoundError, match='nope'):
        ss.read_yaml_spec(spec)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='importlib.import_module returns the cached module from the first spec, '
    'so a second sensor directory with the same import_name is never imported',
)
def test_second_directory_with_the_same_import_name_is_imported(
    write_yaml, isolated_extension_import
):
    specs = {}
    for tag in 'ab':
        write_yaml(f'{tag}/ext/extensions.py', f"MARKER = '{tag}'\n")
        specs[tag] = _write_spec(write_yaml, f'{tag}/sub/spec.yaml', '../ext')

    ss.read_yaml_spec(specs['a'])
    assert sys.modules['extensions'].MARKER == 'a'
    ss.read_yaml_spec(specs['b'])
    assert sys.modules['extensions'].MARKER == 'b'


# %% site-shaped sweep files


def test_site_sweep_decodes(site_sweep):
    assert type(site_sweep).__name__ == 'site_single_tone'
    assert isinstance(site_sweep.captures[0], SiteCaptureCls)
    assert site_sweep.peripherals == EXT.SitePeripherals(control_orientation=False)
    assert site_sweep.description.version == 'v2026-07-01'
    assert site_sweep.options.skip_warmup is True

    # adjust_analysis is untyped until it is applied to a measurement
    (c,) = site_sweep.captures
    assert dict(c.adjust_analysis) == {
        'time_statistic': ('mean', 'max'),
        'frame_slots': None,
    }

    hint = site_sweep.plot_hint
    assert hint.data.select['spectrogram_time'] == 'slice(0, 20e-3)'
    assert hint.variables['cellular_5g_pss_correlation'] == {'dB': True}

    a = site_sweep.analysis
    assert a.power_spectral_density.fractional_overlap == Fraction(13, 28)
    assert a.power_spectral_density.window_fill == Fraction(15, 28)
    assert a.power_spectral_density.window == ('kaiser', 11.884)
    assert a.power_spectral_density.time_statistic == ('mean', 0.5, 0.99, 'max')
    assert a.spectrogram.integration_bandwidth == pytest.approx(360e3)
    assert a.channel_power_time_series.detector_period == Fraction(1, 28000)


def test_per_port_fragment_decodes():
    (tree,) = sa.lib.io.decode_from_yaml_file(FRAGMENTS / 'captures' / 'per_port.yaml')
    c = SiteCaptureCls.from_dict(tree)
    assert c.center_frequency == (7350e6, 7350e6)
    assert c.external_lo_frequency == (11.38e9, 11.38e9)
    assert c.channel_name == ('7350 MHz', '7350 MHz')
    assert math.isnan(c.azimuth_offset)
    assert c.adjust_analysis['delay'] == '50e-6'
    assert c.adjust_analysis['frame_slots'] == 'dddsuudddddddsuudddd'


def test_calibration_sweeps_decode(calibration_sweep, site_calibration_sweep):
    assert isinstance(calibration_sweep, CalSweepCls)

    sweep = site_calibration_sweep
    assert type(sweep).__name__ == 'site_single_tone_calibration'
    assert sweep.calibration.enr == pytest.approx(20.9)
    assert sweep.source.calibration is None
    assert isinstance(sweep.captures[0], SiteCalCaptureCls)
    assert sweep.captures[0].noise_diode_enabled is False
    assert loop_fields(sweep) == [
        'port',
        'noise_diode_enabled',
        'sample_rate',
        'center_frequency',
        'gain',
        'analysis_bandwidth',
        'lo_shift',
    ]
