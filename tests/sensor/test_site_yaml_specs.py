"""decoding sweep YAML files shaped like the downstream sensor and fragment files"""

from __future__ import annotations

import math
from fractions import Fraction

import msgspec
import pytest
from conftest import SWEEP_DIR
from site_strategies import (
    EXT,
    OTHER_ID,
    RADIO_ID,
    SiteCaptureCls,
    SurveyCaptureCls,
    SurveySweepCls,
    capture_dict,
    make_site_sweep,
)

import striqt.analysis as sa
import striqt.sensor as ss

H = ss.specs.helpers
FRAGMENTS = SWEEP_DIR / 'fragments'

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


def test_site_sweep_decodes_to_the_extension_binding(site_sweep):
    assert type(site_sweep).__name__ == 'site_single_tone'
    assert isinstance(site_sweep.captures[0], SiteCaptureCls)
    assert site_sweep.peripherals == EXT.SitePeripherals(control_orientation=False)
    assert site_sweep.description.version == 'v2026-07-01'
    assert site_sweep.options.skip_warmup is True


def test_list_include_override_clears_the_trigger(site_sweep):
    assert site_sweep.source.signal_trigger is None
    assert site_sweep.source.master_clock_rate == pytest.approx(125e6)


def test_number_strings_convert_when_the_field_type_is_known(site_sweep):
    (c,) = site_sweep.captures
    assert type(c.sample_rate) is float and c.sample_rate == pytest.approx(53.76e6)
    assert c.center_frequency == pytest.approx(3750e6)
    assert c.host_resample is True and c.backend_sample_rate is None
    # adjust_analysis is untyped until it is applied to a measurement
    assert dict(c.adjust_analysis) == {
        'time_statistic': ('mean', 'max'),
        'frame_slots': None,
    }


def test_sites_glob_merge(site_sweep):
    adjust = site_sweep.adjust_captures
    assert set(adjust) == {'defaults', RADIO_ID, OTHER_ID}
    assert adjust[OTHER_ID]['mast_height'] is None
    assert adjust[RADIO_ID]['elevation_offset'] == 0


def test_second_glob_replaces_the_whole_source_block(site_survey_sweep):
    # documented as current: multi-file includes merge one level deep
    block = site_survey_sweep.adjust_captures[RADIO_ID]
    assert set(block) == {
        'radio_name',
        'site_name',
        'antenna_name',
        'tx1_power',
        'test_case',
    }


def test_nan_string_fixed_value_becomes_float_nan(site_survey_sweep):
    c = H.loop_captures(site_survey_sweep, source_id=RADIO_ID, limit=1)[0]
    assert isinstance(c, SurveyCaptureCls)
    assert math.isnan(c.tx1_power)
    assert c.test_case == 3


def test_per_port_fragment_decodes_but_the_channel_lookup_fails(site_sweep):
    (tree,) = sa.lib.io.decode_from_yaml_file(FRAGMENTS / 'captures' / 'per_port.yaml')
    c = SiteCaptureCls.from_dict(tree)
    assert c.center_frequency == (7350e6, 7350e6)
    assert c.external_lo_frequency == (11.38e9, 11.38e9)
    assert c.channel_name == ('7350 MHz', '7350 MHz')
    assert math.isnan(c.azimuth_offset)
    assert c.adjust_analysis['delay'] == '50e-6'
    assert c.adjust_analysis['frame_slots'] == 'dddsuudddddddsuudddd'

    # documented as current; see test_per_port_miss_uses_the_default
    with pytest.raises(KeyError):
        H.loop_captures(site_sweep.replace(captures=(c,)), source_id=RADIO_ID)


def test_nan_fails_a_lower_bound():
    with pytest.raises(msgspec.ValidationError, match='mast_height'):
        SiteCaptureCls.from_dict(capture_dict(mast_height=math.nan))


def test_plot_hint_decodes(site_sweep):
    hint = site_sweep.plot_hint
    assert hint.data.sweep_index == -1
    assert hint.data.select['spectrogram_time'] == 'slice(0, 20e-3)'
    assert hint.plotter.style == 'striqt.figures.presentation_full_width'
    assert '{channel_name}' in hint.plotter.suptitle_fmt
    assert set(hint.variables) == {'spectrogram', 'cellular_5g_pss_correlation'}
    assert hint.variables['cellular_5g_pss_correlation'] == {'dB': True}


def test_plot_hint_rejects_unknown_fields():
    with pytest.raises(msgspec.ValidationError, match='foo'):
        ss.specs.PlotOptions.from_dict({
            'data': {'sweep_index': 0},
            'plotter': {'foo': 1},
        })


def test_analysis_fragment_fractions_and_anchors(site_sweep):
    a = site_sweep.analysis
    assert a.power_spectral_density.fractional_overlap == Fraction(13, 28)
    assert a.power_spectral_density.window_fill == Fraction(15, 28)
    assert a.power_spectral_density.window == ('kaiser', 11.884)
    assert a.power_spectral_density.time_statistic == ('mean', 0.5, 0.99, 'max')
    assert a.spectrogram.integration_bandwidth == pytest.approx(360e3)
    assert a.channel_power_time_series.detector_period == Fraction(1, 28000)


def test_survey_loops_decode_and_expand(site_survey_sweep):
    loops = site_survey_sweep.loops
    assert [type(l).__name__ for l in loops] == [
        'Repeat',
        'Range',
        'Range',
        'List',
        'Range',
        'FrequencyBinRange',
    ]
    assert [len(l.get_points()) for l in loops[1:]] == [9, 4, 2, 4, 9]

    captures = H.loop_captures(site_survey_sweep, source_id=RADIO_ID)
    # the repeat loop is left to the sweep runner, so it does not multiply the count
    assert len(captures) == 9 * 4 * 2 * 4 * 9
    c = captures[0]
    assert (c.azimuth, c.elevation) == (-180.0, 0.0)
    assert type(c.azimuth_repeat) is int
    assert dict(c.adjust_analysis) == {'frequency_offset': -46.08e6}
    assert c.frequency_offset == pytest.approx(-3e6)


def test_frequency_bin_range_endpoints_are_inclusive():
    loop = ss.specs.FrequencyBinRange(
        field='frequency_offset',
        isin='analysis',
        start=-46.08e6,
        stop=46.08e6,
        step=1.44e6,
    )
    points = loop.get_points()
    assert len(points) == 65
    assert points[0] == pytest.approx(-46.08e6)
    assert points[-1] == pytest.approx(46.08e6)


def test_meta_bounds_are_enforced_at_expansion_not_decode(site_survey_sweep):
    loops = (
        ss.specs.Range(field='azimuth', start=-45, stop=226, step=2.5),
        ss.specs.Range(field='elevation', start=0, stop=5, step=5),
    )
    sweep = site_survey_sweep.replace(loops=loops)
    with pytest.raises(msgspec.ValidationError, match='<= 180'):
        H.loop_captures(sweep, source_id=RADIO_ID)


def test_capture_post_init_is_enforced_at_expansion(site_survey_sweep):
    loops = (ss.specs.Range(field='azimuth', start=-180, stop=180, step=90),)
    sweep = site_survey_sweep.replace(loops=loops)
    with pytest.raises(msgspec.ValidationError, match='azimuth and elevation'):
        H.loop_captures(sweep, source_id=RADIO_ID)


def test_range_loop_on_an_int_field_accepts_integral_floats():
    loops = (ss.specs.Range(field='azimuth_repeat', start=0, stop=3, step=1),)
    sweep = make_site_sweep(cls=SurveySweepCls, loops=loops)
    assert [c.azimuth_repeat for c in H.loop_captures(sweep)] == [0, 1, 2, 3]


def test_range_loop_on_an_int_field_rejects_fractions():
    loops = (ss.specs.Range(field='azimuth_repeat', start=0, stop=3, step=1.5),)
    sweep = make_site_sweep(cls=SurveySweepCls, loops=loops)
    with pytest.raises(msgspec.ValidationError, match='Expected `int`'):
        H.loop_captures(sweep)


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


def test_legacy_extensions_keys_are_rejected_before_import(write_yaml):
    text = INLINE_SITE_SPEC.replace(
        '  import_name: extensions\n', '  import_name: extensions\n  sweep_struct: x\n'
    )
    spec = write_yaml('spec.yaml', text)
    with pytest.raises(msgspec.ValidationError, match='sweep_struct'):
        ss.read_yaml_spec(spec)


def test_null_analysis_bandwidth_is_rejected():
    # documented as current: inf, not null, disables the analysis filter
    with pytest.raises(msgspec.ValidationError, match='Expected `float`, got `null`'):
        SiteCaptureCls.from_dict(capture_dict(analysis_bandwidth=None))


def test_duplicate_top_level_key_keeps_the_last(write_yaml):
    spec = write_yaml('spec.yaml', 'options: {a: 1}\noptions: {b: 2}\n')
    assert sa.lib.io.decode_from_yaml_file(spec) == {'options': {'b': 2}}


def test_extensions_sink_is_kept_as_a_string():
    ext = ss.specs.Extension.from_dict({'sink': 'striqt.sensor.sinks.NoSink'})
    assert (
        make_site_sweep(extensions=ext).extensions.sink == 'striqt.sensor.sinks.NoSink'
    )


def test_sensor_kwarg_typo_is_a_type_error():
    with pytest.raises(TypeError, match="unexpected keyword argument 'peripherals'"):
        ss.bindings.Sensor(
            source_cls=EXT.SiteToneSource, peripherals=ss.peripherals.NoPeripherals
        )
