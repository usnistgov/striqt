"""striqt.sensor.specs.structs: __post_init__ validation and freezing of the sensor
capture, source, loop, remap and sweep specs"""

from __future__ import annotations

import json
import math

import msgspec
import pytest
from conftest import construct_both, raises_on_both_paths
from hypothesis import given
from hypothesis import strategies as st
from site_strategies import SiteCaptureCls, make_site_capture_kws, make_site_sweep
from sweep_strategies import (
    SOURCE,
    CalSourceCls,
    CalSweepCls,
    SweepCls,
    consistent_port_gain,
    duplicate_field_loops,
    make_calibration_capture,
    make_calibration_sweep,
    make_calibration_sweep_kws,
    make_capture,
    make_sweep,
    make_sweep_kws,
    mismatched_port_gain,
    misplaced_repeat_loops,
    scalar_port_tuple_gain,
)

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.analysis.specs.helpers import frozendict

SoapyCapture = ss.specs.SoapyCapture
FileCapture = ss.specs.FileCapture
SoapySource = ss.specs.SoapySource
Air7101BSource = ss.bindings.air7101b.schema.source
List = ss.specs.List
Repeat = ss.specs.Repeat
Remap = ss.specs.CaptureRemap

BASE = {'center_frequency': 1e9, 'duration': 1e-3, 'sample_rate': 1e6}
MCR = {'master_clock_rate': 1e6}
TRIGGERS = ('cellular_5g_pss_sync', 'cellular_5g_sss_sync')

SCALAR_GAIN_MSG = 'gain must be a single number unless multiple ports are specified'
GAIN_COUNT_MSG = 'gain, when specified as a tuple, must match port count'
FILE_RATE_MSG = 'backend_sample_rate is fixed by the file source'
SYNC_MSG = 'time_sync_at must be "open" when gapless'
RETRIES_MSG = 'receive_retries must be 0 when gapless is enabled'
TRIGGER_MSG = 'signal_trigger must be one of'
REPEAT_MSG = r'a repeat may only be the outermost \(first\) loop'
DUPLICATE_MSG = 'more than one loop specified for capture field'
IMPLIED_MSG = (
    'calibration sweeps may only include explicit capture sequences '
    'if implied_loops are specified'
)
SOURCE_CAL_MSG = 'source.calibration must be None for a calibration sweep'

key_pairs = st.tuples(
    st.integers(min_value=-100, max_value=100),
    st.integers(min_value=-100, max_value=100),
)


# %% Capture specs


class TestSoapyCapture:
    @given(port_gain=consistent_port_gain())
    def test_consistent_port_gain_accepted(self, construct, port_gain):
        port, gain = port_gain
        capture = construct(SoapyCapture, port=port, gain=gain, **BASE)
        assert capture.port == port
        assert capture.gain == gain

    @given(port_gain=scalar_port_tuple_gain())
    def test_tuple_gain_with_scalar_port_rejected(self, port_gain):
        port, gain = port_gain
        raises_on_both_paths(
            SoapyCapture, ValueError, SCALAR_GAIN_MSG, port=port, gain=gain, **BASE
        )

    @given(port_gain=mismatched_port_gain())
    def test_gain_tuple_length_must_match_ports(self, port_gain):
        port, gain = port_gain
        raises_on_both_paths(
            SoapyCapture, ValueError, GAIN_COUNT_MSG, port=port, gain=gain, **BASE
        )

    def test_replace_revalidates_gain(self):
        capture = SoapyCapture(port=(0, 1), gain=0.0, **BASE)
        with pytest.raises(ValueError, match=GAIN_COUNT_MSG):
            capture.replace(gain=(1.0,))

    @pytest.mark.xfail(
        strict=True,
        reason=(
            'only gain is validated against the port count; a center_frequency '
            'tuple of the wrong length is accepted and split_capture_ports then '
            'drops the extras'
        ),
    )
    def test_center_frequency_tuple_length_must_match_ports(self):
        kws = dict(BASE, center_frequency=(1e9, 2e9, 3e9))
        with pytest.raises(ValueError):
            SoapyCapture(port=(0, 1), gain=0.0, **kws)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            'port is a VarTuple union that SpecBase never freezes on direct '
            'construction; the list then fails the lru_cache in _validate_multichannel'
        ),
    )
    def test_list_port_direct_is_frozen(self):
        capture = SoapyCapture(port=[0, 1], gain=0.0, **BASE)
        assert capture.port == (0, 1)

    def test_air7101b_capture_inherits_gain_check(self):
        cls = ss.bindings.air7101b.schema.capture
        assert issubclass(cls, SoapyCapture)
        with pytest.raises(ValueError, match=GAIN_COUNT_MSG):
            cls(port=(0, 1), gain=(1.0,), **BASE)


class TestFileCapture:
    def test_backend_sample_rate_none_ok(self, construct):
        capture = construct(FileCapture, port=0, duration=1e-3, sample_rate=1e6)
        assert capture.backend_sample_rate is None

    def test_backend_sample_rate_rejected(self):
        raises_on_both_paths(
            FileCapture,
            TypeError,
            FILE_RATE_MSG,
            port=0,
            duration=1e-3,
            sample_rate=1e6,
            backend_sample_rate=1e6,
        )


class TestExtensionCapture:
    """Meta bounds and field types on a capture subclass defined out of tree"""

    def test_nan_fails_a_lower_bound(self):
        with pytest.raises(msgspec.ValidationError, match='mast_height'):
            SiteCaptureCls.from_dict(make_site_capture_kws(mast_height=math.nan))

    def test_null_analysis_bandwidth_is_rejected(self):
        # documented as current: inf, not null, disables the analysis filter
        with pytest.raises(
            msgspec.ValidationError, match='Expected `float`, got `null`'
        ):
            SiteCaptureCls.from_dict(make_site_capture_kws(analysis_bandwidth=None))


# %% SoapySource


class TestGapless:
    def test_non_gapless_ignores_sync_and_retries(self, construct):
        source = construct(
            SoapySource, gapless=False, time_sync_at='acquire', receive_retries=3, **MCR
        )
        assert source.time_sync_at == 'acquire'
        assert source.receive_retries == 3

    def test_gapless_requires_open_time_sync(self):
        raises_on_both_paths(SoapySource, ValueError, SYNC_MSG, gapless=True, **MCR)

    def test_gapless_requires_zero_retries(self):
        raises_on_both_paths(
            SoapySource,
            ValueError,
            RETRIES_MSG,
            gapless=True,
            time_sync_at='open',
            receive_retries=3,
            **MCR,
        )

    def test_gapless_open_zero_retries_ok(self, construct):
        source = construct(SoapySource, gapless=True, time_sync_at='open', **MCR)
        assert source.gapless is True

    def test_air7101b_nonzero_default_retries_forbid_gapless(self):
        with pytest.raises(ValueError, match=RETRIES_MSG):
            Air7101BSource(gapless=True, time_sync_at='open')
        assert Air7101BSource(gapless=True, time_sync_at='open', receive_retries=0)


class TestSignalTrigger:
    @pytest.mark.parametrize('name', [*TRIGGERS, None])
    def test_registered_names_accepted(self, construct, name):
        source = construct(SoapySource, signal_trigger=name, **MCR)
        assert source.signal_trigger == name

    @given(
        name=st.text(min_size=1).filter(lambda s: s not in sa.registry.signal_trigger)
    )
    def test_unregistered_name_rejected(self, name):
        raises_on_both_paths(
            SoapySource, ValueError, TRIGGER_MSG, signal_trigger=name, **MCR
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            'signal_trigger is annotated Union[str, AnalysisGroup, None] but '
            '__post_init__ only accepts registered trigger names'
        ),
    )
    def test_analysis_group_accepted(self):
        # BundledTriggers() cannot be instantiated on py3.14, so the bare base is used
        group = sa.specs.AnalysisGroup()
        source = SoapySource(signal_trigger=group, **MCR)
        assert isinstance(source.signal_trigger, sa.specs.AnalysisGroup)


# %% Loops, plot options and extensions


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


def test_plot_options_reject_unknown_fields():
    with pytest.raises(msgspec.ValidationError, match='foo'):
        ss.specs.PlotOptions.from_dict({
            'data': {'sweep_index': 0},
            'plotter': {'foo': 1},
        })


def test_extension_sink_is_kept_as_a_string():
    ext = ss.specs.Extension.from_dict({'sink': 'striqt.sensor.sinks.NoSink'})
    assert (
        make_site_sweep(extensions=ext).extensions.sink == 'striqt.sensor.sinks.NoSink'
    )


# %% CaptureRemap


class TestCaptureRemap:
    """decoding of multi-field lookup keys"""

    def test_multi_key_json_string_keys_become_tuples(self, construct):
        remap = construct(Remap, key=('a', 'b'), lookup={'[1, 2]': 5, '["x", 3]': 6})
        assert isinstance(remap.lookup, frozendict)
        assert remap.lookup == {(1, 2): 5, ('x', 3): 6}
        hash(remap)

    def test_multi_key_tuple_keys_pass_through(self):
        remap = Remap(key=('a', 'b'), lookup={(1, 2): 5})
        assert remap.lookup == {(1, 2): 5}

    @given(keys=st.lists(key_pairs, unique=True, max_size=6))
    def test_multi_key_roundtrip(self, construct, keys):
        lookup = {json.dumps(list(k)): i for i, k in enumerate(keys)}
        remap = construct(Remap, key=('a', 'b'), lookup=lookup)
        assert remap.lookup == {k: i for i, k in enumerate(keys)}

    @pytest.mark.parametrize('key', ['a', ('a',)])
    def test_single_field_key_leaves_string_keys(self, construct, key):
        remap = construct(Remap, key=key, lookup={'[1]': 5})
        assert remap.lookup == {'[1]': 5}

    @pytest.mark.xfail(
        strict=True,
        reason=(
            'CaptureRemap.__post_init__ leaks msgspec.DecodeError/TypeError from '
            'json.decode instead of a ValidationError naming the lookup key'
        ),
    )
    @pytest.mark.parametrize('lookup', [{'foo': 5}, {1: 5}], ids=['text', 'int'])
    def test_multi_key_undecodable_key_is_a_validation_error(self, lookup):
        with pytest.raises(msgspec.ValidationError, match='lookup'):
            Remap(key=('a', 'b'), lookup=lookup)

    def test_nested_lookup_values_are_frozen(self, construct):
        remap = construct(Remap, key='a', lookup={'x': [1, 2]})
        assert remap.lookup == frozendict({'x': (1, 2)})

    def test_replace_reconverts_keys(self):
        remap = Remap(key=('a', 'b'), lookup={(1, 2): 5})
        assert remap.replace(lookup={'[3, 4]': 1}).lookup == {(3, 4): 1}


# %% Sweep


class _ConflictingCapture(ss.specs.SingleToneCapture, frozen=True, kw_only=True):
    spectrogram: int = 0


class TestSweepLoops:
    @given(loops=misplaced_repeat_loops())
    def test_repeat_must_be_first(self, loops):
        kws = make_sweep_kws(captures=(make_capture(),), loops=loops)
        raises_on_both_paths(SweepCls, msgspec.ValidationError, REPEAT_MSG, **kws)

    def test_two_repeats_rejected(self):
        with pytest.raises(msgspec.ValidationError, match=REPEAT_MSG):
            make_sweep(loops=(Repeat(count=2), Repeat(count=3)))

    @given(loops_field=duplicate_field_loops())
    def test_duplicate_loop_field_rejected(self, loops_field):
        loops, field = loops_field
        match = f"{DUPLICATE_MSG} '{field}'"
        raises_on_both_paths(
            SweepCls, msgspec.ValidationError, match, **make_sweep_kws(loops=loops)
        )


class TestSweepCaptures:
    @pytest.mark.parametrize('cls', [ss.specs.Sweep, ss.specs.CalibrationSweep])
    def test_bare_sweep_is_not_constructible(self, cls):
        # the TypeError comes from msgspec.structs.fields on the unbound capture
        # TypeVar, not from a guard, so there is no message to match
        with pytest.raises(TypeError):
            cls(source=SOURCE)

    def test_capture_field_conflicting_with_measurement_name(self):
        capture = _ConflictingCapture(port=0, sample_rate=1e6, duration=1e-3)
        match = r"capture fields \('spectrogram',\) conflict with measurements"
        with pytest.raises(AttributeError, match=match):
            make_sweep(captures=(capture,))


class TestSweepAdjustCaptures:
    def test_lookup_keys_are_converted_before_freezing(self, construct):
        remap = Remap(key='frequency_offset', lookup={'1': 2.0})
        adjust = {'defaults': {'snr': remap}}
        sweep = construct(SweepCls, **make_sweep_kws(adjust_captures=adjust))
        assert isinstance(sweep.adjust_captures, frozendict)
        lookup = sweep.adjust_captures['defaults']['snr'].lookup
        assert lookup == {1.0: 2.0}
        assert isinstance(next(iter(lookup)), float)

    @pytest.mark.parametrize(
        'adjust, match',
        [
            ({'zz': {'snr': 1.0}}, 'is not "global" or a hex string'),
            (
                {'defaults': {'duration': 1.0}},
                "capture field 'duration' is not allowed by adjust_captures",
            ),
        ],
    )
    def test_invalid_adjustments_rejected(self, adjust, match):
        kws = make_sweep_kws(adjust_captures=adjust)
        raises_on_both_paths(SweepCls, msgspec.ValidationError, match, **kws)

    @pytest.mark.xfail(
        strict=True,
        raises=NameError,
        reason=(
            'the tuple branch of _get_capture_adjust_map references an undefined '
            'name source_fields'
        ),
    )
    def test_tuple_of_pairs_form_is_accepted(self):
        sweep = make_sweep(adjust_captures=(('defaults', {'snr': 1.0}),))
        assert sweep.adjust_captures['defaults'] == {'snr': 1.0}


# %% CalibrationSweep


class TestCalibrationSweep:
    def test_single_capture_needs_no_implied_loops(self):
        for sweep in construct_both(CalSweepCls, **make_calibration_sweep_kws()):
            assert len(sweep.captures) == 1

    def test_multiple_captures_need_implied_loops(self):
        captures = (make_calibration_capture(), make_calibration_capture(gain=-10))
        kws = make_calibration_sweep_kws(captures=captures)
        raises_on_both_paths(CalSweepCls, TypeError, IMPLIED_MSG, **kws)

        peripheral = ss.specs.ManualYFactorPeripheral(
            enr=10, ambient_temperature=290, implied_loops=('gain',)
        )
        kws = make_calibration_sweep_kws(captures=captures, calibration=peripheral)
        for sweep in construct_both(CalSweepCls, **kws):
            assert len(sweep.captures) == 2

    def test_source_calibration_must_be_none(self):
        kws = make_calibration_sweep_kws(source=CalSourceCls(calibration='cal.nc'))
        raises_on_both_paths(CalSweepCls, ValueError, SOURCE_CAL_MSG, **kws)

    def test_default_options_loop_only_nyquist(self):
        loops = (List(field='analysis_bandwidth', values=(0.5e6, 2e6, math.inf)),)
        sweep = make_calibration_sweep(loops=loops)
        bandwidths = {
            c.analysis_bandwidth for c in ss.specs.helpers.loop_captures(sweep)
        }
        assert bandwidths == {0.5e6, math.inf}
