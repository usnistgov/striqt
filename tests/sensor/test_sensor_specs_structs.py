"""striqt.sensor.specs.structs: validation and freezing of the sensor specs"""

from __future__ import annotations

import json
import math
import re
from typing import Optional

import msgspec
import pytest
from conftest import construct_both, raises_on_both_paths
from hypothesis import given, strategies as st
from site_strategies import SiteCaptureCls, make_site_capture_kws
from sweep_strategies import (
    RESOLUTION_MSG,
    SPG,
    YFACTOR_ENR_DB,
    CalSourceCls,
    CalSweepCls,
    PlanCls,
    SweepCls,
    make_calibration_capture,
    make_calibration_sweep,
    make_calibration_sweep_kws,
    make_capture,
    make_cell_channel,
    make_sweep,
    make_sweep_kws,
)

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.analysis.specs.helpers import frozendict
from striqt.sensor.specs.types import PerPort

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

SCALAR_GAIN_MSG = 'gain must be a single value'
GAIN_COUNT_MSG = 'must match port count'
PORT_UNIQUE_MSG = 'port indices must be unique'
FILE_RATE_MSG = 'fixed by the file source'
TRIGGER_MSG = 'signal_trigger must be one of'
REPEAT_MSG = re.escape('Expected a `repeat` loop only as the outermost')
DUPLICATE_MSG = 'Expected at most one loop over field'
COLLISION_MSG = 'looped or set per capture, not both'
IMPLIED_MSG = re.escape('$.captures:') + '.*implied_loops'
SOURCE_CAL_MSG = re.escape('$.source.calibration:')

key_pairs = st.tuples(
    st.integers(min_value=-100, max_value=100),
    st.integers(min_value=-100, max_value=100),
)


# %% Capture specs


class _LabeledCapture(SoapyCapture, frozen=True, kw_only=True):
    """a downstream capture with a `PerPort[str]` label, as the site extensions declare"""

    label: Optional[PerPort[str]] = None


class TestSoapyCapture:
    @pytest.mark.parametrize(
        'port, gain',
        [(0, 1.0), ((0, 1), 1.0), ((0, 1), (1.0, 2.0))],
        ids=['scalar_port', 'shared_gain', 'gain_per_port'],
    )
    def test_consistent_port_gain_accepted(self, port, gain):
        for capture in construct_both(SoapyCapture, port=port, gain=gain, **BASE):
            assert (capture.port, capture.gain) == (port, gain)

    @pytest.mark.parametrize(
        'gain', [(1.0,), (1.0, 2.0)], ids=['one_element', 'two_elements']
    )
    def test_tuple_gain_with_scalar_port_rejected(self, gain):
        raises_on_both_paths(
            SoapyCapture, ValueError, SCALAR_GAIN_MSG, port=0, gain=gain, **BASE
        )

    @pytest.mark.parametrize(
        'gain', [(1.0,), (1.0, 2.0, 3.0)], ids=['shorter', 'longer']
    )
    def test_gain_tuple_length_must_match_ports(self, gain):
        raises_on_both_paths(
            SoapyCapture, ValueError, GAIN_COUNT_MSG, port=(0, 1), gain=gain, **BASE
        )

    def test_replace_revalidates_gain(self):
        capture = SoapyCapture(port=(0, 1), gain=0.0, **BASE)
        with pytest.raises(ValueError, match=GAIN_COUNT_MSG):
            capture.replace(gain=(1.0,))

    @pytest.mark.parametrize('port', [(0, 0), (1, 0, 1)], ids=['pair', 'triple'])
    def test_repeated_port_index_rejected(self, port):
        raises_on_both_paths(
            SoapyCapture, ValueError, PORT_UNIQUE_MSG, port=port, gain=0.0, **BASE
        )

    def test_unordered_distinct_ports_accepted(self):
        for capture in construct_both(SoapyCapture, port=(1, 0), gain=0.0, **BASE):
            assert capture.port == (1, 0)

    def test_center_frequency_tuple_length_must_match_ports(self):
        kws = dict(BASE, center_frequency=(1e9, 2e9, 3e9))
        raises_on_both_paths(
            SoapyCapture,
            ValueError,
            'center_frequency, when specified as a tuple, must match port count',
            port=(0, 1),
            gain=0.0,
            **kws,
        )

    @pytest.mark.parametrize('lo', [(1e9,), (1e9, 2e9, 3e9)], ids=['shorter', 'longer'])
    def test_external_lo_frequency_tuple_length_must_match_ports(self, lo):
        raises_on_both_paths(
            SoapyCapture,
            ValueError,
            'external_lo_frequency, when specified as a tuple, must match port count',
            port=(0, 1),
            gain=0.0,
            external_lo_frequency=lo,
            **BASE,
        )

    def test_external_lo_frequency_tuple_per_port_accepted(self):
        lo = (1e9, 2e9)
        for capture in construct_both(
            SoapyCapture, port=(0, 1), gain=0.0, external_lo_frequency=lo, **BASE
        ):
            assert capture.external_lo_frequency == lo

    def test_external_lo_frequency_tuple_with_scalar_port_rejected(self):
        raises_on_both_paths(
            SoapyCapture,
            ValueError,
            'external_lo_frequency must be a single value',
            port=0,
            gain=0.0,
            external_lo_frequency=(1e9,),
            **BASE,
        )

    @pytest.mark.parametrize(
        'port, label',
        [((0, 1), ('omni', 'panel', 'dish')), (0, ('omni',))],
        ids=['wrong_length', 'scalar_port'],
    )
    def test_downstream_per_port_str_field_rejected(self, port, label):
        raises_on_both_paths(
            _LabeledCapture,
            ValueError,
            'label',
            port=port,
            gain=0.0,
            label=label,
            **BASE,
        )

    def test_downstream_per_port_str_field_accepted(self):
        label = ('omni', 'panel')
        for capture in construct_both(
            _LabeledCapture, port=(0, 1), gain=0.0, label=label, **BASE
        ):
            assert capture.label == label

    @pytest.mark.xfail(
        strict=True,
        reason=(
            'port is a VarTuple union that SpecBase never freezes on direct '
            'construction; the capture keeps the list'
        ),
    )
    def test_list_port_direct_is_frozen(self):
        capture = SoapyCapture(port=[0, 1], gain=0.0, **BASE)
        assert capture.port == (0, 1)


class TestFileCapture:
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


# %% Bursts


class TestBursts:
    def test_capture_burst_defaults_to_one_acquisition(self):
        for capture in construct_both(SoapyCapture, port=0, gain=0, **BASE):
            assert capture.burst == 1

    @pytest.mark.parametrize('burst', [0, -1])
    def test_capture_burst_below_one_is_rejected_on_convert(self, burst):
        kws = {'port': 0, 'gain': 0, **BASE}
        with pytest.raises(msgspec.ValidationError, match=r'\$\.burst'):
            SoapyCapture.from_dict({**kws, 'burst': burst})
        with pytest.raises(msgspec.ValidationError, match=r'\$\.burst'):
            SoapyCapture(**kws).replace(burst=burst)

    def test_source_bursts_block_defaults(self):
        """the defaults tolerate a gap, hold one acquisition ahead of analysis, and
        run the signal trigger once per burst"""
        for source in construct_both(SoapySource, **MCR):
            assert source.bursts == ss.specs.BurstOptions(
                on_gap='skip', buffers=1, trigger_on='first'
            )

    @pytest.mark.parametrize(
        'field, value',
        [('on_gap', 'drop'), ('buffers', 0), ('trigger_on', 'last')],
        ids=['on_gap', 'buffers', 'trigger_on'],
    )
    def test_source_bursts_block_rejects_values_outside_the_contract(
        self, field, value
    ):
        match = re.escape(f'$.{field}')
        with pytest.raises(msgspec.ValidationError, match=match):
            ss.specs.BurstOptions.from_dict({field: value})
        with pytest.raises(msgspec.ValidationError, match=match):
            ss.specs.BurstOptions().replace(**{field: value})

    def test_source_accepts_the_bursts_block_as_a_dict(self):
        """the binding call form of a controller passes the fields of `to_dict`"""
        source = SoapySource(bursts={'buffers': 2}, **MCR)
        assert source.bursts == ss.specs.BurstOptions(buffers=2)
        assert hash(source) == hash(SoapySource(**source.to_dict()))

    def test_calibration_rejects_a_capture_burst(self):
        match = re.escape('$.captures[0].burst: Expected 1')
        with pytest.raises(msgspec.ValidationError, match=match):
            make_calibration_sweep(captures=(make_calibration_capture(burst=2),))


class TestSignalTrigger:
    @pytest.mark.parametrize('name', TRIGGERS, ids=list(TRIGGERS))
    def test_registered_names_accepted(self, name):
        assert SoapySource(signal_trigger=name, **MCR).signal_trigger == name

    def test_unregistered_name_rejected(self):
        name = 'cellular_5g_nope'
        assert name not in sa.registry.signal_trigger
        raises_on_both_paths(
            SoapySource, ValueError, TRIGGER_MSG, signal_trigger=name, **MCR
        )

    def test_trigger_bundle_accepted(self, construct):
        bundle = sa.specs.BuiltinTriggers(
            cellular_5g_pss_sync=sa.specs.Cellular5GNPSSSync(subcarrier_spacing=30e3)
        )
        source = construct(SoapySource, signal_trigger=bundle, **MCR)
        assert source.signal_trigger == bundle

    @pytest.mark.parametrize('count', [0, 2], ids=['none_enabled', 'two_enabled'])
    def test_trigger_bundle_must_enable_exactly_one(self, count):
        kws = {}
        if count == 2:
            kws = {
                'cellular_5g_pss_sync': sa.specs.Cellular5GNPSSSync(
                    subcarrier_spacing=30e3
                ),
                'cellular_5g_sss_sync': sa.specs.Cellular5GNSSSSync(
                    subcarrier_spacing=30e3
                ),
            }
        bundle = sa.specs.BuiltinTriggers(**kws)
        raises_on_both_paths(
            SoapySource, ValueError, 'exactly one trigger', signal_trigger=bundle, **MCR
        )


# %% Loops


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


# %% Sink and ZarrSink

Sink = ss.specs.Sink
ZarrSink = ss.specs.ZarrSink


def test_sweep_sink_is_a_zarr_sink():
    assert type(make_sweep().sink) is ZarrSink

    # direct construction keeps the base Sink until revalidation widens it
    direct = make_sweep(sink=Sink(path='x'))
    assert direct.sink == Sink(path='x')
    cls = type(direct)
    revalidated = (
        direct.validate(),
        direct.replace(loops=()),
        cls.from_dict(direct.to_dict(unfreeze=True)),
    )
    for spec in revalidated:
        assert spec.sink == ZarrSink(path='x')


def test_calibration_spec_keeps_the_base_sink():
    assert type(make_calibration_sweep().sink) is Sink

    widened = make_calibration_sweep(sink=ZarrSink())
    with pytest.raises(msgspec.ValidationError, match='store'):
        widened.validate()
    with pytest.raises(msgspec.ValidationError, match='store'):
        widened.replace(loops=())


# %% CaptureRemap


class TestCaptureRemap:
    """decoding of multi-field lookup keys"""

    def test_multi_key_tuple_keys_pass_through(self):
        remap = Remap(key=('a', 'b'), lookup={(1, 2): 5})
        assert remap.lookup == {(1, 2): 5}

    @given(keys=st.lists(key_pairs, unique=True, max_size=6))
    def test_multi_key_roundtrip(self, construct, keys):
        lookup = {json.dumps(list(k)): i for i, k in enumerate(keys)}
        remap = construct(Remap, key=('a', 'b'), lookup=lookup)
        assert isinstance(remap.lookup, frozendict)
        assert remap.lookup == {k: i for i, k in enumerate(keys)}
        hash(remap)

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


# %% ActionSpec, CapturePlan, Sweep: the spec chain


class TestSpecChain:
    @pytest.mark.parametrize(
        'kws, match',
        [
            ({'loops': (Repeat(count=2), Repeat(count=3))}, REPEAT_MSG),
            (
                {
                    'loops': (
                        List(field='snr', values=(0.0,)),
                        List(field='snr', values=(1.0,)),
                    )
                },
                f'{DUPLICATE_MSG} `snr`',
            ),
            (
                {
                    'captures': (
                        make_capture(frequency_offset=-1e6),
                        make_capture(frequency_offset=1e6),
                    ),
                    'loops': (List(field='frequency_offset', values=(0.0,)),),
                },
                COLLISION_MSG,
            ),
        ],
        ids=['repeat_not_first', 'duplicate_field', 'capture_collision'],
    )
    def test_capture_plan_validates_its_loops(self, kws, match):
        raises_on_both_paths(
            PlanCls, msgspec.ValidationError, match, **make_sweep_kws(**kws)
        )


# %% Sweep


class _ConflictingCapture(ss.specs.SingleToneCapture, frozen=True, kw_only=True):
    spectrogram: int = 0


SNR_LOOP = List(field='snr', values=(0.0, 1.0))
OFFSET_LOOP = List(field='frequency_offset', values=(0.0, 1e3))
WINDOW_LOOP = List(field='window', isin='analysis', values=('hann',))


class TestSweepLoops:
    @pytest.mark.parametrize(
        'loops',
        [
            (SNR_LOOP, Repeat(count=2)),
            (SNR_LOOP, Repeat(count=2), WINDOW_LOOP),
            (SNR_LOOP, OFFSET_LOOP, WINDOW_LOOP, Repeat(count=2)),
        ],
        ids=['second_of_two', 'between_loops', 'last_of_four'],
    )
    def test_repeat_must_be_first(self, loops):
        kws = make_sweep_kws(captures=(make_capture(),), loops=loops)
        raises_on_both_paths(SweepCls, msgspec.ValidationError, REPEAT_MSG, **kws)

    @pytest.mark.parametrize(
        'loops, first, second',
        [
            ((SNR_LOOP, List(field='snr', values=(2.0, 3.0))), 0, 1),
            ((SNR_LOOP, WINDOW_LOOP, List(field='snr', values=(2.0, 3.0))), 0, 2),
            ((Repeat(count=1), SNR_LOOP, List(field='snr', values=(2.0,))), 1, 2),
            (
                (OFFSET_LOOP, SNR_LOOP, List(field='frequency_offset', values=(5.0,))),
                0,
                2,
            ),
        ],
        ids=['adjacent', 'after_an_analysis_loop', 'after_a_repeat', 'other_field'],
    )
    def test_duplicate_loop_field_is_rejected_at_the_second_occurrence(
        self, loops, first, second
    ):
        field = loops[second].field
        match = (
            re.escape(f'$.loops[{second}]: {DUPLICATE_MSG} `{field}`')
            + '.*'
            + re.escape(f'at $.loops[{first}]')
        )
        kws = make_sweep_kws(loops=loops)
        raises_on_both_paths(SweepCls, msgspec.ValidationError, match, **kws)

    @pytest.mark.parametrize('lead_repeat', [False, True])
    def test_loop_over_a_field_that_captures_disagree_on_rejected(self, lead_repeat):
        captures = (
            make_capture(frequency_offset=-1e6),
            make_capture(frequency_offset=1e6),
        )
        loops = [List(field='frequency_offset', values=(0.0, 5e5, 1e6))]
        if lead_repeat:
            loops.insert(0, Repeat(count=2))
        index = 1 if lead_repeat else 0
        match = (
            re.escape(f'$.loops[{index}]:')
            + f'.*{COLLISION_MSG}.*'
            + re.escape('$.captures[0] on $.captures[1]')
        )
        kws = make_sweep_kws(captures=captures, loops=tuple(loops))
        raises_on_both_paths(SweepCls, msgspec.ValidationError, match, **kws)

    @pytest.mark.parametrize(
        'captures, loops',
        [
            (
                (make_capture(frequency_offset=-1e6),),
                (List(field='frequency_offset', values=(0.0, 1e6)),),
            ),
            (
                (
                    make_capture(frequency_offset=-1e6, snr=3.0),
                    make_capture(frequency_offset=-1e6, snr=10.0),
                ),
                (List(field='frequency_offset', values=(0.0, 1e6)),),
            ),
            (
                (make_capture(snr=3.0), make_capture(snr=10.0)),
                (List(field='snr', isin='analysis', values=(0.0, 1.0)),),
            ),
        ],
        ids=['single_capture_sets_the_field', 'captures_agree', 'analysis_loop'],
    )
    def test_non_colliding_loops_accepted(self, captures, loops):
        sweep = SweepCls(**make_sweep_kws(captures=captures, loops=loops))
        assert sweep.captures == captures


class TestSweepCaptures:
    def test_capture_field_conflicting_with_measurement_name(self):
        capture = _ConflictingCapture(port=0, sample_rate=1e6, duration=1e-3)
        match = re.escape('$.analysis:') + '.*`spectrogram`.*shadows'
        with pytest.raises(msgspec.ValidationError, match=match):
            make_sweep(captures=(capture,))


# %% Sweep: (capture, analysis) validation

# (capture, loops, message tail): each row breaks SPG's resolution rule at a different
# origin, and the error names the compared values and then that origin
INVALID_RESOLUTION_ORIGINS = {
    'looped_capture': (
        make_capture(),
        (List(field='sample_rate', values=(1e6, 1.005e6)),),
        (
            '(sample_rate: 1005000.0, frequency_resolution: 10000.0) '
            "- at $.loops: {'sample_rate': 1005000.0} on $.captures[0]"
        ),
    ),
    'adjust_analysis': (
        make_capture(adjust_analysis={'frequency_resolution': 3e4}),
        (),
        '(sample_rate: 1000000.0, frequency_resolution: 30000.0) - at $.captures[0]',
    ),
    'analysis_loop': (
        make_capture(),
        (List(field='frequency_resolution', isin='analysis', values=(1e4, 3e4)),),
        (
            '(sample_rate: 1000000.0, frequency_resolution: 30000.0) '
            "- at $.loops: {'frequency_resolution': 30000.0} on $.captures[0]"
        ),
    ),
}


class TestSweepAnalysisValidation:
    @pytest.mark.parametrize(
        'origin', list(INVALID_RESOLUTION_ORIGINS), ids=list(INVALID_RESOLUTION_ORIGINS)
    )
    def test_an_invalid_combination_names_its_origin(self, origin):
        capture, loops, location = INVALID_RESOLUTION_ORIGINS[origin]
        kws = make_sweep_kws(captures=(capture,), loops=loops, analysis=SPG)
        match = re.escape(f'$.analysis.spectrogram: {RESOLUTION_MSG} {location}')
        raises_on_both_paths(SweepCls, msgspec.ValidationError, match, **kws)


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
            (
                {'zz': {'snr': 1.0}},
                re.escape("$.adjust_captures['zz']"),
            ),
            (
                {'defaults': {'duration': 1.0}},
                re.escape('Object contains reserved capture field `duration`'),
            ),
            (
                {'defaults': {'port': 3}},
                re.escape('Object contains reserved capture field `port`'),
            ),
            (
                {'defaults': ['snr', 1.0]},
                # msgspec names the path after the message, SpecValidationError before
                r'(?s)Expected `object`.*adjust_captures\[|adjust_captures\[.*Expected `object`',
            ),
        ],
        ids=[
            'unknown source id',
            'reserved field',
            'reserved port',
            'per-source list not mapping',
        ],
    )
    def test_invalid_adjustments_rejected(self, adjust, match):
        kws = make_sweep_kws(adjust_captures=adjust)
        raises_on_both_paths(SweepCls, msgspec.ValidationError, match, **kws)


# %% ManualYFactorCalibration


class TestManualYFactorCalibration:
    def test_multiple_captures_need_implied_loops(self):
        captures = (make_calibration_capture(), make_calibration_capture(gain=-10))
        kws = make_calibration_sweep_kws(captures=captures)
        raises_on_both_paths(CalSweepCls, msgspec.ValidationError, IMPLIED_MSG, **kws)

        peripheral = ss.specs.ManualYFactorPeripheral(
            enr=10, ambient_temperature=290, implied_loops=('gain',)
        )
        kws = make_calibration_sweep_kws(captures=captures, calibration=peripheral)
        for sweep in construct_both(CalSweepCls, **kws):
            assert len(sweep.captures) == 2

    def test_source_calibration_must_be_none(self):
        kws = make_calibration_sweep_kws(source=CalSourceCls(calibration='cal.nc'))
        raises_on_both_paths(
            CalSweepCls, msgspec.ValidationError, SOURCE_CAL_MSG, **kws
        )

    def test_legacy_analysis_block_is_ignored_with_a_warning(self):
        """a calibration file with an `analysis` block decodes with a
        `DeprecationWarning`, and the block is discarded. The runner generates the
        analysis."""
        tree = make_calibration_sweep().to_dict(unfreeze=True)
        tree['analysis'] = {
            'channel_power_time_series': {'detector_period': 0.01},
        }
        with pytest.warns(DeprecationWarning, match='analysis'):
            sweep = CalSweepCls.from_dict(tree)
        assert sweep.analysis is None, 'discarded, so the spec stays hashable'
        assert hash(sweep) == hash(CalSweepCls.from_dict(sweep.to_dict(unfreeze=True)))
        assert sweep.calibration.enr == pytest.approx(YFACTOR_ENR_DB)


# %% CellChannel and Cellular5GNRSweep

CellChannel = ss.specs.CellChannel
CellularSweepCls = ss.bindings.air7101b_cellular.sensor.sweep_spec_cls
CELL_FREQ = 3750e6


def make_cell_capture(**kws):
    kws = {'center_frequency': CELL_FREQ, 'port': (0, 1), 'gain': -10, **kws}
    return ss.specs.SoapyCapture(duration=0.1, **kws)


def air_cellular_sweep_kws(**kws):
    return {
        'source': ss.bindings.air7101b.schema.source(),
        'captures': (make_cell_capture(),),
        'cell_channels': {'a': make_cell_channel()},
        **kws,
    }


class TestCellChannel:
    def test_only_30khz_spacing_is_accepted(self):
        raises_on_both_paths(
            CellChannel,
            msgspec.ValidationError,
            r'subcarrier_spacing',
            center_frequency=CELL_FREQ,
            channel_bandwidth=100e6,
            subcarrier_spacing=15e3,
        )

    @pytest.mark.parametrize('count', [1, 20], ids=['every_slot', 'frame'])
    def test_tdd_frame_slots_is_one_slot_or_a_frame(self, count):
        """3GPP TS 38.211 Table 4.3.2-1: 20 slots per 10 ms frame at 30 kHz"""
        channel = make_cell_channel(tdd_frame_slots='d' * count)
        assert len(channel.tdd_frame_slots) == count

    def test_tdd_frame_slots_rejects_a_partial_frame(self):
        raises_on_both_paths(
            CellChannel,
            msgspec.ValidationError,
            r'tdd_frame_slots.*1 or 20',
            center_frequency=CELL_FREQ,
            channel_bandwidth=100e6,
            tdd_frame_slots='d' * 19,
        )


class TestCellular5GNRSweep:
    def test_channels_must_have_distinct_center_frequencies(self):
        channels = {
            'a': make_cell_channel(),
            'b': make_cell_channel(channel_bandwidth=20e6),
        }
        raises_on_both_paths(
            CellularSweepCls,
            msgspec.ValidationError,
            r"cell_channels\['b'\]\.center_frequency.*'a' and 'b'",
            **air_cellular_sweep_kws(cell_channels=channels),
        )

    def test_capture_without_a_channel_is_rejected(self):
        captures = (make_cell_capture(center_frequency=3800e6),)
        raises_on_both_paths(
            CellularSweepCls,
            msgspec.ValidationError,
            r'captures\[0\]\.center_frequency',
            **air_cellular_sweep_kws(captures=captures),
        )


# %% ZarrSink


def test_zarr_sink_batched_write_count_is_deprecated():
    """a non-default count warns; the default is silent and the value is kept"""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ss.specs.ZarrSink(path='x.zarr')
    with pytest.warns(DeprecationWarning, match='batched_write_count'):
        sink = ss.specs.ZarrSink(path='x.zarr', batched_write_count=24)
    assert sink.batched_write_count == 24
