"""striqt.sensor.lib.actions.cellular: the channel table of a `Cellular5GNRSweep`
resolved into radio settings and analysis adjustments, the generated analysis, and
the runner that builds and runs the plain `Sweep`
"""

from __future__ import annotations

import copy
from fractions import Fraction
from typing import Optional

import msgspec
import pytest
from sweep_strategies import (
    CELL_CHANNEL,
    CELL_LABEL,
    SOURCE,
    CellularAnalyses,
    make_cell_channel,
    make_cellular_capture,
    make_cellular_runner,
    make_cellular_sweep,
    make_cellular_sweep_kws,
    restricted_cellular_binding,
)
from synthetic_sources import run_in_memory

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.analysis.specs.helpers import subset_of
from striqt.sensor.lib.actions import Cellular5GNRSweepRunner, cellular, registry

H = ss.specs.sequencing
SCS = 30e3
# TS 38.211 §4.3.2: 14 symbols per slot with a normal cyclic prefix, 2**mu slots per
# 1 ms subframe, so 280 symbols per 10 ms frame at 30 kHz (mu = 1)
SYMBOL_30K = Fraction(1, 28000)
FRAME = 10e-3
PSS_TRIGGER = 'cellular_5g_pss_sync'
GAIN_LOOP = ss.specs.List(field='gain', values=(0, -10))
FC_LOOP = ss.specs.List(field='center_frequency', values=(3750e6, 3900e6))

# every block enabled, with the free parameters of sweeps/site/site-cellular.yaml
BLOCKS = {
    'spectrogram': ss.specs.CellularSpectrogramSpec(
        window=('kaiser', 11.884), lo_bandstop=120e3
    ),
    'power_spectral_density': ss.specs.CellularPowerSpectralDensitySpec(
        window=('kaiser', 11.884), time_statistic=('mean', 0.5, 'max')
    ),
    'cellular_resource_power_histogram': ss.specs.CellularResourcePowerHistogramSpec(
        window=('kaiser', 11.884),
        power_low=-150,
        power_high=0,
        power_resolution=0.1,
        average_rbs=True,
    ),
    'cellular_5g_pss_correlation': ss.specs.CellularPSSCorrelationSpec(),
    'cellular_5g_pss_sync': ss.specs.CellularPSSSyncSpec(per_port=True, max_beams=4),
    'cellular_5g_ssb_spectrogram': ss.specs.CellularSSBSpectrogramSpec(
        lo_bandstop=120e3
    ),
    'cyclic_channel_power': ss.specs.CellularCyclicChannelPowerSpec(),
    'channel_power_time_series': ss.specs.CellularChannelPowerTimeSeriesSpec(),
    'channel_power_histogram': ss.specs.CellularChannelPowerHistogramSpec(
        power_low=-150, power_high=0, power_resolution=0.1
    ),
    'cellular_cyclic_autocorrelation': ss.specs.CellularCyclicAutocorrelationSpec(),
    'iq_waveform': ss.specs.CellularIQWaveformSpec(),
}

# the OFDM symbol grid: SCS/2 resolution with 13/28 overlap and 15/28 window fill
GRID = {
    'frequency_resolution': SCS / 2,
    'fractional_overlap': Fraction(13, 28),
    'window_fill': Fraction(15, 28),
}

# block field -> (measurement name, the parameters the spacing fills in). A resource
# block is 12 subcarriers (TS 38.211 §4.4.4.1)
MEASUREMENTS = {
    'spectrogram': ('spectrogram', {**GRID, 'integration_bandwidth': 12 * SCS}),
    'power_spectral_density': (
        'power_spectral_density',
        {**GRID, 'integration_bandwidth': SCS},
    ),
    'cellular_resource_power_histogram': (
        'cellular_resource_power_histogram',
        {'subcarrier_spacing': SCS},
    ),
    'cellular_5g_pss_correlation': (
        'cellular_5g_pss_correlation',
        {'subcarrier_spacing': SCS},
    ),
    'cellular_5g_pss_sync': (PSS_TRIGGER, {'subcarrier_spacing': SCS}),
    'cellular_5g_ssb_spectrogram': (
        'cellular_5g_ssb_spectrogram',
        {'subcarrier_spacing': SCS},
    ),
    'cyclic_channel_power': (
        'cyclic_channel_power',
        {'cyclic_period': FRAME, 'detector_period': SYMBOL_30K},
    ),
    'channel_power_time_series': (
        'channel_power_time_series',
        {'detector_period': SYMBOL_30K},
    ),
    'channel_power_histogram': (
        'channel_power_histogram',
        {'detector_period': SYMBOL_30K},
    ),
    'cellular_cyclic_autocorrelation': (
        'cellular_cyclic_autocorrelation',
        {'subcarrier_spacings': (SCS,)},
    ),
    'iq_waveform': ('iq_waveform', {}),
}


def enabled(analysis: ss.specs.BundledAnalysis) -> dict:
    """return the enabled measurement specs of `analysis` keyed by field name"""
    specs = {k: getattr(analysis, k) for k in analysis.__struct_fields__}
    return {k: v for k, v in specs.items() if v is not None}


def one_block_sweep(field: str, block=None, **kws):
    """return a cellular sweep over one 10 ms frame with only `field` enabled in the
    analysis"""
    block = BLOCKS[field] if block is None else block
    analysis = ss.specs.Cellular5GNRAnalysis(**{field: block})
    kws.setdefault('captures', (make_cellular_capture(duration=FRAME),))
    return make_cellular_sweep(analysis=analysis, **kws)


# %% symbol_period


@pytest.mark.parametrize(
    'scs, period',
    [(30e3, SYMBOL_30K), (15e3, Fraction(1, 14000))],
    ids=['30kHz', '15kHz'],
)
def test_symbol_period_is_the_frame_over_its_symbol_count(scs, period):
    assert cellular.symbol_period(scs) == period


# %% ssb_symbol_indexes


@pytest.mark.parametrize(
    'ssb_case, expected', [(None, 'c'), ('b', 'b')], ids=['default_30khz', 'explicit']
)
def test_ssb_symbol_indexes_default_to_the_case_of_the_spacing(ssb_case, expected):
    assert cellular.ssb_symbol_indexes(make_cell_channel(ssb_case=ssb_case)) == expected


# %% match_channel

OTHER_CHANNEL = make_cell_channel(center_frequency=3900e6)
CHANNELS = {'3900 MHz': OTHER_CHANNEL, CELL_LABEL: CELL_CHANNEL}


@pytest.mark.parametrize(
    'center_frequency', [3750e6, (3750e6, 3750e6)], ids=['scalar', 'per_port']
)
def test_match_channel_selects_by_center_frequency(center_frequency):
    port = 0 if isinstance(center_frequency, float) else (0, 1)
    capture = make_cellular_capture(port=port, center_frequency=center_frequency)
    assert cellular.match_channel(capture, CHANNELS, 0) is CELL_CHANNEL


def test_match_channel_rejects_a_frequency_without_a_channel():
    capture = make_cellular_capture(center_frequency=3800e6)
    with pytest.raises(
        msgspec.ValidationError, match=r'captures\[3\]\.center_frequency'
    ):
        cellular.match_channel(capture, CHANNELS, 3)


def test_match_channel_rejects_ports_at_different_frequencies():
    capture = make_cellular_capture(port=(0, 1), center_frequency=(3750e6, 3900e6))
    with pytest.raises(
        msgspec.ValidationError, match=r'captures\[0\]\.center_frequency'
    ):
        cellular.match_channel(capture, CHANNELS, 0)


# %% resolve_capture
#
# nr_sample_rate steps in multiples of 896 x 30 kHz = 26.88 MS/s above the channel
# bandwidth plus a 250 kHz transition band


@pytest.mark.parametrize(
    'channel_kws, sample_rate',
    [
        ({'channel_bandwidth': 100e6}, 107.52e6),
        ({'channel_bandwidth': 40e6}, 53.76e6),
        ({'channel_bandwidth': 60e6}, 80.64e6),
        ({'channel_bandwidth': 60e6, 'min_sample_rate': 107.52e6}, 107.52e6),
    ],
    ids=['100MHz', '40MHz', '60MHz', '60MHz_min_rate'],
)
def test_resolve_capture_derives_the_radio_settings(channel_kws, sample_rate):
    channel = make_cell_channel(**channel_kws)
    resolved = cellular.resolve_capture(make_cellular_capture(), channel, 0)
    assert (resolved.sample_rate, resolved.analysis_bandwidth) == (
        sample_rate,
        channel.channel_bandwidth,
    )


def test_resolve_capture_generates_the_analysis_adjustments():
    resolved = cellular.resolve_capture(make_cellular_capture(), CELL_CHANNEL, 0)
    assert dict(resolved.adjust_analysis) == {
        'frequency_offset': -40.08e6,
        'guard_bandwidths': (875e3, 575e3),
        'frame_slots': 'dddsuudddddddsuudddd',
        'special_symbols': 'ddddddfffffffu',
        'subcarrier_spacing': 30e3,
        'symbol_indexes': 'c',
    }


def test_resolve_capture_keeps_the_adjustments_of_the_capture():
    """a key set by the capture wins over the generated value, and an unrelated key
    of the capture is kept"""
    capture = make_cellular_capture(
        adjust_analysis={'max_beams': 2, 'frequency_offset': -1e6}
    )
    adjusted = dict(cellular.resolve_capture(capture, CELL_CHANNEL, 0).adjust_analysis)
    assert (
        adjusted['max_beams'],
        adjusted['frequency_offset'],
        adjusted['guard_bandwidths'],
    ) == (2, -1e6, (875e3, 575e3))


def test_resolve_capture_consumed_keys_limit_the_generated_adjustments():
    capture = make_cellular_capture(adjust_analysis={'max_beams': 2})
    resolved = cellular.resolve_capture(
        capture, CELL_CHANNEL, 0, consumed_keys={'frequency_offset', 'max_beams'}
    )
    assert dict(resolved.adjust_analysis) == {
        'frequency_offset': -40.08e6,
        'max_beams': 2,
    }


def test_resolve_capture_accepts_the_derived_sample_rate():
    capture = make_cellular_capture(sample_rate=107.52e6)
    resolved = cellular.resolve_capture(capture, CELL_CHANNEL, 0)
    assert (resolved.sample_rate, resolved.analysis_bandwidth) == (107.52e6, 100e6)


@pytest.mark.parametrize(
    'field, value', [('sample_rate', 53.76e6), ('analysis_bandwidth', 40e6)]
)
def test_resolve_capture_rejects_another_derived_value(field, value):
    capture = make_cellular_capture(**{field: value})
    with pytest.raises(msgspec.ValidationError) as info:
        cellular.resolve_capture(capture, CELL_CHANNEL, 1)
    assert f'captures[1].{field}' in str(info.value)
    assert 'min_sample_rate' in str(info.value)


# %% analysis_parameter_names


def test_analysis_parameter_names_are_the_fields_of_the_enabled_measurements():
    analysis = ss.specs.BundledAnalysis.from_dict({
        'cellular_5g_pss_correlation': {'subcarrier_spacing': SCS},
        'iq_waveform': {},
    })
    assert cellular.analysis_parameter_names(analysis) == {
        'subcarrier_spacing',
        'sample_rate',
        'discovery_periodicity',
        'frequency_offset',
        'shared_spectrum',
        'delay',
        'symbol_indexes',
        'max_lag_symbols',
        'max_block_count',
        'start_time_sec',
        'stop_time_sec',
    }


def test_analysis_parameter_names_of_no_measurements_is_empty():
    assert cellular.analysis_parameter_names(ss.specs.BundledAnalysis()) == frozenset()


# %% channel_subcarrier_spacing


def test_channel_subcarrier_spacing_is_the_shared_spacing():
    channels = {0: CELL_CHANNEL, 1: OTHER_CHANNEL}
    assert cellular.channel_subcarrier_spacing(channels) == pytest.approx(SCS)


def test_channel_subcarrier_spacing_rejects_mixed_spacings():
    """`CellChannel` accepts 30 kHz only. The 15 kHz channel is written onto a copy with
    `force_setattr`."""
    at_15k = copy.copy(OTHER_CHANNEL)
    msgspec.structs.force_setattr(at_15k, 'subcarrier_spacing', 15e3)
    with pytest.raises(msgspec.ValidationError, match='cell_channels'):
        cellular.channel_subcarrier_spacing({0: CELL_CHANNEL, 1: at_15k})


# %% cellular_analysis


@pytest.mark.parametrize('field', list(MEASUREMENTS), ids=list(MEASUREMENTS))
def test_each_block_enables_its_measurement_with_the_spacing_filled(field):
    """the free parameters of the block are copied, the spacing-dependent parameters
    are filled, and no other measurement is enabled"""
    name, derived = MEASUREMENTS[field]
    spec = one_block_sweep(field)
    analysis = enabled(cellular.cellular_analysis(spec, SCS))
    assert list(analysis) == [name]
    measurement = analysis[name]
    assert {k: getattr(measurement, k) for k in derived} == derived
    block = BLOCKS[field]
    free = [k for k in block.__struct_fields__ if k not in derived]
    assert [getattr(measurement, k) for k in free] == [getattr(block, k) for k in free]


@pytest.mark.parametrize(
    'detector_period, expected',
    [(None, SYMBOL_30K), (Fraction(1, 4000), Fraction(1, 4000))],
    ids=['default_symbol', 'explicit'],
)
def test_time_series_detector_period_defaults_to_one_symbol(detector_period, expected):
    block = ss.specs.CellularChannelPowerTimeSeriesSpec(detector_period=detector_period)
    spec = one_block_sweep('channel_power_time_series', block)
    analysis = cellular.cellular_analysis(spec, SCS)
    assert analysis.channel_power_time_series.detector_period == expected


def test_cellular_analysis_is_the_group_of_the_binding():
    """the generated analysis is an instance of the `schema.analysis` of the binding,
    including a group restricted to the cellular measurements"""
    cls = restricted_cellular_binding().sensor.sweep_spec_cls
    block = ss.specs.Cellular5GNRAnalysis(
        channel_power_time_series=BLOCKS['channel_power_time_series']
    )
    spec = cls(**make_cellular_sweep_kws(analysis=block))
    assert type(cellular.cellular_analysis(spec, SCS)) is CellularAnalyses
    assert type(cellular.build_cellular_sweep(spec).analysis) is CellularAnalyses


def test_pss_trigger_requires_the_pss_sync_block():
    source = SOURCE.replace(signal_trigger=PSS_TRIGGER)
    with pytest.raises(
        msgspec.ValidationError, match=r'analysis\.cellular_5g_pss_sync'
    ):
        one_block_sweep('iq_waveform', source=source)


def test_pss_trigger_runs_with_the_pss_sync_block():
    source = SOURCE.replace(signal_trigger=PSS_TRIGGER)
    spec = one_block_sweep('cellular_5g_pss_sync', source=source)
    assert PSS_TRIGGER in enabled(cellular.cellular_analysis(spec, SCS))


def test_struct_trigger_needs_no_pss_sync_block():
    """the struct form of `signal_trigger` includes the trigger spec. The derived sweep
    triggers without the `cellular_5g_pss_sync` measurement in the analysis."""
    bundle = ss.specs.BuiltinTriggers(
        cellular_5g_pss_sync=sa.specs.Cellular5GNPSSSync(subcarrier_spacing=SCS)
    )
    spec = one_block_sweep('iq_waveform', source=SOURCE.replace(signal_trigger=bundle))
    sweep = cellular.build_cellular_sweep(spec)
    assert PSS_TRIGGER not in enabled(sweep.analysis)
    trigger = ss.lib.compute.get_trigger_from_spec(sweep.source, sweep.analysis)
    assert trigger is not None and trigger.info.name == PSS_TRIGGER


def test_other_string_trigger_is_rejected():
    source = SOURCE.replace(signal_trigger='cellular_5g_sss_sync')
    with pytest.raises(msgspec.ValidationError, match=r'source\.signal_trigger'):
        one_block_sweep('iq_waveform', source=source)


# %% check_block_fields


class _ToyMeasurement(sa.specs.Analysis, frozen=True, kw_only=True):
    window: str = 'hann'
    frequency_resolution: float = 1.0
    frame_slots: Optional[str] = None
    detector_period: float = 1.0
    generation: str = '5G'


@subset_of(
    _ToyMeasurement,
    without=('frequency_resolution', 'frame_slots', 'generation'),
    optional=('detector_period',),
)
class _ToyBlock(ss.specs.SpecBase, frozen=True, kw_only=True):
    window: str = 'hann'
    detector_period: Optional[float] = None


def _classify(monkeypatch, **tables):
    """run `check_block_fields` on the toy block with the `FIXED` and `DEFAULTED`
    tables of the module replaced by `tables`"""
    for k in ('FIXED', 'DEFAULTED'):
        monkeypatch.setattr(cellular, k, {'toy': tables.get(k, {})})
    cellular.check_block_fields(_ToyBlock, _ToyMeasurement, 'toy')


def test_check_block_fields_accepts_an_agreeing_classification(monkeypatch):
    """the check passes with `frequency_resolution` fixed, `frame_slots` per capture,
    `detector_period` defaulted, and `generation` omitted without a table entry"""
    _classify(
        monkeypatch,
        FIXED={'frequency_resolution': lambda scs: scs},
        DEFAULTED={'detector_period': lambda scs: scs},
    )


def test_check_block_fields_reports_a_fixed_parameter_the_block_exposes(monkeypatch):
    with pytest.raises(TypeError, match=r"fixed but exposed \['window'\]"):
        _classify(
            monkeypatch,
            FIXED={'window': lambda scs: 'x'},
            DEFAULTED={'detector_period': lambda scs: scs},
        )


def test_check_block_fields_reports_defaulted_optional_disagreement(monkeypatch):
    with pytest.raises(TypeError, match=r'defaulted \[\] differs from optional'):
        _classify(monkeypatch, FIXED={'frequency_resolution': lambda scs: scs})


def test_check_block_fields_rejects_an_undeclared_block():
    class Plain(ss.specs.SpecBase, frozen=True, kw_only=True):
        window: str = 'hann'

    with pytest.raises(TypeError, match='not declared with subset_of'):
        cellular.check_block_fields(Plain, _ToyMeasurement, 'toy')


def test_check_block_fields_rejects_a_block_of_another_measurement():
    with pytest.raises(TypeError, match=r'subset of _ToyMeasurement, not of'):
        cellular.check_block_fields(_ToyBlock, sa.specs.IQWaveform, 'iq_waveform')


def test_every_cellular_block_is_classified():
    """each block is a declared subset of the measurement spec of the same name, and
    the `FIXED`, `PER_CAPTURE`, and `DEFAULTED` tables agree with that declaration.
    Every field of `Cellular5GNRAnalysis` is a field of `BuiltinAnalyses`."""
    cellular.check_all_blocks()
    assert set(ss.specs.Cellular5GNRAnalysis.__struct_fields__) <= set(
        ss.specs.BuiltinAnalyses.__struct_fields__
    )


# %% build_cellular_sweep and validate


def test_build_sweep_is_cached_on_the_spec():
    runner = make_cellular_runner()
    assert cellular.build_cellular_sweep(runner.spec) is runner.sweep


def test_build_sweep_is_the_bound_sweep_class():
    runner = make_cellular_runner()
    assert type(runner.sweep) is type(runner.spec).sweep_cls
    assert isinstance(runner.sweep, ss.specs.Sweep)
    assert not isinstance(runner.sweep, ss.specs.Cellular5GNRSweep)


def test_build_sweep_copies_the_document_fields():
    adjust = {'defaults': {'antenna_name': 'Omni'}}
    plot_hint = ss.specs.PlotOptions.from_dict({
        'data': {'sweep_index': 0},
        'plotter': {},
    })
    runner = make_cellular_runner(
        description=ss.specs.Description(summary='cellular'),
        extensions=ss.specs.Extension(import_name='extensions'),
        sink=ss.specs.ZarrSink(path='out.zarr', log_path='log.json'),
        adjust_captures=adjust,
        plot_hint=plot_hint,
        options=ss.specs.SweepOptions(reuse_iq=True),
        loops=(GAIN_LOOP,),
    )
    doc, sweep = runner.spec, runner.sweep
    copied = (
        'description',
        'source',
        'peripherals',
        'extensions',
        'sink',
        'loops',
        'adjust_captures',
        'plot_hint',
        'options',
    )
    assert {f: getattr(sweep, f) for f in copied} == {
        f: getattr(doc, f) for f in copied
    }
    assert sweep.analysis == cellular.cellular_analysis(doc, SCS)


def test_build_sweep_resolves_every_capture():
    captures = (make_cellular_capture(), make_cellular_capture(port=1))
    runner = make_cellular_runner(captures=captures)
    assert [c.port for c in runner.sweep.captures] == [0, 1]
    assert {(c.sample_rate, c.analysis_bandwidth) for c in runner.sweep.captures} == {
        (107.52e6, 100e6)
    }


ALL_ADJUSTMENTS = {
    'frequency_offset',
    'guard_bandwidths',
    'frame_slots',
    'special_symbols',
    'subcarrier_spacing',
    'symbol_indexes',
}


@pytest.mark.parametrize(
    'fields, keys',
    [
        (tuple(BLOCKS), ALL_ADJUSTMENTS),
        (
            ('cellular_5g_pss_correlation',),
            {'frequency_offset', 'subcarrier_spacing', 'symbol_indexes'},
        ),
        (('iq_waveform',), set()),
    ],
    ids=['every_block', 'cellular_5g_pss_correlation', 'iq_waveform'],
)
def test_build_sweep_adjusts_only_the_parameters_a_measurement_reads(fields, keys):
    """the PSS correlator reads the SSB offset, spacing, and symbol indexes but no
    TDD pattern or guard bandwidth. The IQ waveform reads nothing of the channel."""
    analysis = ss.specs.Cellular5GNRAnalysis(**{f: BLOCKS[f] for f in fields})
    runner = make_cellular_runner(
        captures=(make_cellular_capture(duration=FRAME),), analysis=analysis
    )
    assert set(runner.sweep.captures[0].adjust_analysis) == keys


def test_center_frequency_loop_is_rejected_at_construction():
    with pytest.raises(msgspec.ValidationError, match=r'loops\[0\]\.field'):
        make_cellular_sweep(loops=(FC_LOOP,))


# %% runner protocol


def test_registry_runs_the_cellular_sweep_with_its_runner():
    spec = make_cellular_sweep()
    assert registry.lookup(spec) is Cellular5GNRSweepRunner
    assert Cellular5GNRSweepRunner(spec).sweep is cellular.build_cellular_sweep(spec)


def test_in_memory_run_yields_one_dataset_per_capture_at_the_derived_rate():
    """a 20 MHz channel derives 26.88 MS/s, and 1 ms of the time series at 30 kHz
    spacing is 28 symbols"""
    channel = make_cell_channel(channel_bandwidth=20e6)
    captures = (make_cellular_capture(), make_cellular_capture(gain=0))
    spec = make_cellular_sweep(
        captures=captures,
        cell_channels={CELL_LABEL: channel},
        analysis=ss.specs.Cellular5GNRAnalysis(
            channel_power_time_series=BLOCKS['channel_power_time_series']
        ),
    )
    datasets = run_in_memory(spec)
    assert [ds.gain.item() for ds in datasets] == [-10, 0]
    for ds in datasets:
        assert (ds.sample_rate.item(), ds.analysis_bandwidth.item()) == (26.88e6, 20e6)
        assert ds.channel_power_time_series.sizes['time_elapsed'] == 28
        assert ds.attrs['cell_channels'] == {CELL_LABEL: channel.to_dict()}
