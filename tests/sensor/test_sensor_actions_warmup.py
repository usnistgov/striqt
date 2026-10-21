"""striqt.sensor.lib.actions.warmup: the warmup sweep derived from a sweep, the
resources it runs over, and the runner method that decides whether a sweep warms up
"""

from __future__ import annotations

import pytest
from sweep_strategies import SOURCE, as_sweep, make_calibration_sweep
from synthetic_sources import (
    ANALYSIS,
    IQ_ONLY,
    SCALE_ONLY,
    preset_capture,
    preset_sweep,
)

import striqt.sensor as ss
from striqt.sensor.lib.actions import SweepRunner, registry, warmup

CUPY = SOURCE.replace(array_backend='cupy')
WARMUP_ENABLED = ss.specs.SweepOptions(skip_warmup=False)


def cupy_sweep(captures, *, analysis=IQ_ONLY, loops=(), **kws):
    return preset_sweep(
        'single_tone', captures, analysis=analysis, loops=loops, source=CUPY, **kws
    )


def scale_only(**kws):
    return preset_capture('single_tone', **{**SCALE_ONLY, **kws})


# %% derive_warmup_sweep

PORT_LOOP = ss.specs.List(field='port', values=(0, 1))

WARMUP_PORTS = {
    'port_0': ((scale_only(port=0),), (), 1),
    'ports_0_1': ((scale_only(port=(0, 1)),), (), 2),
    'port_1_only': ((scale_only(port=1),), (), 2),
    'port_loop': ((scale_only(port=0),), (PORT_LOOP,), 2),
}


@pytest.mark.parametrize('case', list(WARMUP_PORTS), ids=list(WARMUP_PORTS))
def test_warmup_source_has_enough_rx_ports(case):
    """port numbers index the receiver's ports from 0, so the warmup source needs
    max(port) + 1 of them for every port a capture or a port loop names"""
    captures, loops, num_rx_ports = WARMUP_PORTS[case]
    warm = warmup.derive_warmup_sweep(cupy_sweep(captures, loops=loops))
    assert isinstance(warm.source, ss.specs.NoSource)
    assert warm.source.num_rx_ports == num_rx_ports


@pytest.mark.parametrize('count', [1, 2], ids=['one', 'two'])
def test_warmup_sweep_flattens_to_count_captures(count):
    captures = tuple(scale_only(frequency_offset=f) for f in (1e6, 2e6, 3e6))
    sweep = cupy_sweep(captures, loops=(ss.specs.Repeat(count=2), PORT_LOOP))
    warm = warmup.derive_warmup_sweep(sweep, count=count)
    assert warm.loops == ()
    assert len(warm.captures) == count
    assert type(warm.captures[0]) is type(captures[0])


def test_warmup_sweep_preserves_analysis_and_trigger_and_nulls_the_sink_path():
    """the analysis and the source's trigger are copied; the sink is copied with
    its path set to None"""
    source = CUPY.replace(signal_trigger='cellular_5g_pss_sync')
    sink = ss.specs.ZarrSink(path='somewhere.zarr', batched_write_count=3)
    analysis = ss.specs.BundledAnalysis.from_dict({
        **ANALYSIS.to_dict(),
        # shared_spectrum picks SSB case C, which a capture with no center_frequency
        # cannot otherwise resolve at 30 kHz
        'cellular_5g_pss_sync': {
            'subcarrier_spacing': 30e3,
            'sample_rate': 7.68e6,
            'discovery_periodicity': 0.02,
            'shared_spectrum': True,
        },
    })
    # the correlator needs whole 10 ms frames
    capture = scale_only(duration=10e-3)
    sweep = preset_sweep(
        'single_tone', (capture,), analysis=analysis, source=source, sink=sink
    )
    warm = warmup.derive_warmup_sweep(sweep)
    assert warm.analysis == analysis
    assert warm.sink == sink.replace(path=None)
    assert warm.source.signal_trigger == 'cellular_5g_pss_sync'
    assert warm.source.master_clock_rate == SOURCE.master_clock_rate
    assert warm.sensor.source_cls is ss.sources.NoSource


def test_warmup_sweep_is_a_sweep_with_the_origin_capture_type():
    """the warmup binding takes its capture type from the sweep's binding, so the
    sweep's captures validate, and runs as a plain sweep"""
    sweep = cupy_sweep((scale_only(),))
    warm = warmup.derive_warmup_sweep(sweep)
    assert isinstance(warm, ss.specs.Sweep)
    assert type(warm).schema.capture is type(sweep).schema.capture
    assert type(warm).sensor.source_cls is ss.sources.NoSource
    assert registry.lookup(warm) is SweepRunner


def test_one_warmup_binding_per_origin():
    a = warmup.warmup_binding(ss.bindings.single_tone)
    assert a is warmup.warmup_binding(ss.bindings.single_tone)
    assert a is not warmup.warmup_binding(ss.bindings.noise)
    assert a.sensor.sweep_spec_cls.__name__ not in ss.lib.bindings.registry


# %% warmup_sweep_for

NO_WARMUP = {
    'skip_option': preset_sweep('single_tone', (scale_only(),), analysis=ANALYSIS),
    'numpy_backend': preset_sweep(
        'single_tone', (scale_only(),), analysis=ANALYSIS, options=WARMUP_ENABLED
    ),
    'iq_passthrough': cupy_sweep((scale_only(),), options=WARMUP_ENABLED),
    'calibration': make_calibration_sweep(),
}


@pytest.mark.parametrize('case', list(NO_WARMUP), ids=list(NO_WARMUP))
def test_warmup_sweep_is_none_when_nothing_needs_the_gpu(case):
    """a sweep that skips the warmup, runs on numpy, only passes IQ through, or is the
    sweep a calibration builds (whose options skip the warmup) has no warmup"""
    spec = NO_WARMUP[case]
    assert warmup.warmup_sweep_for(as_sweep(spec)) is None


def test_warmup_sweep_derives_from_a_gpu_sweep():
    spec = cupy_sweep((scale_only(),), analysis=ANALYSIS, options=WARMUP_ENABLED)
    warm = warmup.warmup_sweep_for(spec)
    assert isinstance(warm, ss.specs.Sweep)
    assert warm.analysis == spec.analysis


def test_a_warmup_has_no_warmup():
    warm = warmup.derive_warmup_sweep(
        cupy_sweep((scale_only(),), analysis=ANALYSIS, options=WARMUP_ENABLED)
    )
    assert warmup.warmup_sweep_for(warm) is None


# %% open_warmup_resources


def test_open_binds_the_warmup_implementation_classes(isolated_lookup):
    warm = warmup.derive_warmup_sweep(preset_sweep('single_tone', (scale_only(),)))
    runner = SweepRunner(warm)
    res = warmup.open_warmup_resources(runner)
    with res['source']:
        assert type(res['source']) is ss.bindings.warmup
        assert isinstance(res['source'].backend, ss.sources.NoSource)
        assert isinstance(res['peripherals'], ss.peripherals.NoPeripherals)
        assert isinstance(res['sink'], ss.sinks.NoSink)
        assert res['sweep_spec'] is warm
        assert res['action_spec'] is warm
        assert res['runner'] is runner
        assert res['calibration'] is None
