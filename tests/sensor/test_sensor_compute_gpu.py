"""striqt.sensor.lib.compute.gpu: whether a sweep needs the GPU warmup, and the
warmup sweep derived from it"""

from __future__ import annotations

import pytest
from sweep_strategies import SOURCE
from synthetic_sources import (
    ANALYSIS,
    FILTER_ONLY,
    SCALE_ONLY,
    make_capture,
    make_sweep,
)

import striqt.sensor as ss
from striqt.sensor.lib.compute import gpu

IQ_ONLY = ss.specs.BundledAnalysis.from_dict({'iq_waveform': {}})
CUPY = SOURCE.replace(array_backend='cupy')
AIR = ss.bindings.air7101b


def cupy_sweep(captures, *, analysis=IQ_ONLY, loops=()):
    return make_sweep(
        'single_tone', captures, analysis=analysis, loops=loops, source=CUPY
    )


def scale_only(**kws):
    return make_capture('single_tone', **{**SCALE_ONLY, **kws})


def air_sweep(**source_kws):
    capture = ss.specs.SoapyCapture(
        port=0, center_frequency=1e9, gain=0.0, host_resample=False
    )
    return AIR.sensor.sweep_spec_cls(
        source=AIR.schema.source(**source_kws), captures=(capture,), analysis=IQ_ONLY
    )


# %% sweep_touches_gpu

TOUCHES_GPU = {
    'numpy': (make_sweep('single_tone', (scale_only(),), analysis=ANALYSIS), False),
    'cupy_measurements': (cupy_sweep((scale_only(),), analysis=ANALYSIS), True),
    'cupy_calibration': (air_sweep(calibration='cal.nc'), True),
    'cupy_iq_scale_only': (cupy_sweep((scale_only(),)), False),
    'cupy_iq_host_resample': (cupy_sweep((scale_only(host_resample=True),)), True),
    'cupy_iq_filter_only': (
        cupy_sweep((make_capture('single_tone', **FILTER_ONLY),)),
        True,
    ),
    'cupy_iq_loop_host_resample': (
        cupy_sweep(
            (scale_only(),),
            loops=(ss.specs.List(field='host_resample', values=(False, True)),),
        ),
        True,
    ),
    'cupy_iq_loop_finite_bandwidth': (
        cupy_sweep(
            (scale_only(),),
            loops=(
                ss.specs.List(field='analysis_bandwidth', values=(float('inf'), 5e6)),
            ),
        ),
        True,
    ),
    'cupy_iq_loop_infinite_bandwidth': (
        cupy_sweep(
            (scale_only(),),
            loops=(ss.specs.List(field='analysis_bandwidth', values=(float('inf'),)),),
        ),
        False,
    ),
}


@pytest.mark.parametrize('case', list(TOUCHES_GPU), ids=list(TOUCHES_GPU))
def test_sweep_touches_gpu(case):
    """iq_waveform alone is a passthrough, so only a calibration, another
    measurement, host resampling or a finite analysis bandwidth (in a capture or a
    loop over them) puts work on the GPU"""
    sweep, expected = TOUCHES_GPU[case]
    assert gpu.sweep_touches_gpu(sweep) is expected


# %% build_warmup_sweep

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
    warmup = gpu.build_warmup_sweep(cupy_sweep(captures, loops=loops))
    assert isinstance(warmup.source, ss.specs.NoSource)
    assert warmup.source.num_rx_ports == num_rx_ports


@pytest.mark.parametrize('count', [1, 2], ids=['one', 'two'])
def test_warmup_sweep_flattens_to_count_captures(count):
    captures = tuple(scale_only(frequency_offset=f) for f in (1e6, 2e6, 3e6))
    sweep = cupy_sweep(captures, loops=(ss.specs.Repeat(count=2), PORT_LOOP))
    warmup = gpu.build_warmup_sweep(sweep, count=count)
    assert warmup.loops == ()
    assert len(warmup.captures) == count
    assert type(warmup.captures[0]) is type(captures[0])


def test_warmup_sweep_preserves_analysis_sink_and_trigger():
    source = CUPY.replace(signal_trigger='cellular_5g_pss_sync')
    sink = ss.specs.Sink(path='somewhere.zarr', batched_write_count=3)
    sweep = make_sweep(
        'single_tone', (scale_only(),), analysis=ANALYSIS, source=source, sink=sink
    )
    warmup = gpu.build_warmup_sweep(sweep)
    assert warmup.analysis == ANALYSIS
    assert warmup.sink == sink
    assert warmup.source.signal_trigger == 'cellular_5g_pss_sync'
    assert warmup.source.master_clock_rate == SOURCE.master_clock_rate
    assert warmup.sensor.source_cls is ss.sources.NoSource
    assert warmup.sensor.sink_cls is ss.sinks.NoSink
