"""striqt.sensor.lib.compute.gpu: whether a sweep needs the GPU warmup"""

from __future__ import annotations

import pytest
from soapy_factories import soapy_capture
from sweep_strategies import SOURCE
from synthetic_sources import (
    ANALYSIS,
    FILTER_ONLY,
    IQ_ONLY,
    SCALE_ONLY,
    preset_capture,
    preset_sweep,
)

import striqt.sensor as ss
from striqt.sensor.lib.compute import gpu

CUPY = SOURCE.replace(array_backend='cupy')
AIR = ss.bindings.air7101b


def cupy_sweep(captures, *, analysis=IQ_ONLY, loops=()):
    return preset_sweep(
        'single_tone', captures, analysis=analysis, loops=loops, source=CUPY
    )


def scale_only(**kws):
    return preset_capture('single_tone', **{**SCALE_ONLY, **kws})


def air_sweep(**source_kws):
    capture = soapy_capture()
    return AIR.sensor.sweep_spec_cls(
        source=AIR.schema.source(**source_kws), captures=(capture,), analysis=IQ_ONLY
    )


# %% sweep_touches_gpu

TOUCHES_GPU = {
    'numpy': (preset_sweep('single_tone', (scale_only(),), analysis=ANALYSIS), False),
    'cupy_measurements': (cupy_sweep((scale_only(),), analysis=ANALYSIS), True),
    'cupy_calibration': (air_sweep(calibration='cal.nc'), True),
    'cupy_iq_scale_only': (cupy_sweep((scale_only(),)), False),
    'cupy_iq_host_resample': (cupy_sweep((scale_only(host_resample=True),)), True),
    'cupy_iq_filter_only': (
        cupy_sweep((preset_capture('single_tone', **FILTER_ONLY),)),
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
    """`iq_waveform` alone is a passthrough. Only a calibration, another measurement,
    host resampling, or a finite analysis bandwidth, in a capture or in a loop over
    those fields, puts work on the GPU."""
    sweep, expected = TOUCHES_GPU[case]
    assert gpu.sweep_touches_gpu(sweep) is expected
