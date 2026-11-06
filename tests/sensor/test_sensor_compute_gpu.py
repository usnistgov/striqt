"""striqt.sensor.lib.compute.gpu: whether a sweep needs the GPU warmup"""

from __future__ import annotations

import pytest
from sweep_strategies import air_sweep
from synthetic_sources import (
    ANALYSIS,
    FILTER_ONLY,
    IQ_ONLY,
    cupy_sweep,
    preset_capture,
    preset_sweep,
    scale_only,
)

import striqt.sensor as ss
from striqt.sensor.lib.compute import gpu

# %% sweep_touches_gpu

TOUCHES_GPU = {
    'numpy': (preset_sweep('single_tone', (scale_only(),), analysis=ANALYSIS), False),
    'cupy_measurements': (cupy_sweep((scale_only(),), analysis=ANALYSIS), True),
    'cupy_calibration': (air_sweep(analysis=IQ_ONLY, calibration='cal.nc'), True),
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
