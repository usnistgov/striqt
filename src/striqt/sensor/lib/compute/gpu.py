"""decide whether a sweep puts work on the GPU"""

from __future__ import annotations as __

import math
from ... import specs
import striqt.analysis as sa


def sweep_touches_gpu(sweep: specs.Sweep) -> bool:
    """returns True if the specified sweep uses the GPU"""

    if sweep.source.array_backend == 'numpy':
        return False

    if sweep.source.calibration is not None:
        return True

    analysis_dict = sweep.analysis.to_dict()
    if tuple(analysis_dict.keys()) != (sa.measurements.iq_waveform.__name__,):
        # everything except iq_waveform requires a warmup
        return True

    # check the inner loop (explicit) values
    for capture in sweep.captures:
        if capture.host_resample or math.isfinite(capture.analysis_bandwidth):
            return True

    # check any values specified in outer loops
    for loop in sweep.loops:
        if loop.isin == 'analysis':
            continue
        if loop.field == 'host_resample' and True in loop.get_points():
            return True
        elif loop.field == 'analysis_bandwidth':
            points = loop.get_points()
            if any(p is not None and math.isfinite(p) for p in points):
                return True

    return False
