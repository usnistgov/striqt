from .lib.arrays import (
    accum_rms,
    accum_rtol,
    axis_to_blocks,
    axis_index,
    axis_slice,
    array_namespace,
    binned_mean,
    configure_cupy,
    cp,
    free_cupy_mempool,
    histogram_last_axis,
    isroundmod,
    is_cupy_array,
    mean_atol,
    pinned_array_as_cupy,
    unit_roundoff,
)

# the roundoff models stay reachable as `arrays.accum_rms` etc. but are kept out of
# the root namespace, which star-imports this module
__all__ = [
    'array_namespace',
    'axis_index',
    'axis_slice',
    'axis_to_blocks',
    'binned_mean',
    'configure_cupy',
    'cp',
    'free_cupy_mempool',
    'histogram_last_axis',
    'is_cupy_array',
    'isroundmod',
    'pinned_array_as_cupy',
]
