from .lib.arrays import (
    accum_rms,
    accum_rtol,
    array_namespace,
    axis_index,
    axis_slice,
    axis_to_blocks,
    binned_mean,
    configure_cupy,
    cp,
    free_cupy_mempool,
    histogram_last_axis,
    is_cupy_array,
    isroundmod,
    mean_atol,
    pinned_array_as_cupy,
    unit_roundoff,
)

# the roundoff models (`accum_rms`, `accum_rtol`, `mean_atol`, `unit_roundoff`) are
# importable from this module but left out of `__all__`, which the root namespace
# star-imports
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
