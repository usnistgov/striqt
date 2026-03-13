from __future__ import annotations as __

import functools
import itertools
import math
from typing import Any, Callable, cast, Generator, TypeVar, TYPE_CHECKING

import array_api_compat

from . import util

_TC = TypeVar('_TC', bound=Callable)


if TYPE_CHECKING:
    from types import ModuleType
    from .typing import ArrayLike, Array, TypeIsCupy
    import numpy as np

    try:
        import cupy as cp  # pyright: ignore[reportMissingImports]
    except ModuleNotFoundError:
        cp = None

else:
    np = util.lazy_import('numpy')
    try:
        cp = util.lazy_import('cupy')
    except ImportError:
        cp = None


#%% rounding

def isroundmod(value: float | np.ndarray, div, atol=1e-6) -> bool:
    ratio = value / div
    try:
        return abs(math.remainder(ratio, 1)) <= atol
    except TypeError:
        return np.abs(np.rint(ratio) - ratio) <= atol


#%% dtype tricks
@util.lru_cache()
def dtype_change_float(
    dtype, float_basis_dtype
) -> type[np.complexfloating] | type[np.floating]:
    """return a complex or float dtype similar to `dtype`, but
    with a float backing with size matching `float_basis_dtype`.

    Examples:
        dtype_change_float(np.complex128, np.float32) -> np.complex64
        dtype_change_float(np.float64, np.float32) -> np.float32
    """

    np_input_type = np.dtype(dtype).type
    np_float_type = np.finfo(np.dtype(float_basis_dtype)).dtype.type

    if np_input_type in (np.complex128, np.complex64):
        if np_float_type is np.float32:
            return np.complex64
        elif np_float_type is np.float64:
            return np.complex128
    elif np_input_type in (np.float16, np.float32, np.float64):
        return np_float_type

    raise ValueError(
        f'unable to identify output dtype similar to {dtype} matching floating point {float_basis_dtype}'
    )


def float_dtype_like(x: Array, min_dtype: Any | None = None):
    """returns a floating-point dtype corresponding to x.

    `x` may be complex, in which case the returned data type corresponds to
    that of the `x.real` or `x.imag`.

    Args:
        min_dtype: dtype with the minimum acceptable float size, or None for no minimum

    Returns:
    * If x.dtype is float16/float32/float64: x.dtype.
    * If x.dtype is complex64/complex128: float32/float64
    """

    if isinstance(x, (int, float)):
        x = np.asarray(x)
        xp = np
    else:
        xp = array_namespace(x)

    try:
        dtype = np.finfo(xp.asarray(x).dtype).dtype
    except ValueError:
        dtype = np.dtype('float32')

    if min_dtype is None:
        pass
    else:
        min_dtype = np.dtype(min_dtype)
        if min_dtype.itemsize > dtype.itemsize:
            dtype = min_dtype

    return dtype


#%% sliding or binned window operations
def binned_mean(
    x: Array,
    count,
    *,
    axis=0,
    truncate=True,
    reject_extrema=False,
    fft=True,
) -> Array:
    """reduce an array by averaging into bins on the specified axis.

    Arguments:
        x: input array
        count: bin count to average
        axis: axis along which to implement the binned mean
        truncate: True to truncate incomplete bins at the edges, or False to raise exception
        reject_extrema: if True, the min and max samples from each bin will be excluded
        fft: if True, bins align with fft bins (centered, instead of left side)
    """

    xp = array_namespace(x)

    if not truncate:
        pass
    elif fft:
        # enforce that index 0 is a center bin
        center_bin = x.shape[axis] // 2
        size_left = center_bin - count // 2
        blocks_left = size_left // count
        block_count = 2 * blocks_left + 1
        start = center_bin - (count * block_count) // 2
        stop = start + count * block_count

        if start > 0 or stop < x.shape[axis]:
            x = axis_slice(x, start, stop, axis=axis)
    else:
        trim = x.shape[axis] % (count)
        if trim:
            dimsize = (x.shape[axis] // count) * count
            x = axis_slice(x, 0, dimsize, axis=axis)

    x = axis_to_blocks(x, count, axis=axis)
    stat_axis = axis + 1 if axis >= 0 else axis
    if reject_extrema:
        x = np.sort(x, axis=stat_axis)
        x = axis_slice(x, 1, -1, axis=stat_axis)
    ret = xp.nanmean(x, axis=stat_axis)

    return ret


@util.lru_cache()
def sliding_window_output_shape(
    array_shape: tuple[int, ...] | int, window_shape: tuple, axis
):
    """return the shape of the output of sliding_window_view, for example
    to pre-create an output buffer."""
    try:
        # numpy >= 2?
        from numpy.lib import _stride_tricks_impl as stride_tricks
    except ImportError:
        # numpy < 2?
        from numpy.lib import stride_tricks

    if not isinstance(array_shape, tuple):
        array_shape = (array_shape,)

    if min(window_shape) < 0:
        raise ValueError('`window_shape` cannot contain negative values')
    ndim = len(array_shape)
    if axis is None:
        axis = tuple(range(ndim))
        if len(window_shape) != len(axis):
            raise ValueError(
                f'Since axis is `None`, must provide window_shape for all dimensions of `x`; got {len(window_shape)} window_shape elements and `x.ndim` is {ndim}.'
            )
    else:
        axis = stride_tricks.normalize_axis_tuple(axis, ndim, allow_duplicate=True)  # type: ignore
        if len(window_shape) != len(axis):
            raise ValueError(
                f'Must provide matching length window_shape and axis; got {len(window_shape)} window_shape elements and {len(axis)} axes elements.'
            )

    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,)
    x_shape_trimmed = list(array_shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError('window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] -= dim - 1
    return tuple(x_shape_trimmed) + window_shape


def sliding_window_view(x, window_shape, axis=None, *, subok=False, writeable=False):
    """
    Create a sliding window view into the array with the given window shape.

    Also known as rolling or moving window, the window slides across all
    dimensions of the array and extracts subsets of the array at all window
    positions.


    Parameters
    ----------
    x : array_like
        Array to create the sliding window view from.
    window_shape : int or tuple of int
        Size of window over each axis that takes part in the sliding window.
        If `axis` is not present, must have same length as the number of input
        array dimensions. Single integers `i` are treated as if they were the
        tuple `(i,)`.
    axis : int or tuple of int, optional
        Axis or axes along which the sliding window is applied.
        By default, the sliding window is applied to all axes and
        `window_shape[i]` will refer to axis `i` of `x`.
        If `axis` is given as a `tuple of int`, `window_shape[i]` will refer to
        the axis `axis[i]` of `x`.
        Single integers `i` are treated as if they were the tuple `(i,)`.
    subok : bool, optional
        If True, sub-classes will be passed-through, otherwise the returned
        array will be forced to be a base-class array (default).
    writeable : bool, optional -- not supported
        When true, allow writing to the returned view. The default is false,
        as this should be used with caution: the returned view contains the
        same memory location multiple times, so writing to one location will
        cause others to change.

    Returns
    -------
    view : ndarray
        Sliding window view of the array. The sliding window dimensions are
        inserted at the end, and the original dimensions are trimmed as
        required by the size of the sliding window.
        That is, ``view.shape = x_shape_trimmed + window_shape``, where
        ``x_shape_trimmed`` is ``x.shape`` with every entry reduced by one less
        than the corresponding window size.


    See also
    --------
    numpy.lib.stride_tricks.as_strided

    Notes
    --------
    This function is adapted from numpy.lib.stride_tricks.as_strided.

    Examples
    --------
    >>> x = _cupy.arange(6)
    >>> x.shape
    (6,)
    >>> v = sliding_window_view(x, 3)
    >>> v.shape
    (4, 3)
    >>> v
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])

    """

    try:
        # numpy >= 2?
        from numpy.lib import _stride_tricks_impl as stride_tricks
    except ImportError:
        # numpy < 2?
        from numpy.lib import stride_tricks

    xp = array_namespace(x, use_compat=False)

    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,)

    # writeable is not supported:
    if writeable:
        raise NotImplementedError('Writeable views are not supported.')

    # first convert input to array, possibly keeping subclass
    x = xp.array(x, copy=False, subok=subok)

    out_shape = sliding_window_output_shape(x.shape, window_shape, axis)
    axis = stride_tricks.normalize_axis_tuple(axis, x.ndim)  # type: ignore
    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    return xp.lib.stride_tricks.as_strided(x, strides=out_strides, shape=out_shape)


def axis_to_blocks(y: Array, size: int, truncate=False, axis=0) -> Array:
    """Returns a view on y reshaped into blocks along axis `axis`.

    Args:
        y: an input array of size (N[0], ... N[K-1])

    Raises:
        TypeError: if not isinstance(size, int)

        IndexError: if y.size == 0

        ValueError: if truncate == False and y.shape[axis] % size != 0

    Returns:
        view on `y` with shape (..., N[axis]//size, size, ..., N[K-1]])
    """

    if not isinstance(size, int):
        raise TypeError('block size must be integer')
    if y.size == 0:
        raise IndexError('cannot form blocks on arrays of size 0')

    # ensure the axis dimension is a multiple of the block size
    ax_size = y.shape[axis]
    if ax_size % size != 0:
        if not truncate:
            raise ValueError(
                f'axis 0 size {ax_size} is not a factor of block size {size}'
            )

        slices = len(y.shape) * [slice(None, None)]
        slices[axis] = slice(None, size * (ax_size // size))
        y = y.__getitem__(tuple(slices))

    if axis == -1:
        shape_after = ()
    else:
        shape_after = y.shape[axis + 1 :]
    newshape = y.shape[:axis] + (ax_size // size, size) + shape_after

    return y.reshape(newshape)




def histogram_last_axis(
    x: Array, bins: int | Array, range: tuple | None = None
) -> Array:
    """computes a histogram along the last axis of an input array.

    For reference see https://stackoverflow.com/questions/44152436/calculate-histograms-along-axis

    Args:
        x: input data of shape (M[0], ..., M[K-1], N)
        bins: Number of bins in the histogram, or a vector of bin edges
        range: Bounds on the histogram bins [lower bound, upper bound] inclusive

    Returns:
        np.ndarray of shape (M[0], ..., M[K-1], n_bins)
    """

    xp = array_namespace(x)

    # Setup bins and determine the bin location for each element for the bins
    hist_size = x.shape[-1]

    if isinstance(bins, int):
        if range is None:
            range = x.min(), x.max()
        bins = xp.linspace(range[0], range[1], bins + 1)
    else:
        bins = xp.asarray(bins)

    size = bins.size  # type: ignore
    flat = x.reshape(-1, hist_size)
    idx = xp.searchsorted(bins, flat, 'right') - 1

    # Some elements would be off limits, so get a mask for those
    bad_mask = (idx == -1) | (idx == size)

    # We need to use bincount to get bin based counts. To have unique IDs for
    # each row and not get confused by the ones from other rows, we need to
    # offset each row by a scale (using row length for this).
    scaled_idx = size * xp.arange(flat.shape[0])[:, None] + idx

    # Set the bad ones to be last possible index+1 : n_bins*data2D.shape[0]
    limit = size * flat.shape[0]
    scaled_idx[bad_mask] = limit

    # Get the counts and reshape to multi-dim
    counts = xp.bincount(scaled_idx.ravel(), minlength=limit + 1)[:-1]
    counts.shape = x.shape[:-1] + (size,)
    return counts[..., :-1], bins


#%% slicing and indexing
def axis_index(a, index, axis=-1):
    """Return a boolean-indexed selection on axis `axis' from `a'.

    Arguments:
    a: numpy.ndarray
        The array to be sliced.
    mask: boolean index array of size a.shape[axis]
    axis : int, optional
        The axis of `a` to be sliced.
    """
    before, after = _pad_slices_to_dim(a.ndim, axis)
    return a[before + (index,) + after]


def axis_slice(a, start, stop=None, step=None, axis=-1):
    """Return a slice on the array `a` on the axis index `axis`.

    Arguments:
    a: numpy.ndarray
        The array to be sliced.
    start, stop=None, step:
        The arguments to `slice` on that axis.
    axis : int, optional
        The axis of `a` to be sliced.
    """

    before, after = _pad_slices_to_dim(a.ndim, axis)
    sl = slice(start, stop, step)
    return a[before + (sl,) + after]


@util.lru_cache()
def grouped_slices_along_axis(shape: tuple[int, ...], max_size: int, axis: int):
    if axis < 0:
        axis = len(shape) + axis

    # tracks the size of all axes > iax
    size_rest = math.prod(shape)

    slices_per_ax = []
    for iax, n in enumerate(shape):
        if iax == axis or size_rest < max_size:
            slices_per_ax.append((slice(None, None),))
            continue

        want_count = max(util.ceildiv(size_rest, max_size), 1)
        count = min(want_count, n)
        step = n // count

        new = (slice(i, min(n, i + step)) for i in range(0, n, step))
        slices_per_ax.append(tuple(new))

        size_rest = size_rest // count

    return slices_per_ax


def grouped_views_along_axis(
    x: Array, max_size: int, axis: int = 0
) -> Generator[ArrayLike]:
    if x.size < max_size:
        yield x
        return

    ax_steps = grouped_slices_along_axis(x.shape, max_size, axis)

    slices = itertools.product(*ax_steps)

    empty = True
    for slice_ in slices:
        empty = False
        yield x[slice_]

    if empty:
        yield x


#%% padding
@functools.cache
def _pad_slices_to_dim(ndim: int, axis: int, /):
    if not isinstance(axis, int):
        raise TypeError('axis argument must be integer')

    if axis < 0:
        axis = ndim + axis

        if axis < 0:
            raise ValueError(f'axis {axis} exceeds the number of dimensions')

    if axis <= ndim // 2:
        before = (slice(None),) * (axis)
        after = ()
    else:
        before = (Ellipsis,)
        after = (slice(None),) * (ndim - axis - 1)

    return before, after


def pad_along_axis(a, pad_width: list, axis=0, *args, **kws):
    if axis >= 0:
        pre_pad = [[0, 0]] * axis
    else:
        pre_pad = [[0, 0]] * (axis + a.ndim - 1)

    xp = array_namespace(a)
    return xp.pad(a, pre_pad + pad_width, *args, **kws)


#%% cupy configuration and memory management
def pinned_array_as_cupy(x, stream=None):
    assert cp is not None
    out = cp.empty_like(x)
    out.data.copy_from_host_async(x.ctypes.data, x.data.nbytes, stream=stream)
    return out


def sync_if_cupy(x: Array):
    if is_cupy_array(x) and cp is not None:
        stream = cp.cuda.get_current_stream()  # type: ignore
        stream.synchronize()


@functools.cache
def configure_cupy():
    if cp is not None:
        import cupy.fft as fft  # type: ignore

        # the FFT plan sets up large caches that don't help us
        fft.config.get_plan_cache().set_size(0)  # type: ignore
        cp.cuda.set_pinned_memory_allocator(None)  # type: ignore


def free_cupy_mempool():
    if cp is not None:
        mempool = cp.get_default_memory_pool()  # type: ignore
        if mempool is not None:
            mempool.free_all_blocks()


@util.lru_cache()
def set_cuda_mem_limit(fraction=0.75):
    if cp is None:
        return

    cp.get_default_memory_pool().set_limit(fraction=fraction)  # type: ignore

    # Alternative: select an absolute amount of memory
    #
    # import psutil
    # available = psutil.virtual_memory().available


def is_cupy_array(x: object) -> TypeIsCupy:
    return array_api_compat.is_cupy_array(x)


#%% Compatibility shims between array APIs
class NonStreamContext:
    """a do-nothing cupy.Stream duck type stand-in for array types that do not support synchronization"""

    def __init__(self, *args, **kws):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def synchronize(self):
        pass

    def use(self):
        pass


def array_stream(obj: Array, null=False, non_blocking=False, ptds=False):
    """returns a cupy.Stream (or a do-nothing stand in) object as appropriate for obj"""
    if is_cupy_array(obj) and cp is not None:
        return cp.cuda.Stream(null=null, non_blocking=non_blocking, ptds=ptds)  # type: ignore
    else:
        return NonStreamContext()


def array_namespace(a, use_compat=False) -> ModuleType:
    try:
        return array_api_compat.array_namespace(a, use_compat=use_compat)
    except TypeError:
        pass

    try:
        import mlx.core as mx  # type: ignore

        if isinstance(a, mx.array):
            return mx
        else:
            raise TypeError
    except (ImportError, TypeError):
        pass

    raise TypeError('unrecognized object type')


def convert_np_to_xp(func: _TC) -> _TC:
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        import array_api_compat.numpy as anp

        xp = kwargs.get('xp', anp)
        if xp is None or xp is np:
            xp = anp

        if xp is anp:
            return func(*args, **kwargs)

        kwargs_with_np = dict(kwargs, xp=anp)
        x = func(*args, **kwargs_with_np)
        if hasattr(xp, 'asarray'):
            x = xp.asarray(x)
        elif hasattr(xp, 'array'):
            x = xp.array(x)  # type: ignore
        else:
            raise AttributeError(f'invalid array module {xp}')

        return xp.asarray(x)

    return cast(_TC, wrapped)


