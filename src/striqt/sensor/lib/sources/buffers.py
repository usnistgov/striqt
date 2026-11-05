from __future__ import annotations as __

import logging
import types
import typing
from math import ceil, isfinite
from typing import TYPE_CHECKING, cast

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis.lib.util import np

from ... import specs
from ..typing import SC

if TYPE_CHECKING:
    from ..controller import Controller
    from ..typing import Array


class ReceiveBuffers(typing.Generic[SC]):
    """the receive buffers of a controller and the carryover between acquisitions.

    An instance keeps the unused tail of an acquisition from `stash_carryover` until
    `clear`. `apply` copies that tail into the head of the next receive buffer.
    """

    carryover_samples: np.ndarray | None
    start_time_ns: int | None
    buffers: list
    extra: np.ndarray | None
    _hold_buffer_swap = False

    def __init__(self, controller: Controller):
        self.controller = controller
        self.buffers = [None, None]
        self.extra = None
        self.clear()

    def apply(self, samples: np.ndarray) -> tuple[int | None, int]:
        """copy the stashed carryover into the head of `samples`.

        Args:
            samples: the receive buffer of the next acquisition, shape (ports, count)

        Returns:
            the stashed time (ns) of ``samples[:, 0]``, or `None` without a stash,
            and the number of carried-over samples, 0 without a stash

        Raises:
            ValueError: a stash is present without its timestamp
        """
        if self.start_time_ns is None and self.carryover_samples is not None:
            raise ValueError(
                'carryover time information present, but missing timestamp'
            )

        if self.carryover_samples is None:
            return self.start_time_ns, 0

        carryover = self.carryover_samples.shape[1]
        stride = samples.itemsize // self.carryover_samples.itemsize
        samples[:, : stride * carryover] = self.carryover_samples.view(samples.dtype)

        return self.start_time_ns, carryover

    def get_next(
        self,
        capture: SC,
        overlaps: tuple[int, int] = (0, 0),
        *,
        include_holdoff: bool = True,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """swap the buffers and return the next one, reallocated when too small.

        Args:
            capture: the armed capture, which sets the port count and the read count
            overlaps: the ``(lead, tail)`` overlap in source samples
            include_holdoff: True sizes the buffer for the transient and strobe
                holdoff ahead of the record; False sizes it for the record and
                overlaps alone, as a burst segment that continues the stream reads

        Returns:
            the buffer with shape (ports, count) and the per-port stream views
        """

        if not self._hold_buffer_swap:
            self.buffers = [self.buffers[1], self.buffers[0]]
        self.buffers[0], ret, self.extra = _alloc_empty_iq(
            self.controller,
            capture,
            self.buffers[0],
            overlaps=overlaps,
            include_holdoff=include_holdoff,
            prior_extra=self.extra,
        )
        self._hold_buffer_swap = False
        return ret

    def skip_next_buffer_swap(self):
        self._hold_buffer_swap = True

    def stash_carryover(
        self,
        samples: np.ndarray,
        sample_start_ns: int,
        unused_sample_count: int,
        capture: specs.SensorCapture,
        start_overlap: int = 0,
    ) -> None:
        """stash the unused tail of `samples` for the head of the next acquisition.

        The stashed `start_time_ns` is the time of the first carried-over sample. The
        stash replaces any earlier one and persists until `clear`.

        Args:
            samples: the source-rate samples returned for the capture, with shape
                (ports, count)
            sample_start_ns: the time (in ns) of the first record sample,
                ``samples[:, start_overlap]``
            unused_sample_count: the count of trailing samples of `samples` past
                the record
            capture: the capture; the resampler design of the capture sets the
                source rate
            start_overlap: the lead overlap (in source-rate samples) that precedes
                the record in `samples`
        """
        from ..compute import design_resampler

        carryover_count = unused_sample_count
        self.carryover_samples = samples[:, -carryover_count:].copy()

        carryover_index = samples.shape[1] - carryover_count
        fs = design_resampler(capture, self.controller.source_spec.master_clock_rate)[
            'fs_sdr'
        ]
        self.start_time_ns = sample_start_ns + round(
            1e9 * (carryover_index - start_overlap) / fs
        )

    def clear(self) -> None:
        """drop the stashed carryover and its timestamp"""
        self.carryover_samples = None
        self.start_time_ns = None

    def __del__(self):
        self.clear()
        self.buffers = [None, None]
        self.extra = None


def get_array_namespace(array_backend: specs.types.ArrayBackend) -> types.ModuleType:
    if array_backend == 'cupy':
        if sw.arrays.cp is None:
            # trigger the import error
            import cupy  # type: ignore  # noqa: F401

            raise ImportError
        else:
            mod = sw.arrays.cp
    elif array_backend == 'numpy':
        mod = np
    else:
        raise TypeError('invalid array_backend argument')

    return cast(types.ModuleType, mod)


@sa.specs.helpers.lru_cache_on_converted(specs.SensorCapture)
def get_read_count(
    capture: specs.SensorCapture,
    setup: specs.Source,
    *,
    include_holdoff: bool = False,
    overlap: int = 0,
) -> int:
    if overlap < 0 or not isinstance(overlap, (np.integer, int)):
        raise ValueError('overlap must be a non-negative integer')
    if not sw.isroundmod(capture.duration * capture.sample_rate, 1):
        msg = f'duration must be an integer multiple of the sample period (1/{capture.sample_rate} s)'
        raise ValueError(msg)

    from .. import compute

    resampler_design = compute.design_resampler(capture, setup.master_clock_rate)
    if capture.host_resample:
        sample_rate = resampler_design['fs']
    else:
        sample_rate = capture.sample_rate

    if compute.needs_resample(resampler_design, capture):
        samples_in = compute.source_record_size(capture, resampler_design) + overlap
    else:
        samples_in = round(capture.sample_rate * capture.duration) + overlap

    if include_holdoff:
        # pad the buffer for triggering and transient holdoff
        tholdoff = setup.transient_holdoff_time or 0
        tstrobe = 2 * (setup.trigger_strobe or 0)
        samples_in += ceil(sample_rate * (tholdoff + tstrobe))

    return samples_in


@sa.util.stopwatch(
    'allocate buffers', 'source', threshold=5e-3, logger_level=logging.DEBUG
)
def _alloc_empty_iq(
    controller: Controller,
    capture: specs.SensorCapture,
    prior: np.ndarray | None = None,
    overlaps: tuple[int, int] = (0, 0),
    *,
    include_holdoff: bool = True,
    prior_extra: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, list[np.ndarray]], np.ndarray | None]:
    """allocate a buffer of IQ return values, or reuse `prior` when it is large enough.

    Args:
        controller: the controller with the armed capture and the source spec
        capture: the capture that sets the port rows
        prior: a buffer to reuse when it has one row per port and at least the read
            count of columns
        overlaps: the ``(lead, tail)`` overlap in source samples
        include_holdoff: True adds the transient and strobe holdoff to the read count
        prior_extra: a throwaway buffer to reuse for the RX ports outside the capture
            when it is at least the read count in the transport dtype

    Returns:
        The buffer, the ``(samples, buffers)`` pair for streaming, and the throwaway
        buffer for the RX ports outside the capture (None when none is streamed).
    """
    count = get_read_count(
        controller.capture_spec,
        controller.source_spec,
        include_holdoff=include_holdoff,
        overlap=sum(overlaps),
    )

    if controller.source_spec.array_backend == 'cupy':
        try:
            from cupyx import empty_pinned as empty  # type: ignore
        except ModuleNotFoundError as ex:
            raise RuntimeError(
                'could not import the configured array backend, "cupy"'
            ) from ex
    else:
        empty = np.empty

    buf_dtype = np.dtype(controller.source_spec.transport_dtype)

    # fast reinterpretation between dtypes requires the waveform to be in the last axis
    # ports = capture.port
    if isinstance(capture.port, tuple):
        ports = tuple(capture.port)
    else:
        ports = (capture.port,)

    # every row of a reused buffer is returned, but only a slice of its columns
    if prior is None or prior.shape[0] != len(ports) or prior.shape[1] < count:
        all_samples = empty((len(ports), count), dtype=np.complex64)
        samples = all_samples
    else:
        samples = all_samples = prior

    # the channel buffers that the stream fills. Under `stream_all_rx_ports`, ports
    # outside the capture point at a throwaway buffer
    num_rx_ports = controller.source_info.min_port_count(len(ports))
    if controller.source_spec.stream_all_rx_ports and len(ports) != num_rx_ports:
        if controller.source_spec.transport_dtype == 'complex64':
            # a throwaway buffer for samples that won't be returned
            extra_count = count
        else:
            extra_count = 2 * count

        if (
            prior_extra is None
            or prior_extra.dtype != buf_dtype
            or prior_extra.size < extra_count
        ):
            extra = empty(extra_count, dtype=buf_dtype)
        else:
            extra = prior_extra
    else:
        extra = None

    buffers = []
    i = 0
    for channel in range(num_rx_ports):
        if channel in ports:
            buffers.append(cast(np.ndarray, samples[i].view(buf_dtype)))
            i += 1
        elif controller.source_spec.stream_all_rx_ports:
            assert extra is not None
            buffers.append(extra)

    return all_samples, (samples, buffers), extra


def find_trigger_holdoff(
    source_spec: specs.Source,
    capture_spec: specs.SensorCapture,
    buffers: ReceiveBuffers,
    start_time_ns: int,
    start_overlap: int = 0,
):
    from ..compute import design_resampler

    resampler = design_resampler(capture_spec, source_spec.master_clock_rate)
    sample_rate = resampler['fs_sdr']
    min_holdoff = start_overlap

    # a rearm leaves no carryover samples and requires the transient holdoff
    if buffers.start_time_ns is None:
        min_holdoff = min_holdoff + round(
            source_spec.transient_holdoff_time * sample_rate
        )

    trigger_strobe = source_spec.trigger_strobe
    if trigger_strobe in (0, None):
        return min_holdoff

    trigger_strobe_ns = round(trigger_strobe * 1e9)

    # integer ns arithmetic avoids the float rounding error in the modulo
    excess_time_ns = start_time_ns % trigger_strobe_ns
    holdoff_ns = (trigger_strobe_ns - excess_time_ns) % trigger_strobe_ns
    holdoff = round(holdoff_ns / 1e9 * sample_rate)

    if holdoff < min_holdoff:
        # the first strobe edge at or past min_holdoff
        trigger_strobe_samples = round(trigger_strobe * sample_rate)
        strobes = ceil((min_holdoff - holdoff) / trigger_strobe_samples)
        holdoff += strobes * trigger_strobe_samples

    return holdoff


def cast_iq(
    spec: specs.Source, buffer: Array, acquired_count: int, *, start: int = 0
) -> Array:
    """return `acquired_count` samples per port of `buffer` from column `start` as complex64.

    The result is a copy, and the receive buffer is free for the next read when the
    call returns. The columns of `buffer` outside the span contribute nothing to the
    copy.

    Args:
        spec: the source spec, for `transport_dtype` and `array_backend`
        buffer: the receive buffer with shape (ports, width) of complex64 slots;
            under an int16 transport the interleaved (I, Q) pairs occupy the head
            of each row
        acquired_count: the number of complex samples per port to keep
        start: the first complex sample of each port to keep

    Returns:
        a C-contiguous complex64 array with shape (ports, acquired_count) in the
        namespace of ``spec.array_backend``

    Raises:
        ValueError: if `start` is negative or ``start + acquired_count`` exceeds the
            width of `buffer`
    """
    ports, width = buffer.shape
    if start < 0 or start + acquired_count > width:
        raise ValueError(
            f'the span [{start}, {start + acquired_count}) exceeds the buffer '
            f'width {width}'
        )

    dtype_in = np.dtype(spec.transport_dtype)
    if dtype_in.kind == 'i':
        # the int16 pairs of complex sample k are at int16 index 2k of the row
        src = buffer.view('int16')[:, 2 * start : 2 * (start + acquired_count)]
    else:
        src = buffer[:, start : start + acquired_count]

    # array_namespace categorizes cupy pinned memory as numpy
    if spec.array_backend == 'cupy':
        xp = sw.arrays.cp
        assert xp is not None, ImportError('cupy is not installed')
        out = xp.empty((ports, acquired_count), dtype='complex64')
        sw.arrays.pinned_array_as_cupy(src, out=out)
        # the H2D copy is asynchronous, and the next segment of a burst reads into
        # the same pinned buffer 2 acquisitions later
        xp.cuda.get_current_stream().synchronize()
        if dtype_in.kind == 'i':
            # the int16 pairs are at the head of each row of `out`, and the widening
            # cast in place avoids a second device allocation and copy
            xp.copyto(
                out.view('float32'),
                out.view('int16')[:, : 2 * acquired_count],
                casting='unsafe',
            )
    elif dtype_in.kind == 'i':
        out = np.empty((ports, acquired_count), dtype='complex64')
        np.copyto(out.view('float32'), src, casting='unsafe')
    else:
        out = np.array(src)

    return out


def get_dtype_scale(transport_dtype: specs.types.TransportDType) -> float:
    """compute the scaling factor to convert the transport dtype to full scale"""

    if transport_dtype == 'int16':
        return 1.0 / float(np.iinfo(transport_dtype).max)
    else:
        return 1.0


def is_reusable(
    c1: specs.SensorCapture | None, c2: specs.SensorCapture | None, mcr: float
) -> bool:
    """return whether the raw IQ acquired for `c1` serves `c2` as well.

    Reuse requires the same radio sample rate and LO offset from `design_resampler`
    at `mcr`, an `analysis_bandwidth` that is finite for both or infinite for both,
    and equality on every other field except `sample_rate`, `host_resample`,
    `backend_sample_rate`, and `adjust_analysis`. A None capture is never reusable.
    """

    if c1 is None or c2 is None:
        return False

    from .. import compute

    d1 = compute.design_resampler(c1, mcr)
    d2 = compute.design_resampler(c2, mcr)

    if d1['fs_sdr'] != d2['fs_sdr'] or d1['lo_offset'] != d2['lo_offset']:
        # the radio must be clocked and tuned the same way
        return False
    if isfinite(c1.analysis_bandwidth) != isfinite(c2.analysis_bandwidth):
        return False

    downstream_kws = {
        'host_resample': False,
        'backend_sample_rate': None,
        'adjust_analysis': specs.frozendict(),
    }

    c1_compare = c1.replace(**downstream_kws)
    c2_compare = c2.replace(
        sample_rate=c1.sample_rate,
        analysis_bandwidth=c1.analysis_bandwidth,
        **downstream_kws,
    )

    return c1_compare == c2_compare
