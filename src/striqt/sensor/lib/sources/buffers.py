from __future__ import annotations as __

import dataclasses
import logging
import types
from typing import Any, cast, TYPE_CHECKING
from math import ceil

import striqt.analysis as sa
import striqt.waveform as sw

from ... import specs
from .. import util

if TYPE_CHECKING:
    import numpy as np
    from ..typing import Array
    from .base import SourceBase

else:
    np = util.lazy_import('numpy')


@dataclasses.dataclass
class AcquiredIQ(sa.dataarrays.AcquiredIQ):
    """extra metadata needed for downstream analysis"""

    info: specs.AcquisitionInfo
    extra_data: dict[str, Any]
    source_spec: specs.Source
    resampler: sw.ResamplerDesign
    alias_func: specs.helpers.PathAliasFormatter | None = None
    # analysis: specs.AnalysisGroup | None = None
    voltage_scale: Array | float = 1


class ReceiveBuffers:
    """remember unused samples from the previous IQ capture"""

    carryover_samples: 'np.ndarray | None'
    start_time_ns: int | None
    buffers: list = [None, None]
    _hold_buffer_swap = False

    def __init__(self, source: 'SourceBase'):
        self.source = source
        self.buffers = [None, None]
        self.clear()

    def apply(self, samples: 'np.ndarray') -> tuple[int | None, int]:
        """carry over samples into `samples` from the previous capture.

        Returns:
            (start_time_ns, number of samples)
        """
        if self.start_time_ns is None and self.carryover_samples is not None:
            raise ValueError(
                'carryover time information present, but missing timestamp'
            )

        if not self.source.setup_spec.gapless:
            return None, 0
        elif self.carryover_samples is None:
            return self.start_time_ns, 0

        carryover = self.carryover_samples.shape[1]
        stride = samples.itemsize // self.carryover_samples.itemsize
        samples[:, : stride * carryover] = self.carryover_samples.view(samples.dtype)

        return self.start_time_ns, carryover

    def get_next(
        self, capture, overlaps: tuple[int, int] = (0, 0)
    ) -> 'tuple[np.ndarray, list[np.ndarray]]':
        """swap the buffers, and reallocate if needed"""

        if not self._hold_buffer_swap:
            self.buffers = [self.buffers[1], self.buffers[0]]
        self.buffers[0], ret = _alloc_empty_iq(
            self.source, capture, self.buffers[0], overlaps=overlaps
        )
        self._hold_buffer_swap = False
        return ret

    def skip_next_buffer_swap(self):
        self._hold_buffer_swap = True

    def stash_carryover(
        self,
        samples: 'np.ndarray',
        sample_start_ns,
        unused_sample_count: int,
        capture: specs.SensorCapture,
    ):
        """stash data needed to carry over extra samples into the next capture"""
        if not self.source.setup_spec.gapless:
            return
        carryover_count = unused_sample_count
        self.carryover_samples = samples[:, -carryover_count:].copy()
        self.start_time_ns = sample_start_ns + round(1e9 * capture.duration)

    def clear(self):
        self.carryover_samples = None
        self.start_time_ns = None

    def __del__(self):
        self.clear()
        self.buffers = [None, None]


def get_array_namespace(array_backend: specs.types.ArrayBackend) -> types.ModuleType:
    if array_backend == 'cupy':
        return sw.arrays.cp
    elif array_backend == 'numpy':
        return np
    else:
        raise TypeError('invalid array_backend argument')


@specs.helpers.convert_capture_arg(specs.SensorCapture)
@sa.util.lru_cache()
def get_read_count(
    capture: specs.SensorCapture,
    setup: specs.Source,
    *,
    include_holdoff: bool = False,
    overlap: int = 0,
) -> int:
    if overlap % 2 == 1 or overlap < 0 or not isinstance(overlap, (np.integer, int)):
        raise ValueError('overlap must be a non-negative even integer')
    if sw.isroundmod(capture.duration * capture.sample_rate, 1):
        samples_out = round(capture.duration * capture.sample_rate)
    else:
        msg = f'duration must be an integer multiple of the sample period (1/{capture.sample_rate} s)'
        raise ValueError(msg)

    from .. import compute

    resampler_design = compute.design_resampler(capture, setup.master_clock_rate)
    if capture.host_resample:
        sample_rate = resampler_design['fs']
    else:
        sample_rate = capture.sample_rate

    if compute.needs_resample(resampler_design, capture):
        nfft = resampler_design['nfft']
        min_samples_in = ceil(samples_out * nfft / resampler_design['nfft_out'])
        samples_in = min_samples_in + overlap
    else:
        samples_in = round(capture.sample_rate * capture.duration) + overlap

    if include_holdoff:
        # pad the buffer for triggering and transient holdoff
        extra_time = (setup.transient_holdoff_time or 0) + 2 * (
            setup.trigger_strobe or 0
        )
        samples_in += ceil(sample_rate * extra_time)

    return samples_in


@sa.util.stopwatch(
    'allocate buffers', 'source', threshold=5e-3, logger_level=logging.DEBUG
)
def _alloc_empty_iq(
    source: 'SourceBase',
    capture: specs.SensorCapture,
    prior: 'np.ndarray|None' = None,
    overlaps: tuple[int, int] = (0, 0),
) -> 'tuple[np.ndarray, tuple[np.ndarray, list[np.ndarray]]]':
    """allocate a buffer of IQ return values.

    Returns:
        The buffer and the list of buffer references for streaming.
    """
    count = get_read_count(
        source.capture_spec,
        source.setup_spec,
        include_holdoff=True,
        overlap=sum(overlaps),
    )

    if source.setup_spec.array_backend == 'cupy':
        try:
            from cupyx import empty_pinned as empty  # type: ignore
        except ModuleNotFoundError as ex:
            raise RuntimeError(
                'could not import the configured array backend, "cupy"'
            ) from ex
    else:
        empty = np.empty

    buf_dtype = np.dtype(source.setup_spec.transport_dtype)

    # fast reinterpretation between dtypes requires the waveform to be in the last axis
    # ports = capture.port
    if isinstance(capture.port, tuple):
        ports = tuple(capture.port)
    else:
        ports = (capture.port,)

    if prior is None or prior.shape < (len(ports), count):
        all_samples = empty((len(ports), count), dtype=np.complex64)
        samples = all_samples
    else:
        samples = all_samples = prior

    # build the list of channel buffers that will actuall be filled with data,
    # including references to the throwaway buffer of extras in case of
    # source.setup_spec.stream_all_rx_ports
    num_rx_ports = source.info.min_port_count(len(ports))
    if source.setup_spec.stream_all_rx_ports and len(ports) != num_rx_ports:
        if source.setup_spec.transport_dtype == 'complex64':
            # a throwaway buffer for samples that won't be returned
            extra_count = count
        else:
            extra_count = 2 * count

        extra = empty(extra_count, dtype=buf_dtype)
    else:
        extra = None

    extra = cast(np.ndarray, extra)

    buffers = []
    i = 0
    for channel in range(num_rx_ports):
        if channel in ports:
            buffers.append(cast(np.ndarray, samples[i].view(buf_dtype)))
            i += 1
        elif source.setup_spec.stream_all_rx_ports:
            assert extra is not None
            buffers.append(extra)

    return all_samples, (samples, buffers)


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

    # transient holdoff if we've rearmed as indicated by the presence of carryover samples
    if buffers.start_time_ns is None:
        min_holdoff = min_holdoff + round(
            source_spec.transient_holdoff_time * sample_rate
        )

    trigger_strobe = source_spec.trigger_strobe
    if trigger_strobe in (0, None):
        return min_holdoff

    trigger_strobe_ns = round(trigger_strobe * 1e9)

    # float rounding errors cause problems here; evaluate based on the 1-ns resolution
    excess_time_ns = start_time_ns % trigger_strobe_ns
    holdoff_ns = (trigger_strobe_ns - excess_time_ns) % trigger_strobe_ns
    holdoff = round(holdoff_ns / 1e9 * sample_rate)

    if holdoff < min_holdoff:
        trigger_strobe_samples = round(trigger_strobe * sample_rate)
        holdoff += ceil(min_holdoff / trigger_strobe_samples) * trigger_strobe_samples

    return holdoff


def cast_iq(spec: specs.Source, buffer: 'Array', acquired_count: int) -> 'Array':
    """cast the buffer to floating point, if necessary"""
    # array_namespace will categorize cupy pinned memory as numpy
    dtype_in = np.dtype(spec.transport_dtype)

    if spec.array_backend == 'cupy':
        xp = sw.arrays.cp
        assert xp is not None, ImportError('cupy is not installed')
        buffer = sw.arrays.pinned_array_as_cupy(buffer)
    else:
        xp = np
        buffer = xp.array(buffer)
    assert xp is not None

    # what follows is some acrobatics to minimize new memory allocation and copy
    if dtype_in.kind == 'i':
        # the same memory buffer, interpreted as int16 without casting
        buffer_int16 = buffer.view('int16')[:, : 2 * acquired_count]
        buffer_float32 = buffer.view('float32')[:, : 2 * acquired_count]

        # in-place cast from the int16 samples, filling the extra allocation in self.buffer
        xp.copyto(buffer_float32, buffer_int16, casting='unsafe')

        # re-interpret the interleaved (float32 I, float32 Q) values as a complex value
        buffer_out = buffer_float32.view('complex64')

    else:
        buffer_out = buffer[:, : 2 * acquired_count]

    return buffer_out


def get_dtype_scale(transport_dtype: specs.types.TransportDType) -> float:
    """compute the scaling factor to convert the transport dtype to full scale"""

    transport_dtype = transport_dtype
    if transport_dtype == 'int16':
        return 1.0 / float(np.iinfo(transport_dtype).max)
    else:
        return 1.0


def is_reusable(
    c1: specs.SensorCapture | None, c2: specs.SensorCapture | None, mcr: float
):
    """return True if c2 is compatible with the raw and uncalibrated IQ acquired for c1"""

    if c1 is None or c2 is None:
        return False

    from .. import compute

    fsb1 = compute.design_resampler(c1, mcr)['fs_sdr']
    fsb2 = compute.design_resampler(c2, mcr)['fs_sdr']

    if fsb1 != fsb2:
        # the realized backend sample rates need to be the same
        return False

    downstream_kws = {
        'host_resample': False,
        'backend_sample_rate': None,
        'adjust_analysis': specs.helpers.frozendict(),
    }

    c1_compare = c1.replace(**downstream_kws)
    c2_compare = c2.replace(
        # ignore parameters that only affect downstream processing
        analysis_bandwidth=c1.analysis_bandwidth,
        sample_rate=c1.sample_rate,
        **downstream_kws,
    )

    return c1_compare == c2_compare
