from __future__ import annotations as __

import dataclasses
import logging
import types
from typing import Any, cast, overload, TYPE_CHECKING
from math import ceil, isfinite

import striqt.analysis as sa
import striqt.waveform as sw

from ... import specs
from .. import util

if TYPE_CHECKING:
    import numpy as np
    from ..typing import Array, ResamplerKws, Unpack
    from .base import SourceBase

else:
    np = util.lazy_import('numpy')


FILTER_SIZE = 4001
MIN_OARESAMPLE_FFT_SIZE = 4 * 4096 - 1
RESAMPLE_COLA_WINDOW = 'hamming'
FILTER_DOMAIN = 'time'


@dataclasses.dataclass
class AcquiredIQ(sa.dataarrays.AcquiredIQ):
    """extra metadata needed for downstream analysis"""

    info: specs.AcquisitionInfo
    extra_data: dict[str, Any]
    alias_func: specs.helpers.PathAliasFormatter | None
    source_spec: specs.Source
    resampler: sw.fourier.ResamplerDesign
    trigger: sa.Trigger | None
    analysis: specs.AnalysisGroup | None = None
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

        if self.carryover_samples is None:
            carryover_count = 0
        else:
            # note: carryover.samples.dtype is np.complex64, samples.dtype is np.float32
            carryover_count = self.carryover_samples.shape[1]
            samples[:, : 2 * carryover_count] = self.carryover_samples.view(
                samples.dtype
            )

        return self.start_time_ns, carryover_count

    def get_next(self, capture) -> 'tuple[np.ndarray, list[np.ndarray]]':
        """swap the buffers, and reallocate if needed"""

        if not self._hold_buffer_swap:
            self.buffers = [self.buffers[1], self.buffers[0]]
        self.buffers[0], ret = _alloc_empty_iq(self.source, capture, self.buffers[0])
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
        return sa.util.cp
    elif array_backend == 'numpy':
        return np
    else:
        raise TypeError('invalid array_backend argument')


@overload
def get_trigger_from_spec(setup: specs.Source, analysis: None = None) -> None: ...


@overload
def get_trigger_from_spec(
    setup: specs.Source, analysis: specs.AnalysisGroup
) -> sa.Trigger: ...


@sa.util.lru_cache()
def get_trigger_from_spec(
    setup: specs.Source, analysis: specs.AnalysisGroup | None = None
) -> sa.Trigger | None:
    name = get_signal_trigger_name(setup)
    if name is None:
        return None

    if analysis is None and isinstance(setup.signal_trigger, specs.AnalysisGroup):
        analysis = setup.signal_trigger

    if analysis is None:
        meas_name = sa.register.get_signal_trigger_measurement_name(name, sa.registry)
        raise ValueError(
            f'signal_trigger {meas_name!r} requires an analysis specification for {setup.signal_trigger!r}'
        )
    elif isinstance(analysis, specs.AnalysisGroup):
        return sa.Trigger.from_spec(name, analysis, registry=sa.registry)
    elif isinstance(analysis, Analysis):
        return sa.Trigger(setup.signal_trigger, analysis, registry=sa.registry)


@specs.helpers.convert_capture_arg(specs.SensorCapture)
@sa.util.lru_cache(30000)
def design_resampler(
    capture: specs.SensorCapture,
    master_clock_rate: float,
    backend_sample_rate: float | None = None,
    **kwargs: Unpack[ResamplerKws],
) -> sw.fourier.ResamplerDesign:
    """design a filter specified by the capture for a radio with the specified MCR.

    For the return value, see `striqt.waveform.fourier.design_cola_resampler`
    """
    kwargs.setdefault('bw_lo', 0.25e6)
    kwargs.setdefault('min_oversampling', 1.1)
    kwargs.setdefault('window', RESAMPLE_COLA_WINDOW)
    kwargs.setdefault('min_fft_size', MIN_OARESAMPLE_FFT_SIZE)

    from .. import resampling

    if resampling.USE_OARESAMPLE:
        kwargs['min_fft_size'] = MIN_OARESAMPLE_FFT_SIZE
    else:
        # this could probably be set to 1?
        kwargs['min_fft_size'] = 256

    if str(capture.lo_shift).lower() == 'none':
        lo_shift = False
    else:
        lo_shift = capture.lo_shift

    if (
        capture.analysis_bandwidth != float('inf')
        and capture.analysis_bandwidth > capture.sample_rate
    ):
        raise ValueError(
            f'analysis bandwidth must be smaller than sample rate in {capture}'
        )

    if master_clock_rate is not None:
        mcr = master_clock_rate
    elif capture.backend_sample_rate is not None:
        mcr = capture.backend_sample_rate
    else:
        raise TypeError(
            'must specify source.master_clock_rate or capture.backend_sample_rate'
        )

    if capture.backend_sample_rate is None:
        fs_sdr = backend_sample_rate
    else:
        fs_sdr = capture.backend_sample_rate

    if capture.host_resample:
        # use GPU DSP to resample from integer divisor of the MCR
        # fs_sdr, lo_offset, kws  = iqwaveform.fourier.design_cola_resampler(
        design = sw.fourier.design_cola_resampler(
            fs_base=mcr,
            fs_target=capture.sample_rate,
            bw=capture.analysis_bandwidth,
            shift=lo_shift,
            fs_sdr=fs_sdr,
            **kwargs,
        )

        if 'window' in kwargs:
            design['window'] = kwargs['window']

        return design

    elif lo_shift:
        raise ValueError('lo_shift requires host_resample=True')
    elif mcr < capture.sample_rate:
        raise ValueError('upsampling requires host_resample=True')
    else:
        # use the SDR firmware to implement the desired sample rate
        return sw.fourier.design_cola_resampler(
            fs_base=capture.sample_rate,
            fs_target=capture.sample_rate,
            bw=capture.analysis_bandwidth,
            shift=False,
        )


def needs_resample(
    analysis_filter: sw.fourier.ResamplerDesign, capture: specs.SensorCapture
) -> bool:
    """determine whether host resampling will be needed to filter or resample"""
    if not capture.host_resample:
        return False

    is_resample = analysis_filter['nfft'] != analysis_filter['nfft_out']
    return is_resample and capture.host_resample


@specs.helpers.convert_capture_arg(specs.SensorCapture)
@sa.util.lru_cache(30000)
def get_fft_resample_pad(
    capture: specs.SensorCapture,
    setup: specs.Source,
    analysis: specs.AnalysisGroup | None = None,
) -> tuple[int, int]:
    # accommodate the large fft by padding to a fast size that includes at least lag_pad
    min_lag_pad = get_trigger_pad_size(setup, capture, analysis)
    design = design_resampler(capture, setup.master_clock_rate)
    analysis_size = round(capture.duration * design['fs_sdr'])

    # treat the block size as the minimum number of samples needed for the resampler
    # output to have an integral number of samples
    if np.isfinite(capture.analysis_bandwidth):
        filter_pad = get_filter_pad(capture)
        min_filter_blocks = sw.util.ceildiv(design['nfft'], filter_pad)
        block_size = design['nfft'] * min_filter_blocks
    else:
        block_size = design['nfft']
    block_count = analysis_size // block_size
    min_blocks = block_count + sw.util.ceildiv(min_lag_pad, block_size)

    # since design_capture_resampler gives us a nice fft size
    # for block_size, then if we make sure pad_blocks is also a nice fft size,
    # then the product (pad_blocks * block_size) will also be a product of small
    # primes
    pad_blocks = _get_next_fast_len(min_blocks + 1, array_backend=setup.array_backend)
    pad_end = pad_blocks * block_size - analysis_size
    assert pad_end % 2 == 0

    return (pad_end // 2, pad_end // 2)


@specs.helpers.convert_capture_arg(specs.SensorCapture)
@sa.util.lru_cache(30000)
def get_dsp_pad_size(
    capture: specs.SensorCapture,
    setup: specs.Source,
    analysis: specs.AnalysisGroup | None = None,
) -> tuple[int, int]:
    """returns the padding before and after a waveform to achieve an integral number of FFT windows"""

    from .. import resampling

    if resampling.USE_OARESAMPLE:
        min_lag_pad = get_trigger_pad_size(setup, capture, analysis)
        oa_pad_low, oa_pad_high = get_oaresample_pad(capture, setup.master_clock_rate)
        return (oa_pad_low, oa_pad_high + min_lag_pad)
    else:
        # this is removed before the FFT, so no need to micromanage its size
        fft_pad = get_fft_resample_pad(capture, setup, analysis)

        filter_pad = get_filter_pad(capture)
        assert fft_pad[0] > filter_pad and fft_pad[1] > filter_pad

        return (fft_pad[0], fft_pad[1])


@sa.util.lru_cache()
def get_signal_trigger_name(setup: specs.Source) -> str | None:
    if isinstance(setup.signal_trigger, specs.AnalysisGroup):
        analysis = setup.signal_trigger
        meas = {
            name: meas for name, meas in analysis.to_dict().items() if meas is not None
        }
        if len(meas) != 1:
            raise ValueError(
                'specify exactly one trigger for an explicit signal_trigger'
            )
        return list(meas.keys())[0]
    elif isinstance(setup.signal_trigger, str):
        return setup.signal_trigger
    else:
        return None


@specs.helpers.convert_capture_arg(specs.SensorCapture)
@sa.util.lru_cache()
def get_read_count(
    capture: specs.SensorCapture,
    setup: specs.Source,
    *,
    include_holdoff: bool = False,
    analysis: specs.AnalysisGroup | None = None,
) -> int:
    if sw.isroundmod(capture.duration * capture.sample_rate, 1):
        samples_out = round(capture.duration * capture.sample_rate)
    else:
        msg = f'duration must be an integer multiple of the sample period (1/{capture.sample_rate} s)'
        raise ValueError(msg)

    resampler_design = design_resampler(capture, setup.master_clock_rate)
    if capture.host_resample:
        sample_rate = resampler_design['fs']
    else:
        sample_rate = capture.sample_rate

    pad_size = sum(get_dsp_pad_size(capture, setup, analysis))
    if needs_resample(resampler_design, capture):
        nfft = resampler_design['nfft']
        min_samples_in = ceil(samples_out * nfft / resampler_design['nfft_out'])
        samples_in = min_samples_in + pad_size
    else:
        samples_in = round(capture.sample_rate * capture.duration) + pad_size

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
) -> 'tuple[np.ndarray, tuple[np.ndarray, list[np.ndarray]]]':
    """allocate a buffer of IQ return values.

    Returns:
        The buffer and the list of buffer references for streaming.
    """
    count = get_read_count(source.capture_spec, source.setup_spec, include_holdoff=True)

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
    dsp_pad_before: int = 0,
):

    sample_rate = design_resampler(capture_spec, source_spec.master_clock_rate)[
        'fs_sdr'
    ]
    min_holdoff = dsp_pad_before

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


def _get_next_fast_len(n, array_backend: specs.types.ArrayBackend) -> int:
    if array_backend == 'cupy':
        import cupyx.scipy.fft as fft  # type: ignore
    elif array_backend == 'numpy':
        import scipy.fft as fft
    else:
        raise TypeError(f'invalid array_backend {array_backend}')

    size = fft.next_fast_len(n)
    assert size is not None, ValueError('failed to determine fft size')
    return size


@sa.util.lru_cache()
def get_oaresample_pad(capture: specs.SensorCapture, master_clock_rate: float):
    resampler_design = design_resampler(capture, master_clock_rate)

    nfft = resampler_design['nfft']
    nfft_out = resampler_design.get('nfft_out', nfft)

    samples_out = round(capture.duration * capture.sample_rate)
    min_samples_in = ceil(samples_out * nfft / resampler_design['nfft_out'])

    # round up to an integral number of FFT windows
    samples_in = ceil(min_samples_in / nfft) * nfft + nfft

    noverlap_out = sw.fourier._ola_filter_parameters(
        samples_in,
        window=resampler_design['window'],
        nfft_out=nfft_out,
        nfft=nfft,
        extend=True,
    )[1]

    noverlap = ceil(noverlap_out * nfft / nfft_out)

    return (samples_in - min_samples_in) + noverlap + nfft // 2, noverlap


def cast_iq(spec: specs.Source, buffer: 'Array', acquired_count: int) -> 'Array':
    """cast the buffer to floating point, if necessary"""
    # array_namespace will categorize cupy pinned memory as numpy
    dtype_in = np.dtype(spec.transport_dtype)

    if spec.array_backend == 'cupy':
        xp = sa.util.cp
        assert xp is not None, ImportError('cupy is not installed')
        buffer = sa.util.pinned_array_as_cupy(buffer)
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


def get_filter_pad(capture: specs.SensorCapture):
    if isfinite(capture.analysis_bandwidth):
        return FILTER_SIZE // 2 + 1
    else:
        return 0


def get_trigger_pad_size(
    setup: specs.Source,
    capture: specs.SensorCapture,
    trigger_info: sa.Trigger | specs.AnalysisGroup | None = None,
) -> int:
    if isinstance(trigger_info, specs.AnalysisGroup):
        trigger = get_trigger_from_spec(setup, trigger_info)
    elif isinstance(trigger_info, sa.Trigger):
        trigger = trigger_info
    else:
        return 0

    if trigger is None:
        return 0

    mcr = setup.master_clock_rate
    max_lag = trigger.max_lag(capture)
    lag_pad = ceil(mcr * max_lag)

    return lag_pad


def is_reusable(
    c1: specs.SensorCapture | None, c2: specs.SensorCapture | None, mcr: float
):
    """return True if c2 is compatible with the raw and uncalibrated IQ acquired for c1"""

    if c1 is None or c2 is None:
        return False

    fsb1 = design_resampler(c1, mcr)['fs_sdr']
    fsb2 = design_resampler(c2, mcr)['fs_sdr']

    if fsb1 != fsb2:
        # the realized backend sample rates need to be the same
        return False

    downstream_kws = {
        'host_resample': False,
        'backend_sample_rate': None,
    }

    c1_compare = c1.replace(**downstream_kws)
    c2_compare = c2.replace(
        # ignore parameters that only affect downstream processing
        analysis_bandwidth=c1.analysis_bandwidth,
        sample_rate=c1.sample_rate,
        **downstream_kws,
    )

    return c1_compare == c2_compare
