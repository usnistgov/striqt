from __future__ import annotations
import functools
import logging
from math import ceil
import numbers
import types
import typing

import numpy as np

from .. import specs, util

from striqt.analysis import registry
from striqt.analysis.lib.util import pinned_array_as_cupy
from striqt.analysis.lib.specs import Analysis
from striqt.analysis.lib import register
from striqt.analysis.lib.dataarrays import AcquiredIQ


if typing.TYPE_CHECKING:
    import striqt.waveform as iqwaveform
    from striqt.waveform._typing import ArrayType
    from striqt.waveform.fourier import ResamplerDesign
    import pandas as pd
else:
    iqwaveform = util.lazy_import('striqt.waveform')
    pd = util.lazy_import('pandas')


OnOverflowType = (
    typing.Literal['ignore'] | typing.Literal['except'] | typing.Literal['log']
)


FILTER_SIZE = 4001
MIN_OARESAMPLE_FFT_SIZE = 4 * 4096 - 1
RESAMPLE_COLA_WINDOW = 'hamming'
FILTER_DOMAIN = 'time'


_TS = typing.TypeVar('_TS', bound=specs.RadioSetup)
_TC = typing.TypeVar('_TC', bound=specs.RadioCapture)


class ReceiveStreamError(IOError):
    pass


class _ReceiveBuffers:
    """remember unused samples from the previous IQ capture"""

    carryover_samples: 'np.ndarray | None'
    start_time_ns: int | None
    buffers: list = [None, None]
    hold_buffer_swap = False

    def __init__(self, source):
        self.radio = source
        self.clear()

    def apply(self, samples: np.ndarray) -> tuple[int | None, int]:
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

    def get_next(self, capture) -> tuple[np.ndarray, list[np.ndarray]]:
        """swap the buffers, and reallocate if needed"""
        if not self.hold_buffer_swap:
            self.buffers = [self.buffers[1], self.buffers[0]]
        self.buffers[0], ret = alloc_empty_iq(self.radio, capture, self.buffers[0])
        self.hold_buffer_swap = False
        return ret

    def stash_carryover(
        self,
        samples: 'np.ndarray',
        sample_start_ns,
        unused_sample_count: int,
        capture: specs.RadioCapture,
    ):
        """stash data needed to carry over extra samples into the next capture"""
        carryover_count = unused_sample_count
        self.carryover_samples = samples[:, -carryover_count:].copy()
        self.start_time_ns = sample_start_ns + round(1e9 * capture.duration)

    def clear(self):
        self.carryover_samples = None
        self.start_time_ns = None
        self.buffers = [None, None]

    def __del__(self):
        self.clear()


def _cast_iq(
    radio: SourceBase, buffer: 'ArrayType', acquired_count: int
) -> 'ArrayType':
    """cast the buffer to floating point, if necessary"""
    # array_namespace will categorize cupy pinned memory as numpy
    dtype_in = np.dtype(radio._setup.transport_dtype)

    if radio._setup.array_backend == 'cupy':
        import cupy as xp

        buffer = pinned_array_as_cupy(buffer)
    else:
        import numpy as xp

        buffer = xp.array(buffer)

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


class BaseSourceInfo(specs.SpecBase, kw_only=True, frozen=True, cache_hash=True):
    num_rx_ports: int

    def to_capture_cls(self, base_cls: type[_TC] = specs.RadioCapture) -> type[_TC]:
        return base_cls

    def to_setup_cls(self, base_cls: type[_TS] = specs.RadioSetup) -> type[_TS]:
        return base_cls


class HasSetupType(typing.Protocol[_TS]):
    _setup: _TS

    def __init__(
        self,
        setup: _TS,
        analysis: Analysis | None = None,
        **setup_kws: typing.Unpack[specs._RadioSetupKeywords],
    ): ...

    def get_setup_spec(self) -> _TS: ...

    def _connect(self, spec: _TS) -> None: ...

    def _apply_setup(self, spec: _TS) -> _TS | None: ...


class HasCaptureType(typing.Protocol[_TC]):
    _capture: typing.Optional[_TC]

    def arm(
        self,
        capture: _TC,
        *,
        force_time_sync: bool = False,
        **capture_kws: typing.Unpack[specs._RadioCaptureKeywords],
    ) -> _TC: ...

    def acquire(
        self,
        capture: _TC | None = None,
        next_capture: typing.Union[_TC, None] = None,
        correction: bool = True,
    ) -> AcquiredIQ: ...

    def get_capture_spec(self) -> _TC: ...

    def _prepare_capture(self, capture: _TC) -> _TC | None: ...


class SourceBase(HasSetupType[_TS], HasCaptureType[_TC]):
    _buffers: _ReceiveBuffers
    _is_open: bool = False

    def __init__(self, setup, *, analysis=None, **setup_kws):
        self._aligner: register.AlignmentCaller | None = None

        # if setup.driver is None:
        #     setup_kws['driver'] = self.__class__.__name__

        setup = setup.replace(**setup_kws)

        if setup.channel_sync_source is None:
            self._aligner = None
        elif analysis is None:
            name = register.get_channel_sync_source_measurement_name(
                setup.channel_sync_source, registry
            )
            raise ValueError(
                f'channel_sync_source {name!r} requires an analysis '
                f'specification for {setup.channel_sync_source!r}'
            )
        else:
            self._aligner = register.get_aligner(
                setup.channel_sync_source, analysis=analysis, registry=registry
            )

        self._setup = setup
        self._capture = None
        self._buffers = _ReceiveBuffers(self)

    @functools.cached_property
    def info(self) -> BaseSourceInfo:
        raise NotImplementedError

    @functools.cached_property
    def id(self) -> str:
        raise NotImplementedError

    @property
    def is_open(self) -> bool:
        return self._is_open

    def close(self):
        self._is_open = False
        self._buffers.clear()

    def __del__(self):
        self.close()

    def __enter__(self):
        if self.is_open:
            raise RuntimeError('cannot enter context for an open device')
        self.open()

    def __exit__(self, *exc_info):
        if self.is_open:
            self.close()

    def open(self):
        if self.is_open:
            return

        setup = self._setup
        self._connect(setup)
        self._setup = self._apply_setup(setup) or setup
        self._is_open = True
        return self._setup

    def get_setup_spec(self):
        """generate the currently armed capture configuration for the specified channel.

        If the truth of realized evaluates as False, only the requested value
        of backend_sample_rate is returned in the given radio capture.
        """

        if self._capture is None:
            raise RuntimeError('arm first to get capture spec')

        return self._setup

    @property
    def _resampler(self) -> ResamplerDesign:
        if self._capture is None:
            raise RuntimeError('arm before designing the resampler')
        return design_capture_resampler(self._setup.base_clock_rate, self._capture)

    @util.stopwatch('arm', 'source', threshold=10e-3)
    def arm(self, capture, *, force_time_sync: bool = False, **capture_kws):
        """stop the stream, apply a capture configuration, and start it"""
        assert self._buffers is not None

        if not self.is_open:
            raise RuntimeError('open the radio before arming')

        capture = capture.replace(**capture_kws)

        if not self._setup.gapless_repeats or capture != self._capture:
            self._buffers.clear()

        self._capture = self._prepare_capture(capture) or capture
        return self._capture

    @util.stopwatch('read_iq', 'source')
    def read_iq(self) -> 'tuple[np.ndarray, int|None]':
        """read IQ for the armed capture"""
        assert self._capture is not None, 'soapy source must be armed to read IQ'

        # the return buffer
        samples, stream_bufs = self._buffers.get_next(self._capture)

        # holdoff parameters, valid when we already have a clock reading
        dsp_pad_before, _ = _get_dsp_pad_size(self._setup, self._capture, self._aligner)

        # carryover from the previous acquisition
        missing_start_time = True
        start_ns, carryover_count = self._buffers.apply(samples)
        stream_time_ns = start_ns

        # the number of holdoff samples from the end of the holdoff period
        # to include with the returned waveform
        included_holdoff = dsp_pad_before

        fs = self._resampler['fs_sdr']

        # the number of valid samples to return per channel
        output_count = get_channel_read_buffer_count(self, include_holdoff=False)

        # the total number of samples to acquire per channel
        buffer_count = get_channel_read_buffer_count(self, include_holdoff=True)

        received_count = 0
        chunk_count = remaining = output_count - carryover_count

        while remaining > 0:
            if received_count > 0 or self._setup.gapless_repeats:
                on_overflow = 'except'
            else:
                on_overflow = 'ignore'

            request_count = min(chunk_count, remaining)

            if (received_count + request_count) > buffer_count:
                # this could happen if there is a slight mismatch between
                # the requested and realized sample rate
                break

            # Read the samples from the data buffer
            this_count, ret_time_ns = self._read_stream(
                stream_bufs,
                offset=carryover_count + received_count,
                count=request_count,
                timeout_sec=request_count / fs + 10e-3,
                on_overflow=on_overflow,
            )

            if (this_count + received_count) > buffer_count:
                # this should never happen
                raise MemoryError(
                    f'overfilled receive buffer by {(this_count + received_count) - samples.size}'
                )

            if stream_time_ns is None:
                # after the first stream read, subsequent reads are treated as
                # contiguous unless TimeoutError is raised
                stream_time_ns = ret_time_ns

            if missing_start_time:
                included_holdoff = find_trigger_holdoff(
                    self, stream_time_ns, dsp_pad_before=dsp_pad_before
                )
                remaining = remaining + included_holdoff - dsp_pad_before

                start_ns = stream_time_ns + round(included_holdoff * 1e9 / fs)
                missing_start_time = False

            remaining = remaining - this_count
            received_count += this_count

        samples = samples.view('complex64')
        sample_offs = included_holdoff - dsp_pad_before
        sample_span = slice(sample_offs, sample_offs + output_count)

        unused_count = output_count - round(self._capture.duration * fs)
        self._buffers.stash_carryover(
            samples[:, sample_span],
            start_ns,
            unused_sample_count=unused_count,
            capture=self._capture,
        )

        # it seems to be important to convert to cupy here in order
        # to get a full view of the underlying pinned memory. cuda
        # memory corruption has been observed when waiting until after
        samples = _cast_iq(self, samples, buffer_count)

        return samples[:, sample_span], start_ns

    @util.stopwatch('acquire', 'source')
    def acquire(self, capture=None, next_capture=None, correction=True):
        """arm a capture and enable the channel (if necessary), read the resulting IQ waveform.

        Optionally, calibration corrections can be applied, and the radio can be left ready for the next capture.
        """
        from .. import iq_corrections

        if capture is None:
            capture = self.get_capture_spec()
        else:
            capture = self.arm(capture)

        iq, time_ns = self.read_iq()

        if next_capture is not None and capture != next_capture:
            self.arm(next_capture)

        if correction:
            with util.stopwatch(
                'resample and calibrate', 'analysis', threshold=capture.duration / 2
            ):
                iq = iq_corrections.resampling_correction(
                    iq, capture, self, overwrite_x=True
                )
        else:
            iq = AcquiredIQ(raw=iq, aligned=None)

        if time_ns is None:
            ts = None
        else:
            ts = pd.Timestamp(time_ns, unit='ns')

        actual_capture = capture.replace(
            start_time=ts, backend_sample_rate=self._resampler['fs_sdr']
        )

        return AcquiredIQ(iq.raw, iq.aligned, actual_capture)

    def _read_stream(
        self,
        buffers,
        offset,
        count,
        timeout_sec,
        *,
        on_overflow: OnOverflowType = 'except',
    ) -> tuple[int, int]:
        """to be implemented in subclasses"""
        raise NotImplementedError

    def get_capture_spec(self) -> _TC:
        """generate the currently armed capture configuration for the specified channel.

        If the truth of realized evaluates as False, only the requested value
        of backend_sample_rate is returned in the given radio capture.
        """

        if self._capture is None:
            raise RuntimeError('arm first to get capture spec')

        return self._capture

    def get_array_namespace(self: SourceBase) -> types.ModuleType:
        if self._setup.array_backend == 'cupy':
            import cupy

            return cupy
        elif self._setup.array_backend == 'numpy':
            import numpy

            return numpy
        else:
            raise TypeError('invalid array_backend argument')


def assert_open(radio: SourceBase):
    if not radio.is_open:
        raise RuntimeError('radio is not open')


def assert_armed(radio: SourceBase):
    assert_open(radio)
    if radio._capture is None:
        raise RuntimeError('radio is not armed')


def find_trigger_holdoff(
    radio: SourceBase, start_time_ns: int, dsp_pad_before: int = 0
):
    sample_rate = radio._resampler['fs_sdr']
    min_holdoff = dsp_pad_before

    # transient holdoff if we've rearmed as indicated by the presence of carryover samples
    if radio._buffers.start_time_ns is None:
        min_holdoff = min_holdoff + round(
            radio._setup._transient_holdoff_time * sample_rate
        )

    periodic_trigger = radio._setup.periodic_trigger
    if periodic_trigger in (0, None):
        return min_holdoff

    periodic_trigger_ns = round(periodic_trigger * 1e9)

    # float rounding errors cause problems here; evaluate based on the 1-ns resolution
    excess_time_ns = start_time_ns % periodic_trigger_ns
    holdoff_ns = (periodic_trigger_ns - excess_time_ns) % periodic_trigger_ns
    holdoff = round(holdoff_ns / 1e9 * sample_rate)

    if holdoff < min_holdoff:
        periodic_trigger_samples = round(periodic_trigger * sample_rate)
        holdoff += (
            ceil(min_holdoff / periodic_trigger_samples) * periodic_trigger_samples
        )

    return holdoff


@util.lru_cache(30000)
def _design_capture_resampler(
    base_clock_rate: float,
    capture: specs.WaveformCapture,
    bw_lo=0.25e6,
    min_oversampling=1.1,
    window=RESAMPLE_COLA_WINDOW,
    min_fft_size=MIN_OARESAMPLE_FFT_SIZE,
) -> ResamplerDesign:
    """design a filter specified by the capture for a radio with the specified MCR.

    For the return value, see `striqt.waveform.fourier.design_cola_resampler`
    """
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

    if capture.host_resample:
        # use GPU DSP to resample from integer divisor of the MCR
        # fs_sdr, lo_offset, kws  = iqwaveform.fourier.design_cola_resampler(
        design = iqwaveform.fourier.design_cola_resampler(
            fs_base=base_clock_rate,
            fs_target=capture.sample_rate,
            bw=capture.analysis_bandwidth,
            bw_lo=bw_lo,
            shift=lo_shift,
            min_fft_size=min_fft_size,
            min_oversampling=min_oversampling,
            window=window,
            fs_sdr=capture.backend_sample_rate,
        )

        design['window'] = window

        return design

    elif lo_shift:
        raise ValueError('lo_shift requires host_resample=True')
    elif base_clock_rate < capture.sample_rate:
        raise ValueError(
            f'upsampling above {base_clock_rate / 1e6:f} MHz requires host_resample=True'
        )
    else:
        # use the SDR firmware to implement the desired sample rate
        return iqwaveform.fourier.design_cola_resampler(
            fs_base=capture.sample_rate,
            fs_target=capture.sample_rate,
            bw=capture.analysis_bandwidth,
            shift=False,
        )


@functools.wraps(_design_capture_resampler)
def design_capture_resampler(
    base_clock_rate, capture: specs.WaveformCapture, *args, **kws
) -> ResamplerDesign:
    # cast the struct in case it's a subclass
    fixed_capture = specs.WaveformCapture.fromspec(capture)
    kws.setdefault('window', RESAMPLE_COLA_WINDOW)

    from .. import iq_corrections

    if iq_corrections.USE_OARESAMPLE:
        min_fft_size = MIN_OARESAMPLE_FFT_SIZE
    else:
        # this could probably be set to 1?
        min_fft_size = 256

    return _design_capture_resampler(
        base_clock_rate,
        fixed_capture,
        min_fft_size=min_fft_size,
        *args,
        **kws,
    )


def needs_resample(
    analysis_filter: ResamplerDesign, capture: specs.RadioCapture
) -> bool:
    """determine whether an STFT will be needed to filter or resample"""

    is_resample = analysis_filter['nfft'] != analysis_filter['nfft_out']
    return is_resample and capture.host_resample


def _get_filter_pad(capture: specs.RadioCapture):
    if np.isfinite(capture.analysis_bandwidth):
        return FILTER_SIZE // 2 + 1
    else:
        return 0


@util.lru_cache(30000)
def _get_dsp_pad_size(
    setup: specs.RadioSetup,
    capture: specs.RadioCapture,
    aligner: register.AlignmentCaller | None = None,
) -> tuple[int, int]:
    """returns the padding before and after a waveform to achieve an integral number of FFT windows"""

    from .. import iq_corrections

    min_lag_pad = _get_aligner_pad_size(setup.base_clock_rate, capture, aligner)

    if iq_corrections.USE_OARESAMPLE:
        oa_pad_low, oa_pad_high = _get_oaresample_pad(setup.base_clock_rate, capture)
        return (oa_pad_low, oa_pad_high + min_lag_pad)
    else:
        # this is removed before the FFT, so no need to micromanage its size
        filter_pad = _get_filter_pad(capture)

        # accommodate the large fft by padding to a fast size that includes at least lag_pad
        design = design_capture_resampler(setup.base_clock_rate, capture)
        analysis_size = round(capture.duration * design['fs_sdr'])

        # treat the block size as the minimum number of samples needed for the resampler
        # output to have an integral number of samples
        block_size = design['nfft']
        block_count = analysis_size // block_size
        min_blocks = block_count + iqwaveform.util.ceildiv(min_lag_pad, block_size)

        # since design_capture_resampler gives us a nice fft size
        # for block_size, then if we make sure pad_blocks is also a nice fft size,
        # then the product (pad_blocks * block_size) will also be a product of small
        # primes
        pad_blocks = _get_next_fast_len(min_blocks)
        pad_end = pad_blocks * block_size - analysis_size
        return (filter_pad, pad_end)


def _get_aligner_pad_size(
    base_clock_rate: float,
    capture: specs.RadioCapture,
    aligner: register.AlignmentCaller | None = None,
) -> int:
    if aligner is None:
        return 0

    max_lag = aligner.max_lag(capture)
    lag_pad = ceil(base_clock_rate * max_lag)

    return lag_pad


def _get_next_fast_len(n):
    try:
        from cupyx import scipy
    except ModuleNotFoundError:
        import scipy

    return scipy.fft.next_fast_len(n)


def _get_oaresample_pad(base_clock_rate: float, capture: specs.RadioCapture):
    resampler_design = design_capture_resampler(base_clock_rate, capture)

    nfft = resampler_design['nfft']
    nfft_out = resampler_design.get('nfft_out', nfft)

    samples_out = round(capture.duration * capture.sample_rate)
    min_samples_in = ceil(samples_out * nfft / resampler_design['nfft_out'])

    # round up to an integral number of FFT windows
    samples_in = ceil(min_samples_in / nfft) * nfft + nfft

    noverlap_out = iqwaveform.fourier._ola_filter_parameters(
        samples_in,
        window=resampler_design['window'],
        nfft_out=nfft_out,
        nfft=nfft,
        extend=True,
    )[1]

    noverlap = ceil(noverlap_out * nfft / nfft_out)

    return (samples_in - min_samples_in) + noverlap + nfft // 2, noverlap


@util.lru_cache(30000)
def _cached_channel_read_buffer_count(
    setup: specs.RadioSetup,
    capture: specs.RadioCapture,
    *,
    include_holdoff: bool = False,
    aligner: register.AlignmentCaller | None = None,
) -> int:
    if striqt.waveform.power_analysis.isroundmod(
        capture.duration * capture.sample_rate, 1
    ):
        samples_out = round(capture.duration * capture.sample_rate)
    else:
        msg = f'duration must be an integer multiple of the sample period (1/{capture.sample_rate} s)'
        raise ValueError(msg)

    resampler_design = design_capture_resampler(setup.base_clock_rate, capture)
    if capture.host_resample:
        sample_rate = resampler_design['fs']
    else:
        sample_rate = capture.sample_rate

    pad_size = sum(_get_dsp_pad_size(setup, capture, aligner))
    if needs_resample(resampler_design, capture):
        nfft = resampler_design['nfft']
        min_samples_in = ceil(samples_out * nfft / resampler_design['nfft_out'])
        samples_in = min_samples_in + pad_size
    else:
        samples_in = round(capture.sample_rate * capture.duration) + pad_size

    if include_holdoff:
        # pad the buffer for triggering and transient holdoff
        extra_time = (setup.transient_holdoff_time or 0) + 2 * (
            setup.periodic_trigger or 0
        )
        samples_in += ceil(sample_rate * extra_time)

    return samples_in


def get_channel_read_buffer_count(source: SourceBase, include_holdoff=False) -> int:
    assert source._capture is not None

    return _cached_channel_read_buffer_count(
        setup=source._setup,
        capture=source._capture,
        include_holdoff=include_holdoff,
        aligner=source._aligner,
    )


@util.stopwatch(
    'allocate acquisition buffer', 'source', threshold=5e-3, logger_level=logging.DEBUG
)
def alloc_empty_iq(
    radio: SourceBase,
    capture: specs.RadioCapture,
    prior: typing.Optional[np.ndarray] = None,
) -> tuple[np.ndarray, tuple[np.ndarray, list[np.ndarray]]]:
    """allocate a buffer of IQ return values.

    Returns:
        The buffer and the list of buffer references for streaming.
    """
    count = get_channel_read_buffer_count(radio, include_holdoff=True)

    if radio._setup.array_backend == 'cupy':
        try:
            util.configure_cupy()
            from cupyx import empty_pinned as empty
        except ModuleNotFoundError as ex:
            raise RuntimeError(
                'could not import the configured array backend, "cupy"'
            ) from ex
    else:
        empty = np.empty

    buf_dtype = np.dtype(radio._setup.transport_dtype)

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
    # radio._setup.stream_all_rx_ports
    if radio._setup.stream_all_rx_ports and len(ports) != radio.info.num_rx_ports:
        if radio._setup.transport_dtype == 'complex64':
            # a throwaway buffer for samples that won't be returned
            extra_count = count
        else:
            extra_count = 2 * count

        extra = empty(extra_count, dtype=buf_dtype)
    else:
        extra = None

    extra = typing.cast(np.ndarray, extra)

    buffers = []
    i = 0
    for channel in range(radio.info.num_rx_ports):
        if channel in ports:
            buffers.append(typing.cast(np.ndarray, samples[i].view(buf_dtype)))
            i += 1
        elif radio._setup.stream_all_rx_ports:
            assert extra is not None
            buffers.append(extra)

    return all_samples, (samples, buffers)


def _list_radio_classes(
    subclass: type[SourceBase] = SourceBase,
) -> dict[str, type[SourceBase]]:
    """returns a list of radio subclasses that have been imported"""

    clsmap = {c.__name__: c for c in subclass.__subclasses__()}

    for subcls in list(clsmap.values()):
        clsmap.update(_list_radio_classes(subcls))

    clsmap = {name: cls for name, cls in clsmap.items() if not name.startswith('_')}

    return clsmap


def find_radio_cls_helper(
    name: str, parent_cls: type[SourceBase] = SourceBase
) -> type[SourceBase]:
    """returns a list of radio subclasses that have been imported"""

    mapping = _list_radio_classes(parent_cls)

    if name in mapping:
        return mapping[name]
    else:
        raise AttributeError(
            f'invalid driver {repr(name)}. valid names: {tuple(mapping.keys())}'
        )
