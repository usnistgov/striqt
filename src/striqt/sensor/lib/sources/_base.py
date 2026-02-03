from __future__ import annotations as __

import dataclasses
import functools
import logging
import types
import typing
from collections import defaultdict
from math import ceil
from threading import Event
from typing_extensions import Self, Unpack, ParamSpec

import striqt.analysis as sa
import striqt.waveform as sw

from ... import specs
from .. import util

if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from striqt.waveform._typing import ArrayType

else:
    pd = util.lazy_import('pandas')
    np = util.lazy_import('numpy')


OnOverflowType = typing.Literal['ignore', 'except', 'log']


FILTER_SIZE = 4001
MIN_OARESAMPLE_FFT_SIZE = 4 * 4096 - 1
RESAMPLE_COLA_WINDOW = 'hamming'
FILTER_DOMAIN = 'time'

_TS = typing.TypeVar('_TS', bound=specs.Source)
_TC = typing.TypeVar('_TC', bound=specs.SensorCapture)
_PS = ParamSpec('_PS')
_PC = ParamSpec('_PC')
_TB = typing.TypeVar('_TB', bound='specs.SpecBase')
_T = typing.TypeVar('_T', bound='SourceBase')


@dataclasses.dataclass
class AcquiredIQ(sa.dataarrays.AcquiredIQ):
    """extra metadata needed for downstream analysis"""

    info: specs.SourceCoordinates
    extra_data: dict[str, typing.Any]
    alias_func: specs.helpers.PathAliasFormatter | None
    source_spec: specs.Source
    resampler: sw.fourier.ResamplerDesign
    trigger: sa.Trigger | None
    analysis: specs.AnalysisGroup | None = None
    voltage_scale: ArrayType | float = 1


class ReceiveStreamError(IOError):
    pass


class _ReceiveBuffers:
    """remember unused samples from the previous IQ capture"""

    carryover_samples: 'np.ndarray | None'
    start_time_ns: int | None
    buffers: list = [None, None]
    _hold_buffer_swap = False

    def __init__(self, source):
        self.radio = source
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
        self.buffers[0], ret = alloc_empty_iq(self.radio, capture, self.buffers[0])
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


def _get_dtype_scale(transport_dtype: specs.types.TransportDType) -> float:
    """compute the scaling factor needed to scale each of N ports of an IQ waveform

    Returns:
        an array of type `xp.ndarray` with shape (N,)
    """

    transport_dtype = transport_dtype
    if transport_dtype == 'int16':
        return 1.0 / float(np.iinfo(transport_dtype).max)
    else:
        return 1.0


def _cast_iq(
    source: SourceBase, buffer: 'ArrayType', acquired_count: int
) -> 'ArrayType':
    """cast the buffer to floating point, if necessary"""
    # array_namespace will categorize cupy pinned memory as numpy
    dtype_in = np.dtype(source.setup_spec.transport_dtype)

    if source.setup_spec.array_backend == 'cupy':
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


_source_id_map: dict[specs.Source, SourceBase | Event] = defaultdict(Event)


def get_source_id(spec: specs.Source, timeout=0.2) -> str:
    """lookup a source ID from a source specification.

    This assumes that a source is either instantiated, or will be
    within `timeout` seconds.
    """
    obj = _source_id_map[spec]

    if isinstance(obj, Event):
        if not obj.wait(timeout=timeout):
            raise TimeoutError('timeout while waiting for a source ID')
        source = _source_id_map[spec]
        assert isinstance(source, SourceBase)
    else:
        source = obj

    # this triggers a property access that may have its own
    # blocking wait
    return source.id


def _map_source(spec: specs.Source, source: SourceBase):
    maybe_event = _source_id_map[spec]
    _source_id_map[spec] = source

    if isinstance(maybe_event, Event):
        maybe_event.set()


class HasSetupType(typing.Protocol[_TS, _PS]):
    __setup__: _TS

    def __init__(
        self,
        _setup: _TS | None = None,
        /,
        reuse_iq=False,
        *args: _PS.args,
        **kwargs: _PS.kwargs,
    ): ...

    @classmethod
    def from_spec(
        cls,
        spec: _TS,
        *,
        captures: tuple[_TC, ...] | None = None,
        loops: tuple[specs.LoopSpec, ...] | None = None,
        reuse_iq: bool = False,
    ) -> Self: ...

    def _connect(self, spec: _TS) -> None: ...

    def _apply_setup(
        self,
        spec: _TS,
        *,
        captures: tuple[_TC, ...] | None = None,
        loops: tuple[specs.LoopSpec, ...] | None = None,
    ) -> None: ...


class HasCaptureType(typing.Protocol[_TC, _PC]):
    _capture: typing.Optional[_TC]

    def arm(self, *args: _PC.args, **kwargs: _PC.kwargs): ...

    def arm_spec(self, spec: _TC): ...

    def acquire(
        self,
        *,
        correction: bool = True,
        alias_func: specs.helpers.PathAliasFormatter | None = None,
    ) -> AcquiredIQ: ...

    @property
    def capture_spec(self) -> _TC: ...

    def _prepare_capture(self, capture: _TC) -> _TC | None: ...

    def get_resampler(
        self, capture: _TC | None = None
    ) -> sw.fourier.ResamplerDesign: ...


@typing.overload
def get_trigger_from_spec(setup: specs.Source, analysis: None = None) -> None: ...


@typing.overload
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


@dataclasses.dataclass(frozen=True)
class Schema(typing.Generic[_TS, _TC]):
    source: type[_TS]
    capture: type[_TC]


def bind_schema_types(
    source: type[_TS], capture: type[_TC]
) -> typing.Callable[[type[_T]], type[_T]]:
    """set the default to a SourceBase subclass"""

    def decorator(cls: type[_T]) -> type[_T]:
        cls.__bindings__ = Schema(source=source, capture=capture)
        return cls

    return decorator


def get_bound_spec(spec: specs.SpecBase | None, cls: type[_TB] | None, **kws) -> _TB:
    if isinstance(spec, specs.SpecBase):
        if cls is not None:
            spec = typing.cast(_TB, cls.from_spec(spec))
        capture = spec.replace(**kws)
    elif spec is not None:
        raise TypeError(f'spec must be an instance of {cls.__qualname__!r} or None')
    elif cls is None:
        raise TypeError('an explicit argument of type specs.Capture is required')
    else:
        capture = cls(**kws)

    return typing.cast(_TB, capture)


def get_array_namespace(array_backend: specs.types.ArrayBackend) -> types.ModuleType:
    if array_backend == 'cupy':
        return sa.util.cp
    elif array_backend == 'numpy':
        return np
    else:
        raise TypeError('invalid array_backend argument')


class SourceBase(
    typing.Generic[_TS, _TC, _PS, _PC], HasSetupType[_TS, _PS], HasCaptureType[_TC, _PC]
):
    __bindings__: typing.ClassVar[Schema | None] = None

    _buffers: _ReceiveBuffers
    _is_open: bool | Event = False
    _timeout: float = 10
    _sweep_time: 'pd.Timestamp | None' = None

    def __init__(self, reuse_iq=False, *args: _PS.args, **kwargs: _PS.kwargs):
        open_event = self._is_open = Event()  # first, to serve other threads

        # back door from .from_spec
        _extra_specs = typing.cast(dict, kwargs.pop('__specs', {}))
        _spec = _extra_specs.pop('source', None)

        if _spec is not None:
            _spec = typing.cast(_TS, _spec)

        if self.__bindings__ is None:
            spec_cls = None
        else:
            spec_cls = typing.cast(type[_TS], self.__bindings__.source)

        _spec = get_bound_spec(_spec, spec_cls, **kwargs)

        _map_source(_spec, self)

        self.__setup__ = _spec
        self._capture = None
        self._buffers = _ReceiveBuffers(self)
        self._prev_iq: AcquiredIQ | None = None
        self._reuse_iq = reuse_iq

        try:
            self._connect(_spec)
        except:
            self._is_open = False
            raise
        else:
            self._is_open = True
        finally:
            open_event.set()

        if _spec.array_backend == 'cupy':
            util.safe_import('cupy')
            sa.util.configure_cupy()

        self._apply_setup(_spec, **_extra_specs)

    @classmethod
    def from_spec(cls, spec: _TS, *, captures=None, loops=None, reuse_iq=False) -> Self:
        kwargs = spec.to_dict()
        kwargs['__specs'] = {'source': spec, 'captures': captures, 'loops': loops}

        if captures is not None and len(captures) > 0 and cls.__bindings__ is None:
            raise TypeError('can only hint captures for source class bindings')

        return cls(reuse_iq=reuse_iq, **kwargs)  # type: ignore

    @functools.cached_property
    def info(self) -> specs.BaseSourceInfo:
        raise NotImplementedError

    @functools.cached_property
    def id(self) -> str:
        raise NotImplementedError

    def is_open(self, wait=True) -> bool:
        obj = self._is_open
        if isinstance(obj, Event):
            if wait:
                obj.wait(self._timeout + 0.2)
                return typing.cast(bool, self._is_open)
            else:
                return False
        else:
            return obj

    def close(self):
        self._is_open = False
        if hasattr(self, '_buffers'):
            self._buffers.clear()

    def __del__(self):
        self.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info):
        if self.is_open():
            self.close()

    @functools.cached_property
    def setup_spec(self) -> _TS:
        return self.__setup__

    @sa.util.stopwatch('arm', 'source', threshold=10e-3)
    def arm(self, *args, **kwargs):
        """stop the stream, apply a capture configuration, and start it"""
        assert self._buffers is not None

        if self.__bindings__ is not None:
            capture_cls = self.__bindings__.capture
        elif self._capture is not None:
            capture_cls = type(self._capture)
        else:
            raise TypeError('no capture bindings were supplied')

        capture = get_bound_spec(None, capture_cls, **kwargs)

        return self.arm_spec(capture)

    def arm_spec(self, spec: _TC):
        if not self.is_open():
            raise RuntimeError('open the radio before arming')

        if self._capture is not None:
            mcr = self.setup_spec.master_clock_rate
            if self._reuse_iq and _is_reusable(self.capture_spec, spec, mcr):
                pass
            else:
                self._prev_iq = None

        if spec == self._capture and self._capture is not None:
            return

        if not self.setup_spec.gapless or spec != self._capture:
            self._buffers.clear()

        self._capture = self._prepare_capture(spec) or spec

    def read_iq(
        self, analysis: specs.AnalysisGroup | None = None
    ) -> 'tuple[ArrayType, int|None]':
        """read IQ for the armed capture"""
        assert self._capture is not None, 'soapy source must be armed to read IQ'

        # the return buffer
        samples, stream_bufs = self._buffers.get_next(self._capture)

        # holdoff parameters, valid when we already have a clock reading
        dsp_pad_before, _ = get_dsp_pad_size(
            self.setup_spec, self._capture, analysis=analysis
        )

        # carryover from the previous acquisition
        missing_start_time = True
        start_ns, carryover_count = self._buffers.apply(samples)
        stream_time_ns = start_ns

        # the number of holdoff samples from the end of the holdoff period
        # to include with the returned waveform
        included_holdoff = dsp_pad_before

        fs = self.get_resampler()['fs_sdr']

        # the number of valid samples to return per channel
        output_count = get_channel_read_buffer_count(self, include_holdoff=False)

        # the total number of samples to acquire per channel
        buffer_count = get_channel_read_buffer_count(self, include_holdoff=True)

        received_count = 0
        chunk_count = remaining = output_count - carryover_count

        while remaining > 0:
            if received_count > 0 or self.setup_spec.gapless:
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

    @sa.util.stopwatch('acquire', 'source')
    def acquire(self, *, analysis=None, correction=True, alias_func=None) -> AcquiredIQ:
        """arm a capture and enable the channel (if necessary), read the resulting IQ waveform.

        Optionally, calibration corrections can be applied, and the radio can be left ready for the next capture.
        """
        from .. import resampling

        self.capture_spec  # ensure we are armed

        trigger = get_trigger_from_spec(self.setup_spec, analysis)

        if self._prev_iq is None:
            samples, time_ns = self.read_iq(analysis)
            iq = self._package_acquisition(
                samples,
                time_ns,
                analysis=analysis,
                correction=correction,
                alias_func=alias_func,
            )

        else:
            iq = dataclasses.replace(
                self._prev_iq,
                capture=self.capture_spec,
                info=self._prev_iq.info.replace(start_time=None),
                trigger=trigger,
                analysis=analysis,
            )

        if self._reuse_iq:
            self._prev_iq = iq

        if not correction:
            return iq
        else:
            tmin = self.capture_spec.duration / 2
            with sa.util.stopwatch('resampling filter', threshold=tmin):
                return resampling.resampling_correction(iq, overwrite_x=True)

    def _package_acquisition(
        self,
        samples: ArrayType,
        time_ns: int | None,
        *,
        analysis=None,
        correction=True,
        alias_func: specs.helpers.PathAliasFormatter | None = None,
    ) -> AcquiredIQ:
        info = specs.SourceCoordinates(source_id=self.id)

        trigger = get_trigger_from_spec(self.setup_spec, analysis)

        return AcquiredIQ(
            raw=samples,
            aligned=None,
            capture=self.capture_spec,
            info=info,
            extra_data={},
            alias_func=alias_func,
            source_spec=self.setup_spec,
            resampler=self.get_resampler(),
            trigger=trigger,
            analysis=analysis,
            voltage_scale=_get_dtype_scale(self.setup_spec.transport_dtype),
        )

    def _read_stream(
        self,
        buffers,
        offset,
        count,
        timeout_sec,
        *,
        on_overflow: specs.types.OnOverflowType = 'except',
    ) -> tuple[int, int]:
        """to be implemented in subclasses"""
        raise NotImplementedError

    @property
    def capture_spec(self) -> _TC:
        """generate the currently armed capture configuration for the specified channel.

        If the truth of realized evaluates as False, only the requested value
        of backend_sample_rate is returned in the given radio capture.
        """

        if self._capture is None:
            raise AttributeError('arm to set the capture spec')

        return self._capture

    def get_resampler(self, capture=None) -> sw.fourier.ResamplerDesign:
        if capture is None:
            capture = self.capture_spec

        return design_capture_resampler(self.setup_spec.master_clock_rate, capture)


class VirtualSourceBase(SourceBase[_TS, _TC, _PS, _PC]):
    _samples_elapsed = 0

    def reset_sample_counter(self, value=0):
        self._sync_time_source()
        self._samples_elapsed = value
        self._sample_start_index = value

    def _apply_setup(self, spec, *, captures=None, loops=None):
        self.reset_sample_counter()

    def _prepare_capture(self, capture):
        self.reset_sample_counter()

    def _read_stream(
        self,
        buffers,
        offset,
        count,
        timeout_sec=None,
        *,
        on_overflow: OnOverflowType = 'except',
    ):
        assert self._capture is not None

        if not isinstance(self._capture.port, tuple):
            ports = (self._capture.port,)
        else:
            ports = self._capture.port

        for port, buf in zip(ports, buffers):
            values = self.get_waveform(
                count,
                self._samples_elapsed,
                port=port,
                xp=getattr(self, 'xp', np),
            )
            buf[offset : (offset + count)] = values

        fs = float(self.get_resampler()['fs_sdr'])
        sample_period_ns = 1_000_000_000 / fs
        timestamp_ns = self._sync_time_ns + self._samples_elapsed * sample_period_ns

        self._samples_elapsed += count

        return count, round(timestamp_ns)

    def get_waveform(
        self, count: int, offset: int, *, port: int = 0, xp, dtype='complex64'
    ) -> ArrayType:
        raise NotImplementedError

    def _sync_time_source(self):
        self._sync_time_ns = round(1_000_000_000 * self._samples_elapsed)


def find_trigger_holdoff(
    source: SourceBase, start_time_ns: int, dsp_pad_before: int = 0
):
    sample_rate = source.get_resampler()['fs_sdr']
    min_holdoff = dsp_pad_before

    # transient holdoff if we've rearmed as indicated by the presence of carryover samples
    if source._buffers.start_time_ns is None:
        min_holdoff = min_holdoff + round(
            source.setup_spec.transient_holdoff_time * sample_rate
        )

    trigger_strobe = source.setup_spec.trigger_strobe
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


class _ResamplerKws(typing.TypedDict, total=False):
    bw_lo: float
    min_oversampling: float
    window: str
    min_fft_size: int


@sa.util.lru_cache(30000)
def _design_capture_resampler(
    master_clock_rate: float,
    capture: specs.SensorCapture,
    backend_sample_rate: float | None = None,
    **kwargs: Unpack[_ResamplerKws],
) -> sw.fourier.ResamplerDesign:
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


@functools.wraps(_design_capture_resampler)
def design_capture_resampler(
    master_clock_rate: float,
    capture: specs.SensorCapture,
    backend_sample_rate: float | None = None,
    **kws: Unpack[_ResamplerKws],
) -> sw.fourier.ResamplerDesign:
    # cast the struct in case it's a subclass
    fixed_capture = specs.SensorCapture.from_spec(capture)
    kws = _ResamplerKws(
        bw_lo=0.25e6,
        min_oversampling=1.1,
        window=RESAMPLE_COLA_WINDOW,
        min_fft_size=MIN_OARESAMPLE_FFT_SIZE,
    )

    from .. import resampling

    if resampling.USE_OARESAMPLE:
        kws['min_fft_size'] = MIN_OARESAMPLE_FFT_SIZE
    else:
        # this could probably be set to 1?
        kws['min_fft_size'] = 256

    return _design_capture_resampler(
        master_clock_rate,
        fixed_capture,
        backend_sample_rate=backend_sample_rate,
        **kws,
    )


def needs_resample(
    analysis_filter: sw.fourier.ResamplerDesign, capture: specs.SensorCapture
) -> bool:
    """determine whether an STFT will be needed to filter or resample"""

    is_resample = analysis_filter['nfft'] != analysis_filter['nfft_out']
    return is_resample and capture.host_resample


def _get_filter_pad(capture: specs.SensorCapture):
    if np.isfinite(capture.analysis_bandwidth):
        return FILTER_SIZE // 2 + 1
    else:
        return 0


@sa.util.lru_cache(30000)
def _get_fft_resample_pad(
    setup: specs.Source,
    capture: specs.SensorCapture,
    analysis: specs.AnalysisGroup | None = None,
) -> tuple[int, int]:
    # accommodate the large fft by padding to a fast size that includes at least lag_pad
    min_lag_pad = _get_trigger_pad_size(setup, capture, analysis)
    design = design_capture_resampler(setup.master_clock_rate, capture)
    analysis_size = round(capture.duration * design['fs_sdr'])

    # treat the block size as the minimum number of samples needed for the resampler
    # output to have an integral number of samples
    if np.isfinite(capture.analysis_bandwidth):
        filter_pad = _get_filter_pad(capture)
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


def get_fft_resample_pad(
    setup: specs.Source,
    capture: specs.SensorCapture,
    analysis: specs.AnalysisGroup | None = None,
) -> tuple[int, int]:
    capture = specs.SensorCapture.from_spec(capture)
    return _get_fft_resample_pad(setup, capture, analysis)


@sa.util.lru_cache(30000)
def _get_dsp_pad_size(
    setup: specs.Source,
    capture: specs.SensorCapture,
    analysis: specs.AnalysisGroup | None = None,
) -> tuple[int, int]:
    """returns the padding before and after a waveform to achieve an integral number of FFT windows"""

    from .. import resampling

    if resampling.USE_OARESAMPLE:
        min_lag_pad = _get_trigger_pad_size(setup, capture, analysis)
        oa_pad_low, oa_pad_high = _get_oaresample_pad(setup.master_clock_rate, capture)
        return (oa_pad_low, oa_pad_high + min_lag_pad)
    else:
        # this is removed before the FFT, so no need to micromanage its size
        fft_pad = _get_fft_resample_pad(setup, capture, analysis)

        filter_pad = _get_filter_pad(capture)
        assert fft_pad[0] > 2 * filter_pad

        return (fft_pad[0], fft_pad[1])


def get_dsp_pad_size(
    setup: specs.Source,
    capture: specs.SensorCapture,
    analysis: specs.AnalysisGroup | None = None,
) -> tuple[int, int]:
    capture = specs.SensorCapture.from_spec(capture)
    return _get_dsp_pad_size(setup, capture, analysis)


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


def _get_trigger_pad_size(
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
def _get_oaresample_pad(master_clock_rate: float, capture: specs.SensorCapture):
    resampler_design = design_capture_resampler(master_clock_rate, capture)

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


@sa.util.lru_cache()
def _cached_channel_read_buffer_count(
    setup: specs.Source,
    capture: specs.SensorCapture,
    *,
    include_holdoff: bool = False,
    analysis: specs.AnalysisGroup | None = None,
) -> int:
    if sw.isroundmod(capture.duration * capture.sample_rate, 1):
        samples_out = round(capture.duration * capture.sample_rate)
    else:
        msg = f'duration must be an integer multiple of the sample period (1/{capture.sample_rate} s)'
        raise ValueError(msg)

    resampler_design = design_capture_resampler(setup.master_clock_rate, capture)
    if capture.host_resample:
        sample_rate = resampler_design['fs']
    else:
        sample_rate = capture.sample_rate

    pad_size = sum(_get_dsp_pad_size(setup, capture, analysis))
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


def get_channel_read_buffer_count(
    source: SourceBase,
    analysis: specs.AnalysisGroup | None = None,
    include_holdoff=False,
) -> int:
    assert source._capture is not None

    capture = specs.SensorCapture.from_spec(source._capture)

    return _cached_channel_read_buffer_count(
        setup=source.setup_spec,
        capture=capture,
        include_holdoff=include_holdoff,
        analysis=analysis,
    )


@sa.util.stopwatch(
    'allocate buffers', 'source', threshold=5e-3, logger_level=logging.DEBUG
)
def alloc_empty_iq(
    source: SourceBase,
    capture: specs.SensorCapture,
    prior: typing.Optional['np.ndarray'] = None,
) -> 'tuple[np.ndarray, tuple[np.ndarray, list[np.ndarray]]]':
    """allocate a buffer of IQ return values.

    Returns:
        The buffer and the list of buffer references for streaming.
    """
    count = get_channel_read_buffer_count(source, include_holdoff=True)

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

    extra = typing.cast(np.ndarray, extra)

    buffers = []
    i = 0
    for channel in range(num_rx_ports):
        if channel in ports:
            buffers.append(typing.cast(np.ndarray, samples[i].view(buf_dtype)))
            i += 1
        elif source.setup_spec.stream_all_rx_ports:
            assert extra is not None
            buffers.append(extra)

    return all_samples, (samples, buffers)


def _is_reusable(
    c1: specs.SensorCapture | None,
    c2: specs.SensorCapture | None,
    master_clock_rate,
):
    """return True if c2 is compatible with the raw and uncalibrated IQ acquired for c1"""

    if c1 is None or c2 is None:
        return False

    fsb1 = design_capture_resampler(master_clock_rate, c1)['fs_sdr']
    fsb2 = design_capture_resampler(master_clock_rate, c2)['fs_sdr']

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
