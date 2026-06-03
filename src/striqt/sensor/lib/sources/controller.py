from __future__ import annotations as __

import contextlib
import dataclasses
import functools
from typing import Any, Callable, cast, ClassVar, Generic, Generator, TYPE_CHECKING
from collections import defaultdict
from threading import Event

import striqt.analysis as sa
import striqt.waveform as sw

from . import base, buffers
from ... import specs
from .. import bindings, util
from ..typing import SS, SC, S, PC, PS, SourceBackend, TypeVar

if TYPE_CHECKING:
    from ..typing import Array, Self
    import numpy as np

    T = TypeVar('T', bound='ControllerBase')
    PendingController = 'ControllerBase | Event | BaseException'

else:
    np = util.lazy_import('numpy')


class lookup:
    """asynchronous lookup of controller objects, connection status, or source ID.

    This assumes that a source instantiation is in progress or will be
    within `timeout` seconds. Otherwise, `TimeoutError` is raised.
    """

    _obj: dict[specs.Source, 'PendingController'] = defaultdict(Event)
    _id: dict[specs.Source, Event | str] = defaultdict(Event)
    _ready: dict[specs.Source, Event | bool] = defaultdict(Event)

    @classmethod
    def instance(cls, spec: specs.Source, timeout=0.5) -> ControllerBase:
        obj = cls._obj[spec]
        if isinstance(obj, BaseException):
            util.propagate_thread_interrupts()
            raise util.ThreadInterruptRequest()
        elif isinstance(obj, ControllerBase):
            # already have this source!
            return obj
        else:
            obj.wait(timeout)

        obj = cls._obj[spec]
        if isinstance(obj, BaseException):
            util.propagate_thread_interrupts()
            raise util.ThreadInterruptRequest()
        elif isinstance(obj, ControllerBase):
            return obj
        else:
            raise TimeoutError('no controller instance initializing given spec')

    @classmethod
    def id(cls, spec: specs.Source, timeout=0.5) -> str:
        """lookup a source ID from a source specification."""
        controller = cls.instance(spec, timeout)

        obj = cls._id[spec]
        if isinstance(obj, BaseException):
            util.propagate_thread_interrupts()
            raise util.ThreadInterruptRequest()
        elif isinstance(obj, Event):
            obj.wait()
        else:
            return obj

        obj = cls._id[spec]
        if isinstance(obj, BaseException):
            util.propagate_thread_interrupts()
            raise util.ThreadInterruptRequest()
        elif isinstance(obj, Event):
            raise TypeError
        elif isinstance(obj, str):
            return obj

    @classmethod
    def is_ready(cls, spec: specs.Source, timeout: float, wait: bool = True) -> bool:
        if wait:
            cls.id(spec, 0.5)
        elif not isinstance(cls._id[spec], str):
            return False
        obj = cls._ready[spec]
        if isinstance(obj, Event):
            if wait:
                obj.wait(max(0, timeout - 0.3))
                return cast(bool, cls._ready[spec])
            else:
                return False
        elif isinstance(obj, bool):
            return obj
        else:
            raise TypeError

    @classmethod
    def _clear(cls, spec: specs.Source):
        cls._obj[spec] = Event()
        cls._ready[spec] = Event()

    @classmethod
    def _register(cls, spec: specs.Source, controller: ControllerBase):
        obj = cls._obj[spec]
        if isinstance(obj, Event):
            cls._obj[spec] = controller
            obj.set()
        else:
            raise TypeError('controller object was already registered for this spec')

    @classmethod
    def _set_ready(cls, spec: specs.Source, is_ready: bool):
        obj = cls._ready[spec]
        if isinstance(obj, Event):
            cls._ready[spec] = is_ready
            obj.set()
        else:
            raise TypeError('source was already setup')

    @classmethod
    def _set_id(cls, spec: specs.Source, id: str):
        obj = cls._id[spec]
        if isinstance(obj, Event):
            cls._id[spec] = id
            obj.set()
        else:
            raise TypeError('id was already setup')

    @classmethod
    def _set_open(cls, spec: specs.Source, controller: ControllerBase):
        cls._obj[spec] = controller

    @classmethod
    def _raise(cls, spec: specs.Source, exc: BaseException):
        cls._obj[spec] = exc
        cls._set_ready(spec, False)
        raise exc


@contextlib.contextmanager
def read_retries(source: ControllerBase) -> Generator[None]:
    """in this context, retry source.read_iq on stream errors"""

    EXC_TYPES = (base.ReceiveStreamError, OverflowError)

    max_count = source.source_info.retries

    if max_count is None or max_count == 0:
        yield
        return

    def prepare_retrigger(*args, **kws):
        source._buffers.clear()
        source._buffers.skip_next_buffer_swap()

    initial = source.read_iq
    retry = util.retry(EXC_TYPES, tries=max_count + 1, exception_func=prepare_retrigger)
    source.read_iq = retry(source.read_iq)  # ty: ignore

    try:
        yield
    finally:
        source.read_iq = initial  # ty: ignore


class ControllerBase(Generic[SS, SC, PS, PC]):
    backend: SourceBackend[SS, SC]
    __setup__: SS
    _capture: SC | None
    _binding: bindings.SensorBinding[SS, Any, SC, PS, PC]
    _buffers: buffers.ReceiveBuffers
    _timeout: float = 10
    _prev_iq: specs.AcquiredIQ | None = None
    _reuse_iq: bool

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def arm(self, *args, **kwargs) -> SC | None:
        raise NotImplementedError

    def is_open(self, wait=True) -> bool:
        return lookup.is_ready(self.__setup__, self._timeout, wait=wait)

    def close(self):
        lookup._clear(self.__setup__)
        try:
            backend = self.backend
            if backend is not None:
                backend.close()
        finally:
            if hasattr(self, '_buffers'):
                self._buffers.clear()

    def __del__(self):
        self.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info):
        if self.is_open():
            self.close()

    @util.cached_property
    def spec(self) -> SS:
        return self.__setup__

    @functools.cached_property
    def source_id(self) -> str:
        return self.backend.get_id()

    @functools.cached_property
    def source_info(self) -> specs.SourceInfo:
        if not self.is_open():
            raise ConnectionError('backend is closed')
        return self.backend.get_info()

    def read_iq(self, overlaps: tuple[int, int] = (0, 0)) -> 'tuple[Array, int|None]':
        """read IQ for the armed capture"""
        assert self._capture is not None, 'source must be armed to read IQ'

        if not isinstance(overlaps, (tuple, list)) or len(overlaps) != 2:
            raise ValueError('overlaps must be a sequence of 2 integers')
        for ol in overlaps:
            if ol % 2 == 1 or ol < 0 or not isinstance(ol, (np.integer, int)):
                raise ValueError('overlaps must be non-negative even integers')

        self.backend.trigger(overlaps)

        # the return buffer
        samples, stream_bufs = self._buffers.get_next(self._capture, overlaps=overlaps)

        # carryover from the previous acquisition
        missing_start_time = True
        start_ns, carryover_count = self._buffers.apply(samples)
        stream_time_ns = start_ns

        # the number of holdoff samples from the end of the holdoff period
        # to include with the returned waveform
        included_holdoff = overlaps[0]

        fs = self.get_resampler()['fs_sdr']

        # the number of valid samples to return per channel
        output_count = buffers.get_read_count(
            self.armed_capture,
            self.spec,
            include_holdoff=False,
            overlap=sum(overlaps),
        )

        # the total number of samples to acquire per channel
        buffer_count = buffers.get_read_count(
            self.armed_capture,
            self.spec,
            include_holdoff=True,
            overlap=sum(overlaps),
        )

        received_count = 0
        chunk_count = remaining = output_count - carryover_count

        while remaining > 0:
            if received_count > 0 or self.spec.gapless:
                on_overflow = 'except'
            else:
                on_overflow = 'ignore'

            request_count = min(chunk_count, remaining)

            if (received_count + request_count) > buffer_count:
                # this could happen if there is a slight mismatch between
                # the requested and realized sample rate
                break

            # Read the samples from the data buffer
            this_count, ret_time_ns = self.backend.read(
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
                included_holdoff = buffers.find_trigger_holdoff(
                    self.spec,
                    self.armed_capture,
                    self._buffers,
                    stream_time_ns,
                    start_overlap=overlaps[0],
                )
                remaining = remaining + included_holdoff - overlaps[0]

                start_ns = stream_time_ns + round(included_holdoff * 1e9 / fs)
                missing_start_time = False

            remaining = remaining - this_count
            received_count += this_count

        samples = samples.view('complex64')
        sample_offs = included_holdoff - overlaps[0]
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
        samples = buffers.cast_iq(self.spec, samples, buffer_count)

        return samples[:, sample_span], start_ns

    @sa.util.stopwatch('acquire', 'source')
    def acquire(
        self,
        overlaps: tuple[int, int] = (0, 0),
        format_path: specs.helpers.PathFormatter | None = None,
    ) -> specs.AcquiredIQ:
        """arm a capture and enable the channel (if necessary), read the resulting IQ waveform.

        Optionally, calibration corrections can be applied, and the radio can be left ready for the next capture.
        """

        self.armed_capture  # ensure we are armed

        if self._prev_iq is None:
            with read_retries(self):
                samples, time_ns = self.read_iq(overlaps)

            info = specs.AcquisitionInfo(source_id=self.source_id)

            iq = specs.AcquiredIQ(
                format_path=format_path,
                pre_align=samples,
                pre_filter=None,
                aligned=None,
                capture=self.armed_capture,
                info=info,
                extra_data={},
                source_spec=self.spec,
                resampler=self.get_resampler(),
                voltage_scale=buffers.get_dtype_scale(self.spec.transport_dtype),
            )

            iq = self.backend.package_iq(iq, samples, time_ns)

        else:
            iq = dataclasses.replace(
                self._prev_iq,
                capture=self.armed_capture,
                info=self._prev_iq.info.replace(start_time=None),
            )

        if self._reuse_iq:
            self._prev_iq = iq

        return iq

    @property
    def armed_capture(self) -> SC:
        """return the specification of the currently armed capture"""

        if self._capture is None:
            raise AttributeError('no capture has been armed')

        return self._capture

    def get_resampler(self) -> sw.ResamplerDesign:
        return self.backend.get_resampler(self.armed_capture)


class RawController(ControllerBase[SS, SC, PS, PC]):
    @sa.util.stopwatch(
        'open IQ source', 'sweep', threshold=0.5, logger_level=util.logging.INFO
    )
    def __init__(
        self,
        spec: SS,
        *,
        reuse_iq: bool = False,
        rx_ports: tuple[int, ...] | None = None,
    ):
        lookup._register(spec, self)

        self.__setup__ = spec
        self._capture = None
        self._reuse_iq = reuse_iq

        try:
            if not hasattr(self, '_binding'):
                raise TypeError('bind a source controller')
            self._buffers = buffers.ReceiveBuffers(self)
            self.backend = self._binding.source(spec)
            lookup._set_id(spec, self.source_id)
        except BaseException as ex:
            lookup._raise(spec, ex)
        else:
            lookup._set_open(spec, self)
        try:
            if spec.array_backend == 'cupy':
                sw.arrays.configure_cupy()
            util.propagate_thread_interrupts()
            self.backend.setup(rx_ports=rx_ports)
            lookup._set_ready(spec, True)
        except:
            lookup._set_ready(spec, False)
            self.close()
            raise

    @sa.util.stopwatch('arm', 'source', threshold=10e-3)
    def arm(self, spec: SC):
        assert self._buffers is not None

        if not self.is_open():
            raise RuntimeError('open the radio before arming')

        if self._capture is not None:
            mcr = self.spec.master_clock_rate
            if self._reuse_iq and buffers.is_reusable(self.armed_capture, spec, mcr):
                pass
            else:
                self._prev_iq = None

        if spec == self._capture and self._capture is not None:
            return

        if not self.spec.gapless or spec != self._capture:
            self._buffers.clear()

        self._capture = self.backend.arm(spec) or spec


class Controller(ControllerBase[SS, SC, PS, PC]):
    def __init__(
        self,
        reuse_iq=False,
        rx_ports: tuple[int, ...] | None = None,
        *args: PS.args,
        **kwargs: PS.kwargs,
    ):
        if not hasattr(self, '_binding'):
            raise TypeError('bind a source controller')

        spec = self._binding.schema.source(*args, **kwargs)  # type: ignore
        cast_self = cast(RawController, self)
        RawController.__init__(cast_self, spec, reuse_iq=reuse_iq, rx_ports=rx_ports)

    def arm(self, *args: PC.args, **kwargs: PC.kwargs):
        """stop the stream, apply a capture configuration, and start it"""

        capture = self._binding.schema.capture(*args, **kwargs)  # type: ignore
        cast_self = cast(RawController, self)
        return RawController.arm(cast_self, capture)
