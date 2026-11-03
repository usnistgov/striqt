from __future__ import annotations as __

import contextlib
import dataclasses
import functools
import inspect
import weakref
from collections import defaultdict, deque
from collections.abc import Generator
from queue import Full, Queue
from threading import Event, RLock
from typing import TYPE_CHECKING, Any, ClassVar, Generic, cast

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis.lib.util import np

from .. import specs
from . import compute, util
from .sources import base, buffers
from .typing import PC, PS, SA, SC, SP, SS, SourceBackend, TypeVar

if TYPE_CHECKING:
    from typing import Callable

    from .typing import Array, Self, TypeAlias

    T = TypeVar('T', bound='Controller')
    PendingController: TypeAlias = 'Controller | Event | BaseException'
    PendingSlot: TypeAlias = 'weakref.ref[Controller] | Event | BaseException'
    from . import bindings

    _QueueBase = Queue[Any]
else:
    # queue.Queue is not subscriptable at runtime on python 3.9
    _QueueBase = Queue


class lookup:
    """lookup of the controller, source ID, and readiness of a source spec.

    Each lookup blocks until a controller is registered for the spec. `TimeoutError`
    is raised when none is registered within `timeout` seconds.
    """

    _obj: ClassVar[dict[specs.Source, PendingSlot]] = defaultdict(Event)
    _id: ClassVar[dict[specs.Source, Event | str | BaseException]] = defaultdict(Event)
    _ready: ClassVar[dict[specs.Source, Event | bool]] = defaultdict(Event)
    _lock = RLock()

    @classmethod
    def _get(cls, spec: specs.Source) -> PendingController:
        """return the controller, pending event, or stored exception for `spec`.

        The registry references controllers weakly. A dead reference is cleared. The
        pending event that replaces the reference is returned.
        """
        obj = cls._obj[spec]
        if isinstance(obj, weakref.ref):
            controller = obj()
            if controller is None:
                cls._clear(spec)
                obj = cls._obj[spec]
                assert isinstance(obj, Event)
                return obj
            return controller
        return obj

    @classmethod
    def _finalized(cls, spec: specs.Source, ref: weakref.ref[Controller]) -> None:
        with cls._lock:
            if cls._obj.get(spec) is ref:
                cls._clear(spec)

    @classmethod
    def instance(cls, spec: specs.Source, timeout=0.5) -> Controller:
        obj = cls._get(spec)
        if isinstance(obj, BaseException):
            util.propagate_thread_interrupts()
            raise util.ThreadInterruptRequest()
        elif isinstance(obj, Controller):
            return obj
        else:
            obj.wait(timeout)

        obj = cls._get(spec)
        if isinstance(obj, BaseException):
            util.propagate_thread_interrupts()
            raise util.ThreadInterruptRequest()
        elif isinstance(obj, Controller):
            return obj
        else:
            # a canceled open may never register a controller. A TimeoutError here
            # would mask the error that ExceptionStack raises in place of the
            # interrupt, so the interrupt propagates first
            util.propagate_thread_interrupts()
            raise TimeoutError('no controller instance initializing given spec')

    @classmethod
    def id(cls, spec: specs.Source, timeout=0.5) -> str:
        """lookup a source ID from a source specification."""
        cls.instance(spec, timeout)

        obj: Event | str | BaseException = cls._id[spec]
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
            raise RuntimeError('the source closed before it reported a source id')
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
    def _clear(cls, spec: specs.Source) -> None:
        with cls._lock:
            for table in (cls._obj, cls._ready, cls._id):
                # a waiter holds the event it found in the slot. Replacing the event
                # without setting it leaves the waiter blocked
                old = table.get(spec)
                if isinstance(old, Event):
                    old.set()
                table[spec] = Event()

    @classmethod
    def _register(cls, spec: specs.Source, controller: Controller) -> None:
        """register `controller` in the slot of `spec` and wake the waiters on the slot.

        An open controller already in the slot is closed first, with a WARNING on
        the ``source`` logger.
        """
        with cls._lock:
            obj = cls._get(spec)
            if isinstance(obj, Controller):
                if not obj._closed:
                    sa.util.get_logger('source').warning(
                        'closing the open controller for an equal source spec before '
                        'reopening it; release controllers with close() or a `with` '
                        'block'
                    )
                    obj.close()
                else:
                    cls._clear(spec)
                obj = cls._obj[spec]
            elif isinstance(obj, BaseException):
                cls._clear(spec)
                obj = cls._obj[spec]
            assert isinstance(obj, Event)
            cls._obj[spec] = weakref.ref(
                controller, functools.partial(cls._finalized, spec)
            )
            obj.set()

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
    def _raise(cls, spec: specs.Source, exc: BaseException):
        cls._obj[spec] = exc
        pending = cls._id[spec]
        cls._id[spec] = exc
        if isinstance(pending, Event):
            pending.set()
        cls._set_ready(spec, False)
        raise exc


@contextlib.contextmanager
def read_retries(source: Controller) -> Generator[None]:
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
    source.read_iq = retry(source.read_iq)  # ty: ignore[invalid-assignment]

    try:
        yield
    finally:
        source.read_iq = initial  # ty: ignore[invalid-assignment]


def _put_waiting(queue: AcquisitionQueue, iq: specs.AcquiredIQ) -> None:
    """put `iq`, waiting for a free slot and checking for a thread interrupt."""
    while True:
        try:
            queue.put(iq, timeout=0.1)
        except Full:
            util.propagate_thread_interrupts()
        else:
            return


class AcquisitionQueue(_QueueBase):
    """the bounded hand-off of acquisitions from the acquisition thread to analysis.

    The items are `AcquiredIQ` objects and the sentinel or exception objects that the
    producer puts to end the iteration. `maxsize` is the `buffers` count given at
    construction.
    """

    nbytes: int
    """the total `x.nbytes` of the `AcquiredIQ` items in the queue; the holdoff
    samples ahead of the record of a first segment are not counted"""

    def __init__(self, buffers: int):
        """create an empty queue that holds up to `buffers` items."""
        super().__init__(maxsize=buffers)
        self.nbytes = 0

    def _put(self, item: Any) -> None:
        super()._put(item)
        if isinstance(item, specs.AcquiredIQ):
            self.nbytes += int(item.x.nbytes)

    def _get(self) -> Any:
        item = super()._get()
        if isinstance(item, specs.AcquiredIQ):
            self.nbytes -= int(item.x.nbytes)
        return item

    def purge(self, capture_index: int) -> int:
        """remove the `AcquiredIQ` items with ``info.capture_index == capture_index``.

        Returns:
            the number of items removed
        """
        with self.mutex:
            kept: deque[Any] = deque()
            removed = 0
            for item in self.queue:
                if (
                    isinstance(item, specs.AcquiredIQ)
                    and item.info.capture_index == capture_index
                ):
                    removed += 1
                    self.nbytes -= int(item.x.nbytes)
                else:
                    kept.append(item)
            self.queue = kept
            if removed:
                self.unfinished_tasks -= removed
                self.not_full.notify_all()
        return removed


@dataclasses.dataclass
class ControllerConfig:
    reuse_iq: bool
    format_path: specs.paths.PathFormatter | None
    analysis: specs.AnalysisGroup | None
    init_rx_ports: tuple[int, ...] | None


def bind_controller(
    binding: bindings.SensorBinding[SS, SP, SC],
    schema: specs.Schema[SS, SP, SC, SA, PS, PC],
) -> type[Controller[SS, SP, SC, SA, PS, PC]]:
    class cls(Controller, sensor=binding, schema=schema):
        # wrapper shims for customized docstrings
        def __init__(self, *args: PS.args, **kwargs: PS.kwargs):
            super().__init__(*args, **kwargs)

        def arm(self, *args: PC.args, **kwargs: PC.kwargs) -> SC | None:
            return super().arm(*args, **kwargs)

    sensor = cls.sensor
    sensor_cls = type(sensor)
    cls.__module__ = sensor_cls.__module__
    cls.__name__ = f'{sensor_cls.__name__}.controller'
    cls.__qualname__ = f'{sensor_cls.__qualname__}.controller'

    return cls


class Controller(Generic[SS, SP, SC, SA, PS, PC]):
    """Open the source backend for control and acquisition."""

    backend: SourceBackend[SS, SC]
    __setup__: SS
    _capture: SC | None
    _buffers: buffers.ReceiveBuffers
    _timeout: float = 10
    _closed: bool = False
    _prev_iq: specs.AcquiredIQ | None = None
    _config: ControllerConfig
    schema: ClassVar[specs.Schema[Any, Any, Any, Any, Any, Any]]
    sensor: ClassVar[bindings.SensorBinding[Any, Any, Any]]

    @sa.util.stopwatch(
        'open IQ source', 'sweep', threshold=0.5, logger_level=util.logging.INFO
    )
    def __init__(self, *args: PS.args, **kwargs: PS.kwargs):
        if not hasattr(self, 'sensor'):
            raise TypeError(
                'use a subclass of Controller that has been bound with bind_sensor'
            )

        config = ControllerConfig(
            init_rx_ports=None, reuse_iq=False, analysis=None, format_path=None
        )
        spec = self.schema.source(*args, **kwargs)
        self._setup(spec, config)

    def __init_subclass__(
        cls,
        *,
        sensor: bindings.SensorBinding[SS, SP, SC],
        schema: specs.Schema[SS, SP, SC, SA, PS, PC],
    ):
        """customize documentation on subclassing"""

        if not isinstance(schema, specs.Schema):
            raise TypeError(
                'schema argment is not a striqt.sensor.specs.Schema instance'
            )
        cls.schema = schema
        cls.sensor = sensor
        super().__init_subclass__()
        source_cls = cls.sensor.source_cls
        backend_name = sa.util.qualified_name(source_cls)

        # document cls.__init__()
        source_spec = cls.schema.source
        cls.__doc__ = '\n\n'.join((
            f'Open and control the backend `{backend_name}`.',
            source_cls.__doc__ or '',
        ))

        descs, types = sa.specs.doc.describe_msgspec_fields(source_spec)
        arg_descs = [
            f'    - **{n}** (*{t}*): {d}\n'
            for n, t, d in zip(descs, types.values(), descs.values())
        ]
        cls.__doc__ += '\n\n' + '\n'.join(['Parameters:'] + arg_descs)

        # document cls.arm()
        capture_spec = cls.schema.capture
        descs, types = sa.specs.doc.describe_msgspec_fields(capture_spec)
        arg_descs = [
            f'    - **{n}** (*{t}*): {d}\n'
            for n, t, d in zip(descs, types.values(), descs.values())
        ]
        cls.arm.__doc__ = '\n\n' + '\n'.join(['Parameters:'] + arg_descs)
        cls.__signature__ = inspect.signature(cls.schema.source)
        sig = inspect.signature(cls.schema.capture).replace(
            return_annotation=cls.schema.capture
        )
        cls.arm.__signature__ = sig  # ty: ignore[unresolved-attribute]
        return cls

    def _setup(self, spec: SS, config: ControllerConfig) -> Self:
        self._config = config
        self.__setup__ = spec
        self._capture = None

        lookup._register(spec, self)

        try:
            if not hasattr(self, 'sensor'):
                raise TypeError('access controller from a binding')
            self._buffers = buffers.ReceiveBuffers(self)
            self.backend = self.sensor.source_cls(spec)
            lookup._set_id(spec, self.source_id)
        except BaseException as ex:
            try:
                lookup._raise(spec, ex)
            finally:
                self.close()
        try:
            if spec.array_backend == 'cupy':
                sw.arrays.configure_cupy()
            util.propagate_thread_interrupts()
            self.backend.setup(rx_ports=config.init_rx_ports)
            lookup._set_ready(spec, True)
        except:
            lookup._set_ready(spec, False)
            self.close()
            raise
        return self

    @classmethod
    def from_sweep_spec(
        cls,
        spec: specs.Sweep[SS, Any, SC, SA],
        format_path: specs.paths.PathFormatter | None = None,
    ) -> Controller[SS, SP, SC, SA, PS, PC]:
        self = cast(Controller[SS, SP, SC, SA, PS, PC], object.__new__(cls))

        config = ControllerConfig(
            init_rx_ports=specs.captures.get_unique_ports(spec.captures, spec.loops),
            reuse_iq=spec.options.reuse_iq,
            analysis=spec.analysis,
            format_path=format_path,
        )
        return self._setup(spec.source, config)

    @classmethod
    def from_source_spec(
        cls,
        spec: SS,
        reuse_iq: bool = False,
        rx_ports: tuple[int, ...] | None = None,
        format_path: specs.paths.PathFormatter | None = None,
    ) -> Controller[SS, SP, SC, SA, PS, PC]:
        self = cast(Controller[SS, SP, SC, SA, PS, PC], object.__new__(cls))
        config = ControllerConfig(
            init_rx_ports=rx_ports,
            reuse_iq=reuse_iq,
            analysis=None,
            format_path=format_path,
        )
        return self._setup(spec, config)

    def arm(self, *args: PC.args, **kwargs: PC.kwargs) -> SC | None:
        assert self._buffers is not None
        spec = self.schema.capture(*args, **kwargs)
        return self._arm_spec(spec)

    @sa.util.stopwatch('arm', 'source', threshold=10e-3)
    def _arm_spec(self, spec: SC) -> SC | None:
        if not self.is_open():
            raise RuntimeError('open the radio before arming')

        cal = self.source_spec.calibration
        if cal and specs.paths.get_format_fields(cal) and not self._config.format_path:
            raise TypeError(
                'calibration is specified with path formatter fields - '
                'set a formatter with set_path_formatter before arming'
            )

        if self._capture is not None:
            mcr = self.source_spec.master_clock_rate
            if self._config.reuse_iq and buffers.is_reusable(
                self.capture_spec, spec, mcr
            ):
                pass
            else:
                self._prev_iq = None

        # a carryover belongs to the burst of the armed capture alone
        self._buffers.clear()

        if spec == self._capture and self._capture is not None:
            return

        self._capture = self.backend.arm(spec) or spec

    def is_open(self, wait: bool = True) -> bool:
        """return whether the backend of this instance is open.

        Args:
            wait: if True, block up to the controller timeout for an open in
                progress; if False, report the state at the time of the call

        Returns:
            False once `close` has run on this instance, even if another controller
            has since been opened for an equal source spec
        """
        if self._closed:
            return False
        return lookup.is_ready(self.__setup__, self._timeout, wait=wait)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        # the registry is keyed by spec value, so a stale controller that the
        # garbage collector finalizes late must not clear the entry of a live
        # controller that was opened afterward with an equal spec
        current = lookup._get(self.__setup__)
        if current is self or isinstance(current, BaseException):
            lookup._clear(self.__setup__)

        try:
            backend = getattr(self, 'backend', None)
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
        self.close()

    def target_analysis(self, analysis: specs.AnalysisGroup | None):
        """sets or disables an analysis target to auto-select acquisition overlap"""

        if analysis is None:
            pass
        elif not isinstance(analysis, specs.AnalysisGroup):
            raise TypeError('analysis argument must be None or an AnalysisGroup spec')

        self._config.analysis = analysis

    def discard_iq(self) -> None:
        """drop the acquisition that `reuse_iq` would serve to the next matching capture.

        The next `acquire` reads the source even when the armed capture is unchanged.
        """
        self._prev_iq = None

    def set_calibration_formatter(
        self, format_path: specs.paths.PathFormatter | None = None
    ):
        """set the formatter that expands the path fields of `source.calibration`.

        None clears the formatter. A formatter is required when the calibration path
        has fields such as ``{name}``.

        Raises:
            TypeError: if `format_path` is not a `PathFormatter` or None
        """
        if format_path is None or isinstance(format_path, specs.paths.PathFormatter):
            self._config.format_path = format_path
        else:
            raise TypeError('format_path must be a PathFormatter or None')

    @util.cached_property
    def source_spec(self) -> SS:
        return self.__setup__

    @functools.cached_property
    def source_id(self) -> str:
        return self.backend.get_id()

    @functools.cached_property
    def source_info(self) -> specs.SourceInfo:
        if not self.is_open():
            raise ConnectionError('backend is closed')
        return self.backend.get_info()

    def read_iq(
        self,
        overlaps: tuple[int, int] = (0, 0),
        *,
        burst_index: int = 0,
        resume: bool = False,
    ) -> tuple[Array, int | None]:
        """read the IQ of one burst segment of the armed capture.

        Segment 0, and a segment read with `resume`, triggers the backend and applies
        the transient holdoff. A later segment continues the enabled stream from the
        samples carried over after the previous segment. The carryover is kept for
        the next segment and cleared after the last segment of the burst.

        Args:
            overlaps: the ``(lead, tail)`` overlap in source samples around the
                record
            burst_index: the index of the segment within the burst of the armed
                capture, from 0 to ``capture.burst - 1``
            resume: True reads segment `burst_index` as the first segment after a
                stream error, with no carryover

        Returns:
            ``(samples, start_time_ns)``: the samples with shape (ports, count), and
            the time of the first record sample or None without a time stamp

        Raises:
            ValueError: if `overlaps` is not 2 non-negative integers
            OverflowError: on a stream overflow after the first read of a segment, or
                at any read of a segment that continues the stream
            MemoryError: if the backend reports more samples than requested
        """
        assert self._capture is not None, 'source must be armed to read IQ'
        first = burst_index == 0 or resume

        if not isinstance(overlaps, (tuple, list)) or len(overlaps) != 2:
            raise ValueError('overlaps must be a sequence of 2 integers')
        for ol in overlaps:
            if ol < 0 or not isinstance(ol, (np.integer, int)):
                raise ValueError('overlaps must be non-negative integers')

        if first:
            self.backend.trigger(overlaps)

        # the return buffer
        samples, stream_bufs = self._buffers.get_next(
            self._capture, overlaps=overlaps, include_holdoff=first
        )

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
            self.capture_spec,
            self.source_spec,
            include_holdoff=False,
            overlap=sum(overlaps),
        )

        # the total number of samples to acquire per channel
        # a segment that continues the stream consumes no holdoff, so the read, the
        # copy, and the queued memory are bounded by the record and overlaps
        buffer_count = buffers.get_read_count(
            self.capture_spec,
            self.source_spec,
            include_holdoff=first,
            overlap=sum(overlaps),
        )

        received_count = 0
        chunk_count = remaining = output_count - carryover_count

        while remaining > 0:
            if received_count > 0 or not first:
                on_overflow = 'except'
            else:
                on_overflow = 'ignore'

            request_count = min(chunk_count, remaining)

            if (received_count + request_count) > buffer_count:
                # a slight mismatch between the requested and realized sample rate
                # can push the request past the buffer
                break

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
                    self.source_spec,
                    self.capture_spec,
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

        record = compute.source_record_size(self._capture, self.get_resampler())
        unused_count = output_count - record
        if burst_index < self._capture.burst - 1 and start_ns is not None:
            self._buffers.stash_carryover(
                samples[:, sample_span],
                start_ns,
                unused_sample_count=unused_count,
                capture=self._capture,
                start_overlap=overlaps[0],
            )
        else:
            self._buffers.clear()

        samples = buffers.cast_iq(self.source_spec, samples, buffer_count)

        return samples[:, sample_span], start_ns

    @sa.util.stopwatch('acquire', 'source')
    def acquire(
        self,
        overlaps: tuple[int, int] | None = None,
        *,
        burst_index: int = 0,
        resume: bool = False,
    ) -> specs.AcquiredIQ:
        """acquire one burst segment of the armed capture.

        The read of segment 0, or of a segment with `resume`, is repeated up to
        ``source_info.retries`` times on a stream error or overflow. A stream error
        at a later segment is raised to the caller, which owns the burst.

        Args:
            overlaps: the ``(lead, tail)`` overlap in source samples, or None to
                derive the overlap from the capture and the analysis target
            burst_index: the index of the segment within the burst, stamped on
                ``info.burst_index``
            resume: True reads segment `burst_index` as the first segment after a
                stream error

        Returns:
            the acquisition, with `x` of shape ``(ports, lead + record + tail)`` at
            the source rate; the `reuse_iq` copy of the previous acquisition when
            the armed capture is reusable

        Raises:
            AttributeError: if no capture is armed
            ValueError: if `overlaps` is neither a tuple nor None
            ReceiveStreamError: on a stream error once the retries are exhausted
            OverflowError: on a stream overflow once the retries are exhausted
        """

        self.capture_spec  # noqa: B018  raises AttributeError when no capture is armed
        signal_trigger = None
        first = burst_index == 0 or resume

        if isinstance(overlaps, tuple):
            pass
        elif overlaps is not None:
            raise ValueError('overlaps must be a tuple or None')
        else:
            if self._config.analysis is None:
                analysis = None
                signal_trigger = None
            else:
                analysis = specs.sequencing.adjust_analysis(
                    self._config.analysis, self.capture_spec.adjust_analysis
                )
                signal_trigger = compute.get_trigger_from_spec(
                    self.source_spec, analysis
                )

            overlaps = compute.get_correction_overlaps(
                self.capture_spec, self.source_spec, analysis
            )

        if self._prev_iq is None:
            if first:
                with read_retries(self):
                    samples, time_ns = self.read_iq(
                        overlaps, burst_index=burst_index, resume=resume
                    )
            else:
                samples, time_ns = self.read_iq(overlaps, burst_index=burst_index)

            info = specs.AcquisitionInfo(
                source_id=self.source_id, burst_index=burst_index
            )

            iq = specs.AcquiredIQ(
                format_path=self._config.format_path,
                x=samples,
                capture=self.capture_spec,
                info=info,
                extra_data={},
                source_spec=self.source_spec,
                resampler=self.get_resampler(),
                voltage_scale=buffers.get_dtype_scale(self.source_spec.transport_dtype),
            )

            iq = self.backend.package_iq(iq, samples, time_ns)
            iq.info = iq.info.replace(signal_trigger=signal_trigger)

        else:
            info = self._prev_iq.info
            if hasattr(info, 'start_time'):
                info = info.replace(start_time=None)
            iq = dataclasses.replace(
                self._prev_iq, capture=self.capture_spec, info=info
            )

        if self._config.reuse_iq:
            self._prev_iq = iq

        return iq

    def acquire_burst(
        self,
        overlaps: tuple[int, int] | None = None,
        *,
        queue: AcquisitionQueue | None = None,
        finalize: Callable[[specs.AcquiredIQ], specs.AcquiredIQ] | None = None,
    ) -> list[specs.AcquiredIQ]:
        """acquire the ``capture.burst`` segments of the armed capture.

        The segments are read back to back with the stream enabled. A stream error
        or overflow at segment `k` is retried up to ``source_info.retries`` times in
        total per burst, after `prepare_retrigger` on the backend: under
        ``bursts.on_gap='skip'`` segment `k` is read again as a first segment; under
        ``'except'`` the segments of the burst still in `queue` are removed and the
        burst restarts at segment 0. With no retries the error is raised.

        Each segment passes through `finalize` before it is queued or returned. A
        segment that finds `queue` full is dropped with a WARNING on the ``source``
        logger under ``'skip'``, and raises under ``'except'``. The acquisition of a
        capture with `burst` 1 waits for a free slot instead.

        Args:
            overlaps: the ``(lead, tail)`` overlap in source samples, or None to
                derive the overlap from the capture and the analysis target
            queue: the queue that receives each segment, or None to return the
                segments
            finalize: a function applied to each segment before the queue put

        Returns:
            the segments in order when `queue` is None, otherwise an empty list

        Raises:
            ValueError: if `burst` is above 1 and `capture.duration` is not a whole
                number of `trigger_strobe` periods
            AcquisitionOverrunError: if `queue` is full under
                ``bursts.on_gap='except'``
            ReceiveStreamError: on a stream error once the retries are exhausted
            OverflowError: on a stream overflow once the retries are exhausted
            ThreadInterruptRequest: in a worker thread, when `cancel_threads` has
                been called
        """
        capture = self.capture_spec
        bursts = self.source_spec.bursts
        retries = self.source_info.retries or 0
        strobe = self.source_spec.trigger_strobe or 0
        if capture.burst > 1 and strobe and not sw.isroundmod(capture.duration, strobe):
            raise ValueError(
                'the duration of a capture with burst above 1 must be a whole number '
                'of trigger_strobe periods'
            )

        results: list[specs.AcquiredIQ] = []
        k = 0
        attempts = 0
        resume = False
        queued_index: int | None = None

        while k < capture.burst:
            try:
                iq = self.acquire(overlaps, burst_index=k, resume=k > 0 and resume)
            except (base.ReceiveStreamError, OverflowError):
                attempts += 1
                if attempts > retries:
                    raise
                self.backend.prepare_retrigger()
                self._buffers.clear()
                self._buffers.skip_next_buffer_swap()
                if bursts.on_gap == 'skip':
                    resume = True
                else:
                    if queue is not None and queued_index is not None:
                        queue.purge(queued_index)
                    k = 0
                continue

            resume = False
            if finalize is not None:
                iq = finalize(iq)

            if queue is None:
                results.append(iq)
            elif capture.burst == 1:
                _put_waiting(queue, iq)
            else:
                queued_index = iq.info.capture_index
                try:
                    queue.put_nowait(iq)
                except Full:
                    if bursts.on_gap == 'skip':
                        sa.util.get_logger('source').warning(
                            f'dropped burst segment {k} of capture '
                            f'{iq.info.capture_index}: the acquisition queue is full'
                        )
                    else:
                        raise base.AcquisitionOverrunError(
                            f'no free acquisition queue slot for burst segment {k} '
                            f'of capture {iq.info.capture_index}'
                        ) from None

            k += 1
            if k < capture.burst:
                util.propagate_thread_interrupts()

        return results

    def disarm(self) -> None:
        """disable the stream of the backend until the next `arm` or `acquire`."""
        self.backend.disarm()

    @property
    def capture_spec(self) -> SC:
        """return the specification of the currently armed capture"""

        if self._capture is None:
            raise AttributeError('no capture has been armed')

        return self._capture

    def get_resampler(self) -> sw.ResamplerDesign:
        return self.backend.get_resampler(self.capture_spec)
