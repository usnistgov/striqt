from __future__ import annotations as __

import contextlib
import dataclasses
import functools
from typing import Any, Callable, cast, ClassVar, Generic, Generator, TYPE_CHECKING
from collections import defaultdict
from threading import Event

import striqt.analysis as sa
import striqt.waveform as sw

from .sources import base, buffers
from .typing import SS, SP, SC, S, PC, PS, SourceBackend, TypeVar
from . import compute, util
from .. import specs

if TYPE_CHECKING:
    from .typing import Array, Self
    import numpy as np

    T = TypeVar('T', bound='Controller')
    PendingController = 'Controller | Event | BaseException'
    from . import bindings

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
    def instance(cls, spec: specs.Source, timeout=0.5) -> Controller:
        obj = cls._obj[spec]
        if isinstance(obj, BaseException):
            util.propagate_thread_interrupts()
            raise util.ThreadInterruptRequest()
        elif isinstance(obj, Controller):
            # already have this source!
            return obj
        else:
            obj.wait(timeout)

        obj = cls._obj[spec]
        if isinstance(obj, BaseException):
            util.propagate_thread_interrupts()
            raise util.ThreadInterruptRequest()
        elif isinstance(obj, Controller):
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
        cls._id[spec] = Event()

    @classmethod
    def _register(cls, spec: specs.Source, controller: Controller):
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
    def _set_open(cls, spec: specs.Source, controller: Controller):
        cls._obj[spec] = controller

    @classmethod
    def _raise(cls, spec: specs.Source, exc: BaseException):
        cls._obj[spec] = exc
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
    source.read_iq = retry(source.read_iq)  # ty: ignore

    try:
        yield
    finally:
        source.read_iq = initial  # ty: ignore


@dataclasses.dataclass
class ControllerConfig:
    reuse_iq: bool
    format_path: specs.helpers.PathFormatter | None
    analysis: specs.AnalysisGroup | None
    init_rx_ports: tuple[int, ...] | None


class Controller(Generic[SS, SP, SC, PS, PC]):
    """Opens a source backend and controls it with the binding's setup
    and capture specifications a binding.
    """
    backend: SourceBackend[SS, SC]
    __setup__: SS
    _capture: SC | None
    _binding: 'bindings.SensorBinding[SS, SP, SC, PS, PC]'
    _buffers: buffers.ReceiveBuffers
    _timeout: float = 10
    _prev_iq: specs.AcquiredIQ | None = None
    _config: ControllerConfig

    @sa.util.stopwatch(
        'open IQ source', 'sweep', threshold=0.5, logger_level=util.logging.INFO
    )
    def __init__(self, *args: PS.args, **kwargs: PS.kwargs):
        if not hasattr(self, '_binding'):
            raise TypeError('use this after binding a source controller')

        config = ControllerConfig(
            init_rx_ports=None, reuse_iq=False, analysis=None, format_path=None
        )
        spec = self._binding.schema.source(*args, **kwargs)  # type: ignore
        self._setup(spec, config)

    def _setup(self, spec: SS, config: ControllerConfig) -> 'Self':
        self._config = config
        self.__setup__ = spec
        self._capture = None

        lookup._register(spec, self)

        try:
            if not hasattr(self, '_binding'):
                raise TypeError('access controller from a binding')
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
        spec: specs.Sweep[SS, Any, SC],
        format_path: specs.helpers.PathFormatter | None = None,
    ) -> Controller[SS, SP, SC, PS, PC]:
        self = cast(Controller[SS, SP, SC, PS, PC], object.__new__(cls))

        config = ControllerConfig(
            init_rx_ports=specs.helpers.get_unique_ports(spec.captures, spec.loops),
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
        format_path: specs.helpers.PathFormatter | None = None,
    ) -> Controller[SS, SP, SC, PS, PC]:
        self = cast(Controller[SS, SP, SC, PS, PC], object.__new__(cls))
        config = ControllerConfig(
            init_rx_ports=rx_ports,
            reuse_iq=reuse_iq,
            analysis=None,
            format_path=format_path,
        )
        return self._setup(spec, config)

    def arm(self, *args: PC.args, **kwargs: PC.kwargs) -> SC | None:
        assert self._buffers is not None
        spec = self._binding.schema.capture(*args, **kwargs)  # type: ignore
        return self._arm_spec(spec)

    @sa.util.stopwatch('arm', 'source', threshold=10e-3)
    def _arm_spec(self, spec: SC) -> SC | None:
        if not self.is_open():
            raise RuntimeError('open the radio before arming')

        cal = self.source_spec.calibration
        if cal and specs.helpers.get_format_fields(cal):
            if not self._config.format_path:
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

        if spec == self._capture and self._capture is not None:
            return

        if not self.source_spec.gapless or spec != self._capture:
            self._buffers.clear()

        self._capture = self.backend.arm(spec) or spec

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
        else:
            lookup._clear(self.__setup__)

    def target_analysis(self, analysis: specs.AnalysisGroup | None):
        """sets or disables an analysis target to auto-select acquisition overlap"""

        if analysis is None:
            pass
        elif not isinstance(analysis, specs.AnalysisGroup):
            raise TypeError('analysis argument must be None or an AnalysisGroup spec')

        self._config.analysis = analysis

    def set_calibration_formatter(
        self, format_path: specs.helpers.PathFormatter | None = None
    ):
        """set the formatter to use to expand calibration file formatting fields.

        This is needed if calibration is defined in the source spec with format fields
        like `{name}`.
        """
        if format_path is None or isinstance(format_path, specs.helpers.PathFormatter):
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
            self.capture_spec,
            self.source_spec,
            include_holdoff=False,
            overlap=sum(overlaps),
        )

        # the total number of samples to acquire per channel
        buffer_count = buffers.get_read_count(
            self.capture_spec,
            self.source_spec,
            include_holdoff=True,
            overlap=sum(overlaps),
        )

        received_count = 0
        chunk_count = remaining = output_count - carryover_count

        while remaining > 0:
            if received_count > 0 or self.source_spec.gapless:
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
        samples = buffers.cast_iq(self.source_spec, samples, buffer_count)

        return samples[:, sample_span], start_ns

    @sa.util.stopwatch('acquire', 'source')
    def acquire(self, overlaps: tuple[int, int] | None = None) -> specs.AcquiredIQ:
        """acquire IQ samples needed for the armed capture."""

        self.capture_spec  # ensure we are armed

        if isinstance(overlaps, tuple):
            pass
        elif overlaps is not None:
            raise ValueError('overlaps must be a tuple or None')
        elif self._config.analysis is None:
            raise ValueError(
                'call .target_analysis() or pass overlap (start, stop) samples'
            )
        else:
            analysis = specs.helpers.adjust_analysis(
                self._config.analysis, self.capture_spec.adjust_analysis
            )
            overlaps = compute.get_correction_overlaps(
                self.capture_spec, self.source_spec, analysis
            )

        if self._prev_iq is None:
            with read_retries(self):
                samples, time_ns = self.read_iq(overlaps)

            info = specs.AcquisitionInfo(source_id=self.source_id)

            iq = specs.AcquiredIQ(
                format_path=self._config.format_path,
                pre_align=samples,
                pre_filter=None,
                aligned=None,
                capture=self.capture_spec,
                info=info,
                extra_data={},
                source_spec=self.source_spec,
                resampler=self.get_resampler(),
                voltage_scale=buffers.get_dtype_scale(self.source_spec.transport_dtype),
            )

            iq = self.backend.package_iq(iq, samples, time_ns)

        else:
            iq = dataclasses.replace(
                self._prev_iq,
                capture=self.capture_spec,
                info=self._prev_iq.info.replace(start_time=None),
            )

        if self._config.reuse_iq:
            self._prev_iq = iq

        return iq

    @property
    def capture_spec(self) -> SC:
        """return the specification of the currently armed capture"""

        if self._capture is None:
            raise AttributeError('no capture has been armed')

        return self._capture

    def get_resampler(self) -> sw.ResamplerDesign:
        return self.backend.get_resampler(self.capture_spec)
