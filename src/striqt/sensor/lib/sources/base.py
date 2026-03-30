from __future__ import annotations as __

import dataclasses
import functools
from typing import Callable, cast, ClassVar, Generic, TYPE_CHECKING
from collections import defaultdict
from threading import Event

import striqt.analysis as sa
import striqt.waveform as sw

from . import buffers
from ... import specs
from .. import util
from ..typing import SS, SC, S, PC, PS, Source, TypeVar

if TYPE_CHECKING:
    from ..typing import Array, Self
    import numpy as np

    T = TypeVar('T', bound='SourceBase')

else:
    np = util.lazy_import('numpy')


_source_id_map: dict[specs.Source, SourceBase | Event] = defaultdict(Event)


def get_source_id(spec: specs.Source, timeout=0.5) -> str:
    """lookup a source ID from a source specification.

    This assumes that a source is either instantiated, or will be
    within `timeout` seconds; otherwise `TimeoutError` is raised.
    """
    obj = _source_id_map[spec]

    if not isinstance(obj, Event):
        # already have this source!
        return obj.id

    if not obj.wait(timeout=timeout):
        util.propagate_thread_interrupts()
        raise TimeoutError('timeout while waiting for a source ID')

    source = _source_id_map[spec]
    assert isinstance(source, SourceBase)

    # this triggers a property access that may have its own
    # blocking wait
    return source.id


@dataclasses.dataclass(frozen=True)
class Schema(Generic[SS, SC]):
    source: type[SS]
    capture: type[SC]


def bind_schema_types(
    source: type[SS], capture: type[SC]
) -> Callable[[type[T]], type[T]]:
    """set the default to a SourceBase subclass"""

    def decorator(cls: type[T]) -> type[T]:
        cls._bindings__ = Schema(source=source, capture=capture)
        return cls

    return decorator


def get_bound_spec(spec: specs.SpecBase | None, cls: type[S] | None, **kws) -> S:
    if isinstance(spec, specs.SpecBase):
        # if cls is not None:
        #     spec = cast(TB, cls.from_spec(spec))
        capture = spec.replace(**kws)
    elif spec is not None:
        raise TypeError(f'spec must be an instance of {cls.__qualname__!r} or None')
    elif cls is None:
        raise TypeError('an explicit argument of type specs.Capture is required')
    else:
        capture = cls(**kws)

    return cast(S, capture)


class SourceBase(Source[SS, SC, PS, PC]):
    _bindings__: ClassVar[Schema | None] = None

    _buffers: buffers.ReceiveBuffers
    _is_open: bool | Event = False
    _timeout: float = 10

    def __init__(self, reuse_iq=False, *args: PS.args, **kwargs: PS.kwargs):
        open_event = self._is_open = Event()  # first, to serve other threads

        # back door from .from_spec
        _extra_specs = cast(dict, kwargs.pop('__specs', {}))
        _spec = _extra_specs.pop('source', None)

        if _spec is not None:
            _spec = cast(SS, _spec)

        if self._bindings__ is None:
            spec_cls = None
        else:
            spec_cls = cast(type[SS], self._bindings__.source)

        _spec = get_bound_spec(_spec, spec_cls, **kwargs)

        _map_source(_spec, self)

        self.__setup__ = _spec
        self._capture = None
        self._buffers = buffers.ReceiveBuffers(self)
        self._prev_iq: buffers.AcquiredIQ | None = None
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
            sw.arrays.configure_cupy()

        try:
            util.propagate_thread_interrupts()
        except:
            self.close()
            raise

        self._apply_setup(_spec, **_extra_specs)

    @classmethod
    def from_spec(cls, spec: SS, *, captures=None, loops=None, reuse_iq=False) -> Self:
        kwargs = spec.to_dict()
        kwargs['__specs'] = {'source': spec, 'captures': captures, 'loops': loops}

        if captures is not None and len(captures) > 0 and cls._bindings__ is None:
            raise TypeError('can only hint captures for source class bindings')

        return cls(reuse_iq=reuse_iq, **kwargs)  # pyright: ignore

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
                return cast(bool, self._is_open)
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
    def setup_spec(self) -> SS:
        return self.__setup__

    @sa.util.stopwatch('arm', 'source', threshold=10e-3)
    def arm(self, *args, **kwargs):
        """stop the stream, apply a capture configuration, and start it"""
        assert self._buffers is not None

        if self._bindings__ is not None:
            capture_cls = self._bindings__.capture
        elif self._capture is not None:
            capture_cls = type(self._capture)
        else:
            raise TypeError('no capture bindings were supplied')

        capture = get_bound_spec(None, capture_cls, **kwargs)

        return self.arm_spec(capture)

    def arm_spec(self, spec: SC):
        if not self.is_open():
            raise RuntimeError('open the radio before arming')

        if self._capture is not None:
            mcr = self.setup_spec.master_clock_rate
            if self._reuse_iq and buffers.is_reusable(self.capture_spec, spec, mcr):
                pass
            else:
                self._prev_iq = None

        if spec == self._capture and self._capture is not None:
            return

        if not self.setup_spec.gapless or spec != self._capture:
            self._buffers.clear()

        self._capture = self._prepare_capture(spec) or spec

    def read_iq(self, overlaps=(0, 0)) -> 'tuple[Array, int|None]':
        """read IQ for the armed capture"""
        assert self._capture is not None, 'soapy source must be armed to read IQ'

        if not isinstance(overlaps, (tuple, list)) or len(overlaps) != 2:
            raise ValueError('overlaps must be a sequence of 2 integers')
        for ol in overlaps:
            if ol % 2 == 1 or ol < 0 or not isinstance(ol, (np.integer, int)):
                raise ValueError('overlaps must be non-negative even integers')

        # the return buffer
        samples, stream_bufs = self._buffers.get_next(self._capture)

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
            self.capture_spec, self.setup_spec, include_holdoff=False
        )

        # the total number of samples to acquire per channel
        buffer_count = buffers.get_read_count(
            self.capture_spec, self.setup_spec, include_holdoff=True
        )

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
                included_holdoff = buffers.find_trigger_holdoff(
                    self.setup_spec,
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
        samples = buffers.cast_iq(self.setup_spec, samples, buffer_count)

        return samples[:, sample_span], start_ns

    @sa.util.stopwatch('acquire', 'source')
    def acquire(self, overlaps=(0, 0)) -> buffers.AcquiredIQ:
        """arm a capture and enable the channel (if necessary), read the resulting IQ waveform.

        Optionally, calibration corrections can be applied, and the radio can be left ready for the next capture.
        """

        self.capture_spec  # ensure we are armed

        if self._prev_iq is None:
            samples, time_ns = self.read_iq(overlaps)
            iq = self._package_acquisition(samples, time_ns)

        else:
            iq = dataclasses.replace(
                self._prev_iq,
                capture=self.capture_spec,
                info=self._prev_iq.info.replace(start_time=None),
            )

        if self._reuse_iq:
            self._prev_iq = iq

        return iq

    def _package_acquisition(
        self,
        samples: Array,
        time_ns: int | None,
    ) -> buffers.AcquiredIQ:
        info = specs.AcquisitionInfo(source_id=self.id)

        return buffers.AcquiredIQ(
            pre_align=samples,
            pre_filter=None,
            aligned=None,
            capture=self.capture_spec,
            info=info,
            extra_data={},
            source_spec=self.setup_spec,
            resampler=self.get_resampler(),
            voltage_scale=buffers.get_dtype_scale(self.setup_spec.transport_dtype),
        )

    def _read_stream(
        self,
        buffers,
        offset,
        count,
        timeout_sec,
        *,
        on_overflow: specs.types.OnOverflow = 'except',
    ) -> tuple[int, int]:
        """to be implemented in subclasses"""
        raise NotImplementedError

    @property
    def capture_spec(self) -> SC:
        """generate the currently armed capture configuration for the specified channel.

        If the truth of realized evaluates as False, only the requested value
        of backend_sample_rate is returned in the given radio capture.
        """

        if self._capture is None:
            raise AttributeError('arm to set the capture spec')

        return self._capture

    def get_resampler(self, capture=None) -> sw.ResamplerDesign:
        from .. import resampling

        if capture is None:
            capture = self.capture_spec

        return resampling.design_resampler(capture, self.setup_spec.master_clock_rate)


class VirtualSource(SourceBase[SS, SC, PS, PC]):
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
        on_overflow: specs.types.OnOverflow = 'except',
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
    ) -> Array:
        raise NotImplementedError

    def _sync_time_source(self):
        self._sync_time_ns = round(1_000_000_000 * self._samples_elapsed)


def _map_source(spec: specs.Source, source: SourceBase):
    maybe_event = _source_id_map[spec]
    _source_id_map[spec] = source

    if isinstance(maybe_event, Event):
        maybe_event.set()
