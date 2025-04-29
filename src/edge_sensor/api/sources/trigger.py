from __future__ import annotations
import numpy as np
from threading import Event, Thread
import typing
from .. import util, structs
from . import base
import labbench as lb
import SoapySDR

if typing.TYPE_CHECKING:
    import numpy as np
    from channel_analysis.api import shmarray
else:
    lb = util.lazy_import('labbench')
    np = util.lazy_import('numpy')
    SoapySDR = util.lazy_import('SoapySDR')
    shmarray = util.lazy_import('channel_analysis.api.shmarray')


def find_trigger_holdoff(start_time, sample_rate, periodic_trigger: float | None):
    if periodic_trigger in (0, None):
        return 0

    periodic_trigger_ns = round(periodic_trigger * 1e9)

    # float rounding errors cause problems here; evaluate based on the 1-ns resolution
    excess_time_ns = start_time % periodic_trigger_ns
    holdoff_ns = (periodic_trigger_ns - excess_time_ns) % periodic_trigger_ns
    holdoff = round(holdoff_ns / 1e9 * sample_rate)

    if holdoff != 0:
        lb.logger.info(f'snapping to periodic trigger with {holdoff}-sample holdoff')

    return holdoff


class ThreadedStream:
    def __init__(self, radio: base.RadioSource, capture: structs.RadioCapture, xp=np):
        self._thread = None
        self.radio = radio
        self._xp = xp
        self.arm(capture)

    def arm(
        self,
        capture: structs.RadioCapture,
    ):
        self.capture = capture

        buf_size_in, self._buf_out_size = base.get_capture_buffer_sizes(
            self.radio, capture, include_holdoff=True
        )
        self.sample_count, _ = base.get_capture_buffer_sizes(
            self.radio, capture, include_holdoff=False
        )

        self._buf_in = np.empty((2 * buf_size_in,), dtype=np.float32)
        self._result = None, None
        self._running = Event()
        self._acquired = Event()
        self._request = Event()
        self._stop = Event()
        self._trigger = Event()
        self._thread_exc = None

    def _fill_buffer(
        self, start_time=None, holdover_samples: 'np.ndarray[np.complex64]' = None
    ) -> tuple['np.ndarray[np.complex64]', float]:
        fs = self.radio.backend_sample_rate()
        chunk_size = min(int(50e-3 * fs), self.sample_count)

        if start_time is None and holdover_samples is not None:
            raise ValueError(
                'when start time is None, holdover_samples must also be None'
            )

        holdoff = 0
        streamed_count = 0
        timeout_sec = chunk_size / fs + 50e-3
        timestamp_ns = start_time

        if holdover_samples is None:
            holdover_count = 0
        else:
            # note: holdover_count.dtype is np.complex64, while self._buf_in.dtype is np.float32
            holdover_count = holdover_samples.size
            self._buf_in[: 2 * holdover_count] = holdover_samples.view(
                self._buf_in.dtype
            )

        remaining = self.sample_count - holdover_count
        first = True

        while remaining > 0:
            if self._stop.is_set() or self._trigger.is_set():
                if start_time is None:
                    return None, start_time
                else:
                    return None, start_time + round(1e9 * streamed_count / fs)

            if streamed_count > 0 or self.radio.gapless_repeats:
                on_overflow = 'except'
            else:
                on_overflow = 'ignore'

            # Read the samples from the data buffer
            this_count, ret_time_ns = self.radio._read_stream(
                [self._buf_in],
                offset=holdover_count + streamed_count,
                count=min(chunk_size, remaining),
                timeout_sec=timeout_sec,
                on_overflow=on_overflow,
            )

            if start_time is None:
                # special case for the first read in the stream, since
                # devices may not always return timestamps
                start_time = ret_time_ns

                if start_time == 0:
                    raise RuntimeError('no timestamp')

            if first:
                holdoff = find_trigger_holdoff(
                    start_time, fs, self.radio.periodic_trigger
                )

                remaining = remaining + holdoff
                timestamp_ns = start_time + round(1e9 * holdoff / fs)

                first = False

            streamed_count += this_count

        samples = self._buf_in.view('complex64')[holdoff : self.sample_count + holdoff]
        return samples, timestamp_ns

    def _service_loop(self):
        """acquire samples in a loop"""

        next_time_ns = None

        # to skip transients in the resampler, pass in extra samples from the
        # previous acquisition allow for settling.
        if self.capture.host_resample:
            holdover_size = self.sample_count - round(
                self.capture.duration * self.radio.backend_sample_rate()
            )
        else:
            holdover_size = None
        holdover_samples = None

        try:
            while True:
                samples, time_ns = self._fill_buffer(next_time_ns, holdover_samples)
                if holdover_size is not None:
                    holdover_samples = samples[-holdover_size:]

                if self._stop.is_set():
                    break
                elif self._trigger.is_set():
                    # on trigger, _read_buffer returns the timestamp is at the
                    # end of its (likely incomplete) buffer read
                    next_time_ns = time_ns
                    self._trigger.clear()
                    continue

                try:
                    # TODO: revisit this constant
                    self._request.wait(timeout=20e-3)
                except TimeoutError:
                    if self.radio.gapless_repeats:
                        raise OverflowError('gapless repeat acquisition failed')
                else:
                    # this runs if the request flag has been set by get(). copy and allocate
                    # first to ensure proper sequencing with the main thread
                    buf_out = self._xp.empty(self._buf_out_size, dtype=np.complex64)
                    buf_out[: samples.size] = self._xp.asarray(samples)
                    self._result = buf_out, time_ns / 1e9
                    self._acquired.set()
                    self._request.clear()

                next_time_ns = time_ns + round(1e9 * self.capture.duration)

        except BaseException as ex:
            self._stop.set()
            self._thread_exc = ex

    def start(self):
        if self._thread is not None:
            raise RuntimeError('already armed')
        self._acquired.clear()
        self._request.set()
        self._running.clear()
        self._stop.clear()
        self._trigger.clear()
        self._thread_exc = None
        self._thread = Thread(target=self._service_loop)

        with lb.stopwatch('enable'):
            self.radio.arm(self.capture)
            self.radio.channel_enabled(True)

        self._thread.start()

    def is_running(self):
        running = self._thread.is_alive() and not self._stop.is_set()
        if not running and self._thread_exc is not None:
            raise self._thread_exc
        return running

    def get(self) -> tuple['np.ndarray', float]:
        if not self.is_running():
            raise RuntimeError('start acquisition with start() before call to get()')
        self._request.set()
        timeout = 50e-3 + self.sample_count / self.radio.backend_sample_rate()

        try:
            self._acquired.wait(timeout=timeout)
        except TimeoutError as ex:
            exc = ex
        else:
            exc = None

        if exc:
            raise TimeoutError('receive stream buffer underflow')

        samples, timestamp = self._result

        self._acquired.clear()

        if samples is None:
            if self._thread_exc is not None:
                raise self._thread_exc
            else:
                raise TimeoutError('no data')

        return samples, timestamp

    def stop(self):
        self.radio.channel_enabled(False)
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None

    def __del__(self):
        self.stop()

    def trigger(self):
        self._trigger.set()

    # def acquire(self):
    #     fs = self.radio.backend.getSampleRate(SoapySDR.SOAPY_SDR_RX, 0)
    #     timeout = round(self.np_buffer.shape[0] / fs) + 100e-3

    #     with self._buffer_lock:
    #         count, holdoff_count, timestamp = self.shared.result_queue.get(timeout=timeout)

    #     samples = self.np_buffer.view('complex64')[holdoff_count : count + (holdoff_count or 0)]
    #     return samples, timestamp
