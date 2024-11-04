import functools
import time
import numpy as np
import SoapySDR
from .base import get_capture_buffer_sizes
from multiprocessing import shared_memory, Event, Lock, Queue, Process
from .. import structs
import labbench as lb

def validate_stream_result(
    sr: 'SoapySDR.StreamResult', remaining: int, on_overflow='except', logger=None
) -> int:
    """validate the stream response after reading.

    Args:
        sr: the return value from self.backend.readStream
        count: the expected number of samples (1 (I,Q) pair each)

    Returns:
        tuple[newly buffered samples, number of samples left to acquire]
    """
    msg = None

    # ensure the proper number of waveform samples was read
    if sr.ret == remaining:
        return sr.ret, 0
    elif sr.ret > 0:
        return sr.ret, remaining - sr.ret
    elif sr.ret == SoapySDR.SOAPY_SDR_OVERFLOW:
        if on_overflow == 'except':
            raise OverflowError(msg)
        elif on_overflow == 'log':
            if logger is None:
                raise OverflowError('tried to log but logger is None')
            logger.debug(msg)
        return 0, remaining
    elif sr.ret < 0:
        raise IOError(f'Error {sr.ret}: {SoapySDR.errToStr(sr.ret)}')
    else:
        raise TypeError(f'did not understand response code {sr.ret}')


@functools.lru_cache
def _array_memory_size(shape, dtype):
    return np.prod(shape) * np.dtype(dtype).itemsize


def empty_shared_array(shape, dtype='float32'):
    shm = shared_memory.SharedMemory(size=_array_memory_size(shape, dtype), create=True)
    array = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
    return array, (shm.name, shape, dtype)


def get_shared_array(shm_name, shape, dtype):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    return np.ndarray(shape=shape, dtype=dtype, buffer=existing_shm.buf)


def get_sweep_max_sample_count(
    base_clock_rate, periodic_trigger, sweep: 'structs.Sweep'
) -> tuple[int, int]:

    kws = dict(
        base_clock_rate=base_clock_rate,
        periodic_trigger=periodic_trigger,
        include_holdoff=True,
    )
    sizes = [
        get_capture_buffer_sizes(capture=capture,**kws)[0]
        for capture in sweep.captures
    ]
    return max(sizes)


class TriggeredAcquisition:
    def __init__(self, radio: SoapySDR.Device, rx_stream, buffer_length, *, channels=0, periodic_trigger=None, on_overflow='except'):
        self.radio = radio
        self.rx_stream = rx_stream
        self.on_overflow=on_overflow
        self.next_timestamp = None
        self.np_buffer = None
        self.periodic_trigger = periodic_trigger
        self.channels = channels

        self._process = None
        self._buffer_lock = Lock()
        self._stop_event = Event()
        self._trigger_queue = Queue(1)
        self._result_queue = Queue(1)

    def _fill_buffer(self, count: int, *, stop_event = None, trigger_queue: Queue=None, start_time=None) -> tuple['np.ndarray', float]:
        fs = self.radio.getSampleRate(SoapySDR.SOAPY_SDR_RX, self.channels[0])
        timeout = max(round(count / fs * 1.5), 50e-3)

        timestamp = start_time

        if self.on_overflow != 'except':
            # this is when the radio_setup is configured as 'host';
            # use the host time to allow overflows between captures,
            # thus avoiding loss of timestamp on overflow
            timestamp = time.time()

        if stop_event is None:
            stop_event = self._stop_event

        if trigger_queue is None:
            trigger_queue = self._trigger_queue

        remaining = count
        block_size = int(50e-3*fs)
        holdoff_count = None
        total_received = 0

        on_overflow = self.on_overflow

        while remaining > 0:
            if stop_event.is_set() or not trigger_queue.empty():
                return None
            
            # Read the samples from the data buffer
            rx_result = self.radio.readStream(
                self.rx_stream,
                [self.np_buffer[total_received * 2 :]],
                min(block_size, remaining),
                timeoutUs=int(timeout * 1e6),
            )

            if timestamp is None:
                timestamp = rx_result.timeNs / 1e9
                if timestamp == 0:
                    raise RuntimeError('radio did not return a timestamp')

            if self.periodic_trigger is not None and holdoff_count is None:
                # determine the number of holdoff samples to reject
                excess_time = timestamp % self.periodic_trigger
                holdoff_count = round(fs * (self.periodic_trigger - excess_time))
                remaining = remaining + holdoff_count
                timestamp = timestamp + holdoff_count / 1e9

            elif holdoff_count is None:
                holdoff_count = 0

            received, remaining = validate_stream_result(rx_result, remaining, on_overflow=on_overflow)
            total_received += received

            # never allow overflow within a capture
            on_overflow = 'except'

        # samples = self.np_buffer.view('complex64')[holdoff_count : samples + (holdoff_count or 0)]
        return holdoff_count, timestamp

    def _worker(self, shm_info, stop_event, buffer_lock, trigger_queue: Queue, received_queue: Queue):
        """this runs in another process"""

        buffer = get_shared_array(*shm_info)

        while True:
            if stop_event.is_set():
                break

            try:
                # update count, if provided
                count = trigger_queue.get(timeout=50e-3)
            except TimeoutError:
                pass

            with buffer_lock:
                result = self._fill_buffer(buffer, count, stop_event=stop_event)
                if result is None:
                    return
                holdoff_count, timestamp = result
                received_queue.put((count, holdoff_count, timestamp))

    def arm(self, buffer_length: int):
        if self._process is not None:
            raise RuntimeError('already armed')
        self.np_buffer, shm_info = empty_shared_array((buffer_length,))       
        self._process = Process(target=self._worker, args=(shm_info, self._stop_event, self._buffer_lock, self._trigger_queue, self._result_queue))
        self._process.start()

    def stop(self):
        if self._process is None:
            return
        self._stop_event.set()
        with self._buffer_lock:
            pass
        self._stop_event.clear()
        self._process = None

    def __del__(self):
        self.stop()

    def trigger(self, count: int, *, channels=[0]):
        self._trigger_queue.put((count, channels))

    def acquire(self):
        fs = self.radio.getSampleRate(SoapySDR.SOAPY_SDR_RX, 0)
        timeout = round(self.np_buffer.shape[0] / fs) + 100e-3

        with self._buffer_lock:
            count, holdoff_count, timestamp = self._result_queue.get(timeout=timeout)

        samples = self.np_buffer.view('complex64')[holdoff_count : count + (holdoff_count or 0)]
        return samples, timestamp
