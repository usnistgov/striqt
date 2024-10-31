import functools
import time
import numpy as np
import SoapySDR
from .base import get_capture_buffer_sizes
from multiprocessing import shared_memory, Lock, Queue
from .. import structs

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


class TriggeredAcquisition:
    def __init__(self, radio: SoapySDR.Device, rx_stream, *, periodic_trigger=None, on_overflow='except'):
        self.radio = radio
        self.rx_stream = rx_stream
        self.on_overflow=on_overflow
        self.next_timestamp = None
        self.np_buffer = None
        self.periodic_trigger = periodic_trigger
        self.shm_info = None

        self.buffer_lock = Lock()
        self.pending_acquire = Queue(1)

    def _prepare_buffer(self, capture: structs.RadioCapture):
        samples_in, _ = get_capture_buffer_sizes(self, capture, include_holdoff=True)

        # total buffer size for 2 values per IQ sample
        size_in = 2 * samples_in

        if self.np_buffer is None or self.np_buffer.size < size_in:
            self._logger.debug(
                f'allocating input sample buffer ({size_in * 2 /1e6:0.2f} MB)'
            )
            self.np_buffer, self.shm_info = empty_shared_array((size_in,))
            self._logger.debug('done')

    def _fill_buffer(self, count: int, *, channels=[0], start_time=None) -> tuple['np.ndarray', float]:
        fs = self.radio.getSampleRate(SoapySDR.SOAPY_SDR_RX, channels[0])
        timeout = max(round(count / fs * 1.5), 50e-3)

        timestamp = start_time

        if self.on_overflow != 'except':
            # this is when the radio_setup is configured as 'host';
            # use the host time to allow overflows between captures,
            # thus avoiding loss of timestamp on overflow
            timestamp = time.time()

        remaining = count
        block_size = int(50e-3*fs)
        strobe_holdoff = None
        total_received = 0

        on_overflow = self.on_overflow

        while remaining > 0:
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

            if self.periodic_trigger is not None and strobe_holdoff is None:
                # determine the number of holdoff samples to reject
                excess_time = timestamp % self.periodic_trigger
                strobe_holdoff = round(fs * (self.periodic_trigger - excess_time))
                remaining = remaining + strobe_holdoff
                timestamp = timestamp + strobe_holdoff / 1e9

            elif strobe_holdoff is None:
                strobe_holdoff = 0

            received, remaining = validate_stream_result(rx_result, remaining, on_overflow=on_overflow)
            total_received += received

            # never allow overflow within a capture
            on_overflow = 'except'

        # samples = self.np_buffer.view('complex64')[strobe_holdoff : samples + (strobe_holdoff or 0)]
        return strobe_holdoff, timestamp

    def service(self):
        while True:
            if self.stop_event.is_set():
                break

            with self.buffer_lock:
                count, channels = self.pending_acquire.get()
                self._fill_buffer(count, channels=channels)

    def read(self, count: int, *, channels=[0]):
        strobe_holdoff, timestamp = self._fill_buffer(count, channels=channels)
        samples = self.np_buffer.view('complex64')[strobe_holdoff : count + (strobe_holdoff or 0)]
        return samples, timestamp
