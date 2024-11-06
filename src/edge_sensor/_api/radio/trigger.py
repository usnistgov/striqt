import time
import numpy as np
from multiprocessing import Event, Lock, Queue, Process
import typing

if typing.TYPE_CHECKING:
    import labbench as lb
    import numpy as np
    import SoapySDR
    from . import base
    from channel_analysis._api import shmarray
else:
    lb = lb.util.lazy_import('labbench')
    np = lb.util.lazy_import('numpy')
    SoapySDR = lb.util.lazy_import('SoapySDR')
    shmarray = lb.util.lazy_import('channel_analysis._api.shmarray')


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


class TriggeredAcquisition:
    def __init__(self, radio: 'base.RadioBase', sample_count):
        self.radio_backend = radio.backend
        self.stream = radio.stream
        self.buffer = radio._inbuf
        self.gapless=radio.gapless_repeats
        self.periodic_trigger = radio.periodic_trigger
        self.channel = radio.channel()
        self.sample_count = sample_count

        self._process = None
        self._buffer_lock = Lock()
        self._stop_event = Event()
        self._result_queue = Queue()

    def _worker(self, buffer_lock, result_queue, stop_event):
        """this runs in another process"""

        self.radio_backend.activateStream(
            self.stream,
            flags=SoapySDR.SOAPY_SDR_HAS_TIME,
        )

        try:
            fs = self.radio_backend.getSampleRate(SoapySDR.SOAPY_SDR_RX, self.channel)
            start_time = None

            while True:
                with buffer_lock:
                    if result_queue.get(block=False) is None:
                        pass
                    elif self.gapless:
                        raise OverflowError('skipped waveform(s)')

                    ret = self.read_once(start_time, stop_event=stop_event)
                    if stop_event.is_set():
                        break
                    _, timestamp = ret   
                    result_queue.put(ret)

                start_time = timestamp + self.sample_count/fs
                time.sleep(20e-3)

        finally:
            self.radio_backend.deactivateStream(self.stream)

    def start(self):
        if self._process is not None:
            raise RuntimeError('already armed')
        self._process = Process(target=self._worker)
        self._process.start()

    def get(self, xp=np):
        fs = self.radio_backend.getSampleRate(SoapySDR.SOAPY_SDR_RX, self.channel)
        timeout = round((20e-3+self.sample_count / fs) * 1.25)

        with self._buffer_lock:
            samples, timestamp = self._result_queue.get(timeout=timeout)
            samples = xp.asarray(samples).copy()
        return samples, timestamp

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
        fs = self.radio_backend.getSampleRate(SoapySDR.SOAPY_SDR_RX, 0)
        timeout = round(self.np_buffer.shape[0] / fs) + 100e-3

        with self._buffer_lock:
            count, holdoff_count, timestamp = self._result_queue.get(timeout=timeout)

        samples = self.np_buffer.view('complex64')[holdoff_count : count + (holdoff_count or 0)]
        return samples, timestamp

    def read_once(self, start_time=None, stop_event=None) -> tuple['shmarray.NDSharedArray', float]:
        fs = self.radio_backend.getSampleRate(SoapySDR.SOAPY_SDR_RX, self.channel)
        timeout = max(round(self.sample_count / fs * 1.5), 50e-3)

        timestamp = start_time

        remaining = self.sample_count
        chunk_size = int(50e-3*fs)
        holdoff_count = None
        total_received = 0

        if stop_event is None:
            stop_event = self._stop_event

        while remaining > 0:
            if stop_event.is_set():
                return None

            if total_received > 0 or self.gapless:
                on_overflow='except'
            else:
                on_overflow='ignore'

            # Read the samples from the data buffer
            rx_result = self.radio_backend.readStream(
                self.stream,
                [self.buffer[total_received * 2 :]],
                min(chunk_size, remaining),
                timeoutUs=int(timeout * 1e6),
            )

            if timestamp is None:
                timestamp = rx_result.timeNs / 1e9
                if timestamp == 0:
                    raise RuntimeError('no timestamp')

            if self.periodic_trigger not in (None, 0) and holdoff_count is None:
                # determine the number of holdoff samples to reject
                excess_time = timestamp % self.periodic_trigger
                holdoff_count = round(fs * (self.periodic_trigger - excess_time))
                remaining = remaining + holdoff_count
                timestamp = timestamp + holdoff_count / fs

            elif holdoff_count is None:
                holdoff_count = 0

            received, remaining = validate_stream_result(rx_result, remaining, on_overflow=on_overflow)
            total_received += received

        samples = self.buffer.view('complex64')[holdoff_count : self.sample_count + (holdoff_count or 0)]
        return samples, timestamp
