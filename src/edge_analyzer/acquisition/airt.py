# Import Packages
import numpy as np
import sys
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CS16, errToStr
import cupy as cp
import time
import numba
import numba.cuda


class AirTCapture:
    """fast simplified single-channel receive acquisition"""
    def __init__(self, frequency_Hz, sample_rate_Hz, rx_channel=0, gain_dB=0.0, timeout_s=5.0):
        # Initialize the AIR-T receiver using SoapyAIRT
        self.rx_buff = None
        self.reset_counts()
        
        self.sdr = SoapySDR.Device(dict(driver="SoapyAIRT"))  # Create AIR-T instance
        self.set_sample_rate(sample_rate_Hz)
        self.sdr.setGainMode(SOAPY_SDR_RX, 0, False)
        self.set_frequency(frequency_Hz)
        self.set_gain(gain_dB)
        self.rx_stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16, [rx_channel])
        self.sdr.activateStream(self.rx_stream)  # this turns the radio on
        self.timeout = timeout_s

    def get_scale(self):
        return 1 / 32767.0

    def set_frequency(self, frequency_Hz: float):
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, frequency_Hz)

    def set_gain(self, gain_dB: float):
        self.sdr.setGain(SOAPY_SDR_RX, 0, gain_dB)

    def set_sample_rate(self, sample_rate_Hz):
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate_Hz)

    def reset_counts(self):
        self.acquisition_counts = {'overflow': 0, 'exception': 0, 'total': 0}

    def get_scale(self):
        return 1 / 32767.0

    def acquire(self, N, scale=True, raise_on_overflow=False) -> cp.ndarray:
        if self.rx_buff is None or self.rx_buff.size < 4 * N:
            # create a buffer for received samples that can be shared across CPU<->GPU
            #     ref: https://github.com/cupy/cupy/issues/3452#issuecomment-903273011
            # 
            # this is double sized compared to the usual number of (int16, int16) IQ pairs, 
            # because later we want to store upcasted np.float32 without an extra (allocate, copy)
            #     ref: notebooks/profile_cupy.ipynb
            self.rx_buff = numba.cuda.mapped_array(
                (4 * N,),
                dtype=np.int16,
                strides=None,
                order="C",
                stream=0,
                portable=False,
                wc=False,
            )

        # Read the samples from the data buffer
        sr = self.sdr.readStream(
            self.rx_stream, [self.rx_buff[:4*N:2]], N, timeoutUs=int(self.timeout * 1e6)
        )

        self.acquisition_counts['total'] += 1
        
        # ensure the proper number of waveform samples was read
        if sr.ret == N:
            pass
        elif sr.ret == -4 and raise_on_overflow:
            raise OverflowError(
                f"buffer overflow on acquisition {self.acquisition_counts['total']}"
            )
        else:
            if sr.ret == -4:
                self.acquisition_counts['overflow'] += 1
                total_info = f"{self.acquisition_counts['overflow']}/{self.acquisition_counts['total']}"
                print(f"{time.perf_counter()}: overflow (total {total_info}")
            else:
                self.acquisition_counts['exceptions'] += 1
                print("Error {}: {}".format(sr.ret, errToStr(sr.ret)), sys.stderr)
            return None

        # what follows is some acrobatics to minimize new memory allocation and copy
        buff_int16 = cp.array(self.rx_buff, copy=False)[:4*N]

        # 1. re-interpret the buffer contents as float32 without casting
        buff_float32 = buff_int16.view('float32')

        # 2. in-place casting from the int16 samples, filling in the extra allocation in self.rx_buff
        cp.copyto(buff_float32, buff_int16[::2], casting='unsafe')

        # 3. last, re-interpret each interleaved (float32 I, float32 Q) as a complex value
        buff_complex64 = buff_float32.view("complex64")        

        # 4. convert to full scale
        if scale:
            buff_complex64 *= self.get_scale()

        return buff_complex64

    def __del__(self):
        self.close()

    def close(self):
        try:
            self.sdr.deactivateStream(self.rx_stream)
            self.sdr.closeStream(self.rx_stream)
        except ValueError as ex:
            if 'invalid parameter' in str(ex):
                # already closed
                pass
            else:
                raise

if __name__ == "__main__":
    airt = AirTCapture(freq=2.44e9, fs=2 * 31.25e6)
    iq = airt.acquire(256 * 1024)
    airt.close()