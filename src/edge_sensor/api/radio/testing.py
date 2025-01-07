"""Fake radios for testing"""

import typing

from . import base
from .null import NullSource
from ..util import import_cupy_with_fallback

import functools
import labbench as lb
from labbench import paramattr as attr


if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import channel_analysis
else:
    np = lb.util.lazy_import('numpy')
    pd = lb.util.lazy_import('pandas')
    channel_analysis = lb.util.lazy_import('channel_analysis')


def lo_shift_tone(inds, radio: base.RadioDevice, xp):
    _, lo_offset, _ = base.design_capture_filter(
        radio.base_clock_rate, radio.get_capture_struct()
    )
    return xp.exp((2j * np.pi * lo_offset) / radio.backend_sample_rate() * inds).astype(
        'complex64'
    )


class TestSource(NullSource):
    def _read_stream(
        self, buffers, offset, count, timeout_sec=None, *, on_overflow='except'
    ) -> tuple[int, int]:
        for channel, buf in zip(self.channel(), buffers):
            values = self.get_waveform(
                count,
                self._samples_elapsed,
                channel=channel,
                xp=getattr(self, 'xp', np),
            )
            buf[2 * offset : 2 * (offset + count)] = values.view('float32')

        return super()._read_stream(
            buffers, offset, count, timeout_sec=timeout_sec, on_overflow=on_overflow
        )

    def get_waveform(self, count, start_index: int, *, channel: int = 0, xp):
        raise NotImplementedError


class SingleToneSource(TestSource):
    resource: float = attr.value.float(
        default=0, help='normalized tone frequency (between -1 and 1)', label='Hz'
    )

    noise_snr: float = attr.value.float(
        None, help='add noise at the specified power level'
    )

    def get_waveform(self, count, start_index: int, *, channel: int = 0, xp=np):
        i = xp.arange(start_index, count + start_index, dtype='uint64')
        f_cw = self.resource
        lo = lo_shift_tone(i, self, xp)

        ret = lo * xp.exp(
            (2j * np.pi * f_cw) / self.backend_sample_rate() * i + np.pi / 2
        )
        ret = ret.astype('complex64')

        if self.noise_snr is not None:
            capture = channel_analysis.Capture(
                duration=self.duration, sample_rate=self.backend_sample_rate()
            )
            noise = cached_noise(capture, xp=xp, power=10 ** (-self.noise_snr / 10))
            noise = noise[i % noise.size]
            ret += noise

        return ret


class SawtoothSource(TestSource):
    resource: float = attr.value.float(
        default=0.01,
        min=0,
        help='sawtooth period',
        label='s',
    )

    def get_waveform(self, count, start_index: int, *, channel: int = 0, xp=np):
        ret = xp.empty(count, dtype='complex64')
        period = self.resource
        ii = xp.arange(start_index, count + start_index, dtype='uint64')
        t = ii / self.backend_sample_rate()
        ret.real[:] = (t % period) / period
        ret.imag[:] = 0
        return ret


@functools.lru_cache(1)
def cached_noise(capture, xp, **kwargs):
    return channel_analysis.simulated_awgn(capture, xp=xp, seed=0, **kwargs)


class NoiseSource(TestSource):
    resource: float = attr.value.float(
        default=1e-3, min=0, help='noise waveform variance'
    )

    def get_waveform(self, count, start_index: int, *, channel: int = 0, xp=np):
        capture = channel_analysis.Capture(
            duration=self.duration, sample_rate=self.backend_sample_rate()
        )
        x = cached_noise(capture, xp=xp)
        ii = xp.arange(start_index, count + start_index, dtype='uint64') % x.size

        ret = x[ii]
        ret *= lo_shift_tone(ii, self, xp)
        return ret


class TDMSFileSource(NullSource):
    """returns IQ waveforms from a TDMS file"""

    resource: str = attr.value.str(default=None, help='path to the tdms file')

    def open(self):
        from nptdms import TdmsFile

        fd = TdmsFile.read(self.resource)
        header_fd, iq_fd = fd.groups()
        self.backend = dict(header_fd=header_fd, iq_fd=iq_fd)

    @property
    def base_clock_rate(self):
        return self.backend['header_fd']['IQ_samples_per_second'][0]

    @attr.method.float(
        min=0,
        cache=True,
        label='Hz',
        help='sample rate before resampling',
    )
    def backend_sample_rate(self):
        return self.base_clock_rate

    @backend_sample_rate.setter
    def _(self, value):
        if value != self.base_clock_rate:
            raise ValueError(
                f'file sample rate must match capture ({self.base_clock_rate})'
            )

    @attr.method.float(
        min=0,
        cache=True,
        label='Hz',
        help='center frequency',
    )
    def center_frequency(self):
        return self.backend['header_fd']['carrier_frequency'][0]

    @center_frequency.setter
    def _(self, value):
        actual = self.center_frequency()
        if value != actual:
            self._logger.warning(
                f'center frequency ignored, using {actual/1e6} MHz from file'
            )

    def get_waveform(
        self, count: int, offset: int, *, channel: int = 0, xp=np, dtype='complex64'
    ):
        size = int(self.backend['header_fd']['total_samples'][0])
        ref_level = self.backend['header_fd']['reference_level_dBm'][0]

        if size < count:
            raise ValueError(
                f'requested {count} samples but file capture length is {size} samples'
            )

        scale = 10 ** (float(ref_level) / 20.0) / np.iinfo(xp.int16).max
        i, q = self.backend['iq_fd'].channels()
        iq = xp.empty((2 * count,), dtype=xp.int16)
        iq[offset * 2 :: 2] = xp.asarray(i[offset : count + offset])
        iq[1 + offset * 2 :: 2] = xp.asarray(q[offset : count + offset])

        float_dtype = np.finfo(np.dtype(dtype)).dtype

        return (iq * float_dtype(scale)).view(dtype).copy()
