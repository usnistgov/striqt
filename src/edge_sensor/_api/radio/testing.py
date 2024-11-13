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


class SingleToneSource(NullSource):
    resource: float = attr.value.float(
        default=0.2, min=-1, max=1, help='normalized tone frequency (between -1 and 1)'
    )

    def get_waveform(self, count, start_index: int, *, channel: int = 0, xp=np):
        i = xp.arange(start_index, count+start_index, dtype='uint64')
        f_cw = self.sample_rate() * self.resource
        return xp.exp((2j * np.pi * f_cw)/self.backend_sample_rate()*i + np.pi/2).astype('complex64')


class SawtoothSource(NullSource):
    resource: float = attr.value.float(
        default=0.5, min=0, help='duration of the ramp normalized by acquisition duration', label='s'
    )

    def get_waveform(self, count, start_index: int, *, channel: int = 0, xp=np):
        ret = xp.empty(count, dtype='complex64')

        period = self.duration * self.resource
        t = xp.arange(start_index, count+start_index, dtype='uint64')/self.backend_sample_rate()
        ret.real[:] = (t%period)/period
        ret.imag[:] = 0
        return ret


@functools.lru_cache(1)
def cached_noise(capture, xp):
    return channel_analysis.simulated_awgn(capture, xp=xp, seed=0)


class NoiseSource(NullSource):
    resource: float = attr.value.float(
        default=1e-3, min=0, help='noise waveform variance'
    )

    def get_waveform(self, count, start_index: int, *, channel: int = 0, xp=np):
        ii = xp.arange(start_index, count+start_index, dtype='uint64') % count
        capture = channel_analysis.Capture(duration=self.duration, sample_rate=self.backend_sample_rate())
        x = cached_noise(capture, xp=xp)

        return x[ii]


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

    def get_waveform(self, count: int, offset: int, *, channel: int = 0, xp=np):
        size = int(self.backend['header_fd']['total_samples'][0])
        ref_level = self.backend['header_fd']['reference_level_dBm'][0]

        if size < count:
            raise ValueError(
                f'requested {count} samples but file capture length is {size} samples'
            )

        scale = 10 ** (float(ref_level) / 20.0) / np.iinfo(xp.int16).max
        i, q = self.backend['iq_fd'].channels()
        iq = xp.empty((2 * count,), dtype=xp.int16)
        iq[offset*2::2] = xp.asarray(i[offset:count+offset])
        iq[1+offset*2::2] = xp.asarray(q[offset:count+offset])

        return (iq * xp.float32(scale)).view('complex64').copy()