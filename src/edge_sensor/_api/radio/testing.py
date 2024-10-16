"""Fake radios for testing"""

import typing

from .null import NullSource
from ..util import import_cupy_with_fallback

import labbench as lb
from labbench import paramattr as attr

if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd
else:
    np = lb.util.lazy_import('numpy')
    pd = lb.util.lazy_import('pandas')


class SingleToneSource(NullSource):
    resource: float = attr.value.float(
        default=0.2, min=-1, max=1, help='normalized tone frequency (between -1 and 1)'
    )

    def _read_stream(self, N):
        xp = import_cupy_with_fallback()
        ret = xp.empty(N, dtype='complex64')

        t = np.arange(N) / self.sample_rate()
        f_cw = self.sample_rate() * self.resource
        ret[:] = xp.cos((2 * np.pi * f_cw) * t)

        return ret, pd.Timestamp('now')


class NoiseSource(NullSource):
    resource: float = attr.value.float(
        default=1e-3, min=0, help='noise waveform variance'
    )

    def _read_stream(self, N):
        xp = import_cupy_with_fallback()
        ret = xp.empty(N, dtype='complex64')

        ret[:] = (
            xp.random.normal(scale=np.sqrt(self.resource), size=2 * N)
            .astype('float32')
            .view('complex64')
        )

        return ret, pd.Timestamp('now')


class TDMSFileSource(NullSource):
    """returns IQ waveforms from a TDMS file"""

    resource: str = attr.value.str(default=None, help='path to the tdms file')

    def open(self):
        from nptdms import TdmsFile

        fd = TdmsFile.read(self.resource)
        header_fd, iq_fd = fd.groups()
        self.backend = dict(header_fd=header_fd, iq_fd=iq_fd)

    @property
    def master_clock_rate(self):
        return self.backend['header_fd']['IQ_samples_per_second'][0]

    @attr.method.float(
        min=0,
        cache=True,
        label='Hz',
        help='sample rate before resampling',
    )
    def backend_sample_rate(self):
        return self.master_clock_rate

    @backend_sample_rate.setter
    def _(self, value):
        if value != self.master_clock_rate:
            raise ValueError(
                f'file sample rate must match capture ({self.master_clock_rate})'
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

    def _read_stream(self, N):
        xp = import_cupy_with_fallback()

        size = int(self.backend['header_fd']['total_samples'][0])
        ref_level = self.backend['header_fd']['reference_level_dBm'][0]

        if size < N:
            raise ValueError(
                f'requested {N} samples but file capture length is {size} samples'
            )

        scale = 10 ** (float(ref_level) / 20.0) / np.iinfo(xp.int16).max
        i, q = self.backend['iq_fd'].channels()
        iq = xp.empty((2 * N,), dtype=xp.int16)
        iq[::2] = i[:N]
        iq[1::2] = q[:N]

        samples = (iq * xp.float32(scale)).view('complex64').copy()
        return samples, pd.Timestamp('now')
