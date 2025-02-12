"""Fake radios for testing"""

import functools
import numbers
from pathlib import Path
import typing

from . import base
from .null import NullSource

import labbench as lb
from labbench import paramattr as attr


if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import channel_analysis
    import xarray as xr
else:
    np = lb.util.lazy_import('numpy')
    pd = lb.util.lazy_import('pandas')
    channel_analysis = lb.util.lazy_import('channel_analysis')
    xr = lb.util.lazy_import('xarray')


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
        channels = self.channel()
        if isinstance(channels, numbers.Number):
            channels = (channels,)

        for channel, buf in zip(channels, buffers):
            values = self.get_waveform(
                count,
                self._samples_elapsed,
                channel=channel,
                xp=getattr(self, 'xp', np),
            )
            buf[offset : (offset + count)] = values

        return super()._read_stream(
            buffers, offset, count, timeout_sec=timeout_sec, on_overflow=on_overflow
        )

    def get_waveform(self, count, start_index: int, *, channel: int = 0, xp):
        raise NotImplementedError


class SingleToneSource(TestSource):
    resource: float = attr.value.float(
        default=0, help='baseband frequency of the tone to generate', label='Hz'
    )

    snr: float = attr.value.float(
        None, label='dB', help='add circular white noise to achieve the specified SNR'
    )

    def get_waveform(self, count, start_index: int, *, channel: int = 0, xp=np):
        i = xp.arange(start_index, count + start_index, dtype='uint64')
        f_cw = self.resource
        fs = self.backend_sample_rate()
        lo = lo_shift_tone(i, self, xp)

        phi = (2 * np.pi * f_cw) / fs * i + np.pi / 2
        ret = lo * xp.exp(1j * phi)
        ret = ret.astype('complex64')

        if self.snr is not None:
            capture = channel_analysis.Capture(duration=self.duration, sample_rate=fs)
            noise = cached_noise(capture, xp=xp, power=10 ** (-self.snr / 10))
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


class TDMSFileSource(TestSource):
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
                f'center frequency ignored, using {actual / 1e6} MHz from file'
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


class ZarrIQSource(TestSource):
    """SDR emulator that draws IQ samples from an xarray returned by
    channel_analysis.iq_waveform, or from a yaml file defined to
    save iq_waveform.
    """

    path: Path = attr.value.Path(help='path to zarr file')
    select: dict = attr.value.dict(
        default={}, help='dictionary to select in the data as .sel(**select)'
    )

    _waveform = None

    def _read_coord(self, name):
        return np.atleast_1d(self._waveform[name])[0]

    def open(self):
        """set the waveform from an xarray.DataArray containing a single capture of IQ samples"""

        waveform = channel_analysis.load(self.path).iq_waveform

        if len(self.select) > 0:
            waveform = waveform.set_xindex(list(self.select.keys()))
            waveform = waveform.sel(**self.select)

        if waveform.ndim != 2:
            raise ValueError('expected 2 dimensions (capture, iq_sample)')

        self.rx_channel_count = waveform.shape[0]

        self._waveform = waveform

    @property
    def base_clock_rate(self):
        return self._read_coord('sample_rate')

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
        return self._read_coord('center_frequency')

    @center_frequency.setter
    def _(self, value):
        actual = self.center_frequency()
        if value != actual:
            self._logger.warning(
                f'center frequency ignored, using {actual / 1e6} MHz from file'
            )

    def _read_stream(
        self, buffers, offset, count, timeout_sec=None, *, on_overflow='except'
    ) -> tuple[int, int]:
        iq, _ = super()._read_stream(
            buffers, offset, count, timeout_sec=timeout_sec, on_overflow=on_overflow
        )

        if offset == 0:
            time_ns = int(self._waveform.start_time[0].values)
        else:
            time_ns = 0

        return iq, time_ns

    def get_waveform(
        self, count: int, offset: int, *, channel: int = 0, xp=np, dtype='complex64'
    ):
        iq_size = self._waveform.shape[1]

        if iq_size < count + offset:
            raise ValueError(
                f'requested {count} samples but file capture length is {iq_size} samples'
            )

        if channel > self._waveform.shape[0]:
            raise ValueError(
                f'requested channel exceeds data channel count of {self._waveform.shape[0]}'
            )

        iq = self._waveform.values[channel, offset : count + offset]

        if dtype is None or self._waveform.dtype == dtype:
            return iq.copy()
        else:
            return iq.astype(dtype)
