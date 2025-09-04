"""Fake radios for testing"""

from __future__ import annotations
import numbers
from pathlib import Path
import typing

from . import base, method_attr
from .null import NullSource
from .. import specs

import labbench as lb
from labbench import paramattr as attr


if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import striqt.analysis as striqt_analysis
    import xarray as xr
else:
    np = lb.util.lazy_import('numpy')
    pd = lb.util.lazy_import('pandas')
    striqt_analysis = lb.util.lazy_import('striqt.analysis')
    xr = lb.util.lazy_import('xarray')


def lo_shift_tone(inds, radio: base.SourceBase, xp, lo_offset=None):
    if lo_offset is None:
        resampler_design = base.design_capture_resampler(
            radio.base_clock_rate, radio.get_capture_struct()
        )
        lo_offset = resampler_design['lo_offset']
    phase_scale = (2j * np.pi * lo_offset) / radio.backend_sample_rate()
    return xp.exp(phase_scale * inds).astype('complex64')


class TestSource(NullSource):
    def _read_stream(
        self, buffers, offset, count, timeout_sec=None, *, on_overflow='except'
    ) -> tuple[int, int]:
        ports = self.port()
        if isinstance(ports, numbers.Number):
            ports = (ports,)

        for channel, buf in zip(ports, buffers):
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

    def sync_time_source(self):
        self._sync_time_ns = round(1_000_000_000 * self._samples_elapsed)


class SingleToneSource(TestSource):
    baseband_frequency: float = attr.value.float(
        default=0, help='baseband frequency of the tone to generate', label='Hz'
    )

    snr: float = attr.value.float(
        None, label='dB', help='add circular white noise to achieve the specified SNR'
    )

    base_clock_rate: float = attr.value.float(
        default=125e6, min=0, help='base clock rate'
    )

    def get_waveform(self, count, start_index: int, *, channel: int = 0, xp=np):
        i = xp.arange(start_index, count + start_index, dtype='int64')
        f_cw = self.baseband_frequency
        fs = self.backend_sample_rate()
        lo = lo_shift_tone(i, self, xp)

        phi = (2 * np.pi * f_cw) / fs * i + np.pi / 2
        ret = lo * xp.exp(1j * phi)
        ret = ret.astype(self._transport_dtype)

        if self.snr is not None:
            capture = striqt_analysis.Capture(duration=self.duration, sample_rate=fs)
            power = 10 ** (-self.snr / 10)
            noise = striqt_analysis.simulated_awgn(
                capture, xp=xp, seed=0, power_spectral_density=power
            )
            noise = noise[i % noise.size]
            ret += noise

        return ret


class DiracDeltaSource(TestSource):
    time: float = attr.value.float(
        default=0,
        help='pulse start time relative to the start of the waveform',
        label='s',
    )

    power: float = attr.value.float(
        0, label='dB', help='instantaneous power level of the discretized delta'
    )

    base_clock_rate: float = attr.value.float(
        default=125e6, min=0, help='base clock rate'
    )

    def get_waveform(self, count, start_index: int, *, channel: int = 0, xp=np):
        abs_pulse_index = round(self.time * self.backend_sample_rate())
        rel_pulse_index = abs_pulse_index - start_index

        ret = xp.zeros(count, dtype=self._transport_dtype)

        if rel_pulse_index >= 0 and rel_pulse_index < count:
            ret[rel_pulse_index] = 10 ** (self.power / 20)

        return ret[np.newaxis,]


class SawtoothSource(TestSource):
    period: float = attr.value.float(
        default=0.01,
        min=0,
        help='sawtooth period',
        label='s',
    )

    power: float = attr.value.float(
        default=0.01,
        min=0,
        help='average output RMS channel power',
        label='mW',
    )

    base_clock_rate: float = attr.value.float(
        default=125e6, min=0, help='base clock rate'
    )

    def get_waveform(self, count, start_index: int, *, channel: int = 0, xp=np):
        ret = xp.empty(count, dtype='complex64')
        period = self.period
        ii = xp.arange(start_index, count + start_index, dtype='uint64')
        t = ii / self.backend_sample_rate()
        magnitude = 2 * np.sqrt(self.power)
        ret.real[:] = (t % period) * (magnitude / period)
        ret.imag[:] = 0
        return ret


class NoiseSource(TestSource):
    power_spectral_density: float = attr.value.float(
        default=1e-17, min=0, help='noise total channel power'
    )

    base_clock_rate: float = attr.value.float(
        default=125e6, min=0, help='base clock rate'
    )

    def get_waveform(self, count, start_index: int, *, channel: int = 0, xp=np):
        duration = (count + start_index) / self.backend_sample_rate()
        capture = striqt_analysis.Capture(
            duration=duration, sample_rate=self.backend_sample_rate()
        )

        x = striqt_analysis.simulated_awgn(
            capture, xp=xp, seed=0, power_spectral_density=self.power_spectral_density
        )
        # x /= np.sqrt(self.backend_sample_rate() / self.sample_rate())

        if start_index < 0:
            pad = -start_index
            start_index = 0
            count = count - pad
        else:
            pad = 0

        ret = x[start_index : count + start_index]

        if pad:
            return xp.pad(ret, [[pad, 0]], mode='constant')
        else:
            return ret


class TDMSFileSource(TestSource):
    """returns IQ waveforms from a TDMS file"""

    path: str = attr.value.str(default=None, help='path to the tdms file')

    def open(self):
        from nptdms import TdmsFile

        fd = TdmsFile.read(self.path)
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


class FileSource(TestSource):
    """returns IQ waveforms from a file"""

    path: str = attr.value.str(default=None, help='path to the tdms file')
    file_format: str = attr.value.str('auto', only=['auto', 'mat', 'tdms'])
    file_metadata: dict = attr.value.dict(
        default={}, help='any capture fields not included in the file'
    )
    loop: bool = attr.value.bool(
        False, help='whether to loop the file to create longer IQ waveforms'
    )

    _iq_capture = specs.FileSourceCapture()

    @property
    def base_clock_rate(self):
        return self._iq_capture.sample_rate

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
        return self._iq_capture.center_frequency

    @center_frequency.setter
    def _(self, value):
        actual = self.center_frequency()
        if value != actual and np.isfinite(actual):
            self._logger.warning(
                f'center frequency ignored, using {actual / 1e6} MHz from file'
            )

    @attr.method.float(
        cache=True,
        label='dB',
        help='gain',
    )
    def gain(self):
        return self._iq_capture.gain

    @gain.setter
    def _(self, value):
        actual = self.gain()
        if value != actual and np.isfinite(actual):
            self._logger.warning(
                f'gain ignored, using {actual / 1e6} MHz from metadata'
            )

    @attr.method.float(
        cache=True,
        label='Hz',
        help='sample rate',
    )
    def sample_rate(self):
        return self.backend_sample_rate()

    @sample_rate.setter
    def _(self, value):
        pass

    @method_attr.ChannelMaybeTupleMethod(inherit=True)
    def port(self):
        return self._iq_capture.port

    @port.setter
    def _(self, value):
        actual = self.port()
        if value != actual:
            self._logger.warning(f'port ignored, using {actual / 1e6} MHz from file')

    def open(self):
        self._file_stream = None

    def setup(
        self,
        radio_setup: specs.RadioSetup = None,
        analysis=None,
        **kws: typing.Unpack[specs._RadioSetupKeywords],
    ) -> specs.RadioSetup:
        radio_setup = super().setup(radio_setup, analysis, **kws)

        if self._file_stream is not None:
            self._file_stream.close()

        meta = self.file_metadata

        self._file_stream = striqt_analysis.io.open_bare_iq(
            self.path,
            format=self.file_format,
            rx_port_count=self.rx_port_count,
            dtype='complex64',
            xp=self.get_array_namespace(),
            loop=self.loop,
            **meta,
        )

        metadata = self._file_stream.get_metadata()
        self._iq_capture = specs.FileSourceCapture.fromdict(metadata)

        return radio_setup

    def arm(self, capture=None, **capture_kws):
        if self._file_stream is None:
            raise RuntimeError('call setup() before arm()')

        if capture_kws.get('backend_sample_rate', self.base_clock_rate) not in (
            None,
            self.base_clock_rate,
        ):
            raise ValueError(f'backend_sample_rate can only be {self.base_clock_rate}')
        else:
            capture_kws = dict(capture_kws, backend_sample_rate=self.base_clock_rate)

        super().arm(capture, **capture_kws)
        self._file_stream.seek(0)

    def get_waveform(
        self, count: int, offset: int, *, channel: int = 0, xp=np, dtype='complex64'
    ):
        if self._file_stream is None:
            raise RuntimeError('call setup() before reading samples')
        self._file_stream.seek(offset - self._sample_start_index)
        ret = self._file_stream.read(count)
        assert ret.shape[1] == count
        return ret.copy()


class ZarrIQSource(TestSource):
    """Emulate an SDR by drawing IQ samples from an xarray returned by
    a striqt.analysis.iq_waveform measurement.
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

        waveform = striqt_analysis.load(self.path).iq_waveform

        if len(self.select) > 0:
            waveform = waveform.set_xindex(list(self.select.keys()))
            waveform = waveform.sel(**self.select)

        if waveform.ndim != 2:
            raise ValueError('expected 2 dimensions (capture, iq_sample)')

        self.rx_port_count = waveform.shape[0]

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
        if value != actual and np.isfinite(actual):
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
                f'requested {count + offset} samples but file capture length is {iq_size} samples'
            )

        if channel > self._waveform.shape[0]:
            raise ValueError(
                f'requested channel exceeds data channel count of {self._waveform.shape[0]}'
            )

        start = offset - self._sample_start_index

        iq = self._waveform.values[[channel], start : count + start]

        if dtype is None or self._waveform.dtype == dtype:
            return iq.copy()
        else:
            return iq.astype(dtype)
