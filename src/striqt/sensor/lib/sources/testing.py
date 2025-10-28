"""Fake radios for testing"""

from __future__ import annotations

import numbers
from pathlib import Path
import typing

from . import base, null
from .. import specs, util
from striqt.analysis import Capture, io, simulated_awgn


if typing.TYPE_CHECKING:
    import numpy as np
else:
    np = util.lazy_import('numpy')

FormatType = typing.Literal['auto'] | typing.Literal['mat'] | typing.Literal['tdms']


def lo_shift_tone(inds, radio: base.SourceBase, xp, lo_offset=None):
    if lo_offset is None:
        resampler_design = base.design_capture_resampler(
            radio.get_setup_spec().base_clock_rate, radio.get_capture_spec()
        )
        lo_offset = resampler_design['lo_offset']
    phase_scale = (2j * np.pi * lo_offset) / radio._capture.backend_sample_rate
    return xp.exp(phase_scale * inds).astype('complex64')


class TestSourceBase(null.NullSource):
    def _read_stream(
        self, buffers, offset, count, timeout_sec=None, *, on_overflow='except'
    ) -> tuple[int, int]:
        ports = self.port()
        if isinstance(ports, numbers.Number):
            ports = (ports,)

        for port, buf in zip(ports, buffers):
            values = self.get_waveform(
                count,
                self._samples_elapsed,
                port=port,
                xp=getattr(self, 'xp', np),
            )
            buf[offset : (offset + count)] = values

        return super()._read_stream(
            buffers, offset, count, timeout_sec=timeout_sec, on_overflow=on_overflow
        )

    def get_waveform(self, count, start_index: int, *, port: int = 0, xp):
        raise NotImplementedError

    def _sync_time_source(self):
        self._sync_time_ns = round(1_000_000_000 * self._samples_elapsed)


class SingleToneCapture(specs.RadioCapture):
    frequency_offset: specs.Annotated[float, specs.meta('Input tone ', 'Hz')] = 0
    snr: typing.Optional[
        specs.Annotated[float, specs.meta('SNR with added noise ', 'dB', gt=0)]
    ] = None


class SingleToneSource(base.bind_schema(TestSourceBase, capture_cls=SingleToneCapture)):
    def get_waveform(self, count, start_index: int, *, port: int = 0, xp=np):
        capture = self.get_capture_spec()
        fs = capture.backend_sample_rate
        i = xp.arange(start_index, count + start_index, dtype='int64')

        lo = lo_shift_tone(i, self, xp)

        phi = (2 * np.pi * capture.frequency_offset) / fs * i + np.pi / 2
        ret = lo * xp.exp(1j * phi)
        ret = ret.astype(self._setup.transport_dtype)

        if capture.snr is not None:
            power = 10 ** (-self.snr / 10)
            noise = simulated_awgn(
                capture.replace(sample_rate=fs),
                xp=xp,
                seed=0,
                power_spectral_density=power,
            )
            noise = noise[i % noise.size]
            ret += noise

        return ret


class DiracDeltaCapture(specs.RadioCapture):
    time: specs.Annotated[
        float, specs.meta('pulse start time relative to the start of the waveform', 's')
    ] = 0
    power: typing.Optional[
        specs.Annotated[
            float,
            specs.meta('instantaneous power level of the impulse function', 'dB', gt=0),
        ]
    ] = 0


class DiracDeltaSource(base.bind_schema(TestSourceBase, capture_cls=DiracDeltaCapture)):
    def get_waveform(self, count, start_index: int, *, port: int = 0, xp=np):
        capture = self.get_capture_spec()
        abs_pulse_index = round(capture.time * self._capture.backend_sample_rate)
        rel_pulse_index = abs_pulse_index - start_index

        ret = xp.zeros(count, dtype=self._setup.transport_dtype)

        if rel_pulse_index >= 0 and rel_pulse_index < count:
            ret[rel_pulse_index] = 10 ** (capture.power / 20)

        return ret[np.newaxis,]


class SawtoothCapture(specs.RadioCapture):
    period: specs.Annotated[
        float,
        specs.meta('pulse start time relative to the start of the waveform', 's', ge=0),
    ] = 0.01
    power: typing.Optional[
        specs.Annotated[
            float,
            specs.meta('instantaneous power level of the impulse function', 'dB', gt=0),
        ]
    ] = 0.0


class SawtoothSource(base.bind_schema(TestSourceBase, capture_cls=SawtoothCapture)):
    def get_waveform(self, count, start_index: int, *, port: int = 0, xp=np):
        capture = self.get_capture_spec()

        ret = xp.empty(count, dtype='complex64')
        ii = xp.arange(start_index, count + start_index, dtype='uint64')
        t = ii / capture.backend_sample_rate
        magnitude = 2 * np.sqrt(capture.power)
        ret.real[:] = (t % capture.period) * (magnitude / capture.period)
        ret.imag[:] = 0
        return ret


class NoiseCapture(specs.RadioCapture):
    power_spectral_density: specs.Annotated[
        float, specs.meta('noise total channel power', 'mW/Hz', ge=0)
    ] = 1e-17


class NoiseSource(base.bind_schema(TestSourceBase, capture_cls=NoiseCapture)):
    def get_waveform(self, count, start_index: int, *, port: int = 0, xp=np):
        capture = self.get_capture_spec()

        backend_capture = Capture(
            duration=(count + start_index) / capture.backend_sample_rate,
            sample_rate=capture.backend_sample_rate,
            analysis_bandwidth=capture.analysis_bandwidth,
        )

        x = simulated_awgn(
            backend_capture,
            xp=xp,
            seed=0,
            power_spectral_density=capture.power_spectral_density,
        )
        # x /= np.sqrt(self._capture.backend_sample_rate / self.sample_rate())

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


class TDMSFileSetup(specs.RadioSetup):
    path: specs.Annotated[Path, specs.meta('path to the tdms file')] | None = None


class TDMSFileSource(
    base.bind_schema(
        TestSourceBase, setup_cls=TDMSFileSetup, capture_cls=SawtoothCapture
    )
):
    """returns IQ waveforms from a TDMS file"""

    def _connect(self, setup: TDMSFileSetup):
        from nptdms import TdmsFile

        fd = TdmsFile.read(setup.path)
        header_fd, iq_fd = fd.groups()
        self._handle = dict(header_fd=header_fd, iq_fd=iq_fd)

    def _prepare_capture(self, capture: specs.RadioCapture) -> specs.RadioCapture:
        fc = self._handle['header_fd']['carrier_frequency'][0]
        if capture.center_frequency is not None:
            self._logger.warning(
                f'center_frequency ignored, using {fc / 1e6} MHz from file'
            )

        mcr = self._setup.base_clock_rate
        if capture.backend_sample_rate is not None:
            self._logger.warning(
                f'backend_sample_rate ignored, using {mcr / 1e6} MHz from file'
            )

        return capture.replace(center_frequency=fc, backend_sample_rate=mcr)

    @property
    def base_clock_rate(self):
        return self._handle['header_fd']['IQ_samples_per_second'][0]

    def get_waveform(
        self, count: int, offset: int, *, channel: int = 0, xp=np, dtype='complex64'
    ):
        size = int(self._handle['header_fd']['total_samples'][0])
        ref_level = self._handle['header_fd']['reference_level_dBm'][0]

        if size < count:
            raise ValueError(
                f'requested {count} samples but file capture length is {size} samples'
            )

        scale = 10 ** (float(ref_level) / 20.0) / np.iinfo(xp.int16).max
        i, q = self._handle['iq_fd'].channels()
        iq = xp.empty((2 * count,), dtype=xp.int16)
        iq[offset * 2 :: 2] = xp.asarray(i[offset : count + offset])
        iq[1 + offset * 2 :: 2] = xp.asarray(q[offset : count + offset])

        float_dtype = np.finfo(np.dtype(dtype)).dtype

        return (iq * float_dtype(scale)).view(dtype).copy()


class FileSetup(specs.RadioSetup):
    path: specs.Annotated[Path, specs.meta('path to the waveform data file')] | None = (
        None
    )
    file_format: specs.Annotated[
        FormatType, specs.meta('data format or auto to guess by extension')
    ] = 'auto'
    file_metadata: specs.Annotated[
        dict, specs.meta('any capture fields not included in the file')
    ] = {}
    loop: specs.Annotated[
        bool, specs.meta('whether to loop the file to create longer IQ waveforms')
    ] = False


class FileCapture(specs.RadioCapture, forbid_unknown_fields=True, cache_hash=True):
    """Capture specification read from a file, with support for None sentinels"""

    # RF and leveling
    center_frequency: typing.Optional[specs.CenterFrequencyType] = None
    port: typing.Optional[specs.PortType] = None
    gain: typing.Optional[specs.GainType] = None
    backend_sample_rate: typing.Optional[specs.BackendSampleRateType] = None


class FileSource(
    base.bind_schema(TestSourceBase, setup_cls=FileSetup, capture_cls=FileCapture)
):
    """returns IQ waveforms from a file"""

    _file_capture: FileCapture | None = None
    _file_stream = None

    @property
    def base_clock_rate(self):
        return self._file_capture.backend_sample_rate

    def _connect(
        self,
        setup: FileSetup,
    ) -> specs.RadioSetup:
        if self._file_stream is not None:
            self.close()

        meta = self.file_metadata

        self._file_stream = io.open_bare_iq(
            setup.path,
            format=setup.file_format,
            num_rx_ports=self.source_info.num_rx_ports,
            dtype='complex64',
            xp=self.get_array_namespace(),
            loop=setup.loop,
            **meta,
        )

        fields = self._file_stream.get_capture_fields()
        self._file_capture = specs.FileSourceCapture.fromdict(fields)

    def _prepare_capture(self, capture: specs.RadioCapture):
        for field in ('center_frequency', 'port', 'gain', 'backend_sample_rate'):
            file_value = getattr(self._file_capture, field)
            if getattr(capture, field) not in (None, file_value):
                raise ValueError(
                    f'capture field {field!r} must be {file_value!r} or None'
                )

        super()._prepare_capture(self._file_capture)
        self._file_stream.seek(0)
        return self._file_capture

    def close(self):
        if self._file_stream is not None:
            self._file_stream.close()

    def get_waveform(
        self, count: int, offset: int, *, port: int = 0, xp=np, dtype='complex64'
    ):
        if self._file_stream is None:
            raise RuntimeError('call setup() before reading samples')
        self._file_stream.seek(offset - self._sample_start_index)
        ret = self._file_stream.read(count)
        assert ret.shape[1] == count
        return ret.copy()


class ZarrFileSetup(specs.RadioSetup):
    path: specs.Annotated[Path, specs.meta('path to the waveform data file')] | None = (
        None
    )
    select: specs.Annotated[
        dict, specs.meta('dictionary to select in the data as .sel(**select)')
    ] = {}


class ZarrIQSource(
    base.bind_schema(TestSourceBase, setup_cls=ZarrFileSetup, capture_cls=FileCapture)
):
    """Emulate an SDR by drawing IQ samples from an xarray returned by
    a striqt.analysis.iq_waveform measurement.
    """

    _waveform = None

    def _read_coord(self, name):
        return np.atleast_1d(self._waveform[name])[0]

    def _connect(self, setup: ZarrFileSetup):
        """set the waveform from an xarray.DataArray containing a single capture of IQ samples"""

        waveform = io.load(setup.path).iq_waveform

        if len(self.select) > 0:
            waveform = waveform.set_xindex(list(setup.select.keys()))
            waveform = waveform.sel(**setup.select)

        if waveform.ndim != 2:
            raise ValueError('expected 2 dimensions (capture, iq_sample)')

        self._source_info = base.BaseSourceInfo(num_rx_ports=waveform.shape[0])
        self._waveform = waveform

    @property
    def base_clock_rate(self):
        return self._read_coord('sample_rate')

    def _prepare_capture(self, capture: FileCapture):
        file_capture = capture.replace(
            center_frequency=self._read_coord('center_frequency'),
            gain=self._read_coord('gain'),
            port=self._read_coord('port'),
            backend_sample_rate=self._read_coord('sample_rate'),
        )

        super()._prepare_capture(file_capture)

        return file_capture

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
