"""Fake radios for testing"""

from __future__ import annotations as __

import functools
import typing

from ... import specs
from .. import util
from striqt.analysis.lib import io

from . import _base

if typing.TYPE_CHECKING:
    import numpy as np
    import xarray as xr
else:
    np = util.lazy_import('numpy')


@_base.bind_schema_types(specs.TDMSSource, specs.FileCapture)
class TDMSSource(
    _base.VirtualSourceBase[specs.TDMSSource, specs.FileCapture, _base._PS, _base._PC]
):
    """a source of IQ waveforms from a TDMS file"""

    _file_info: specs.FileAcquisitionInfo

    def _connect(self, spec):
        from nptdms import TdmsFile

        fd = TdmsFile.read(spec.path)
        header_fd, iq_fd = fd.groups()
        self._handle = dict(header_fd=header_fd, iq_fd=iq_fd)

        self._file_info = specs.FileAcquisitionInfo(
            backend_sample_rate=header_fd['IQ_samples_per_second'][0],
            center_frequency=header_fd['carrier_frequency'][0],
        )

    def get_waveform(
        self, count: int, offset: int, *, port: int = 0, xp, dtype='complex64'
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

        return (iq * float_dtype(scale)).view(dtype).copy()  # type: ignore

    def acquire(self, *, analysis=None, correction=True, alias_func=None):
        iq = super().acquire(
            analysis=analysis, correction=correction, alias_func=alias_func
        )
        iq.info = self._file_info
        return iq

    def get_resampler(
        self, capture: specs.FileCapture | None = None
    ) -> _base.ResamplerDesign:
        if capture is None:
            capture = self.capture_spec

        mcr = self.setup_spec.base_clock_rate
        return _base.design_capture_resampler(
            mcr, capture, backend_sample_rate=self._file_info.backend_sample_rate
        )


@_base.bind_schema_types(specs.MATSource, specs.FileCapture)
class MATSource(
    _base.VirtualSourceBase[specs.MATSource, specs.FileCapture, _base._PS, _base._PC]
):
    """returns IQ waveforms from a .mat file"""

    _file_info: specs.FileAcquisitionInfo
    _file_stream: io._FileStreamBase

    def _connect(self, spec):
        meta = spec.file_metadata

        self._file_stream = io.open_bare_iq(
            spec.path,
            format=spec.file_format,
            num_rx_ports=self.setup_spec.num_rx_ports or 1,
            dtype='complex64',
            xp=self.get_array_namespace(),
            loop=spec.loop,
            backend_sample_rate=spec.base_clock_rate,
            **meta,
        )

        fields = self._file_stream.get_capture_fields()
        self._file_info = specs.FileAcquisitionInfo.from_dict(fields)
        self._file_stream.seek(0)

    def _prepare_capture(self, capture):
        if self.setup_spec.loop:
            self._file_stream.seek(0)

    def close(self):
        if self.is_open():
            self._file_stream.close()

    def get_waveform(
        self, count: int, offset: int, *, port: int = 0, xp, dtype='complex64'
    ):
        self._file_stream.seek(offset - self._sample_start_index)
        ret = self._file_stream.read(count)
        assert ret.shape[1] == count
        return ret.copy()

    def acquire(self, *, analysis=None, correction=True, alias_func=None):
        iq = super().acquire(
            analysis=analysis, correction=correction, alias_func=alias_func
        )
        iq.info = self._file_info
        return iq

    def get_resampler(
        self, capture: specs.FileCapture | None = None
    ) -> _base.ResamplerDesign:
        if capture is None:
            capture = self.capture_spec
        mcr = self._file_info.backend_sample_rate
        fs_sdr = self._file_info.backend_sample_rate
        return _base.design_capture_resampler(mcr, capture, backend_sample_rate=fs_sdr)

    @functools.cached_property
    def info(self):
        return _base.BaseSourceInfo(num_rx_ports=self.setup_spec.num_rx_ports)

    @functools.cached_property
    def id(self):
        return str(self.setup_spec.path)


@_base.bind_schema_types(specs.ZarrIQSource, specs.FileCapture)
class ZarrIQSource(
    _base.VirtualSourceBase[specs.ZarrIQSource, specs.FileCapture, _base._PS, _base._PC]
):
    """a sources of IQ samples from iq_waveform variables in a zarr store"""

    _waveform: 'xr.DataArray'
    _capture_info: specs.FileAcquisitionInfo

    def _read_coord(self, name):
        assert self._waveform is not None
        return np.atleast_1d(self._waveform[name])[0]

    def _connect(self, spec):
        """set the waveform from an xarray.DataArray containing a single capture of IQ samples"""

        waveform = io.load(spec.path).iq_waveform

        if len(spec.select) > 0:
            waveform = waveform.set_xindex(list(spec.select.keys()))
            waveform = waveform.sel(**spec.select)

        if waveform.ndim != 2:
            raise ValueError('expected 2 dimensions (capture, iq_sample)')

        self._waveform = waveform

    @functools.cached_property
    def info(self):
        return _base.BaseSourceInfo(num_rx_ports=self._waveform.shape[0])

    def _prepare_capture(self, capture):
        super()._prepare_capture(capture)

        self._capture_info = specs.FileAcquisitionInfo(
            center_frequency=self._read_coord('center_frequency'),
            gain=self._read_coord('gain'),
            port=self._read_coord('port'),
            backend_sample_rate=self._read_coord('sample_rate'),
        )

    def get_resampler(self, capture=None) -> _base.ResamplerDesign:
        if capture is None:
            capture = self.capture_spec
        fs = self._read_coord('sample_rate')
        return _base.design_capture_resampler(fs, capture, fs)

    def _read_stream(
        self,
        buffers,
        offset,
        count,
        timeout_sec=None,
        *,
        on_overflow: specs.types.OnOverflowType = 'except',
    ) -> tuple[int, int]:
        assert self._waveform is not None
        iq, _ = super()._read_stream(
            buffers, offset, count, timeout_sec=timeout_sec, on_overflow=on_overflow
        )

        if offset == 0:
            time_ns = int(self._waveform.start_time[0].values)
        else:
            time_ns = 0

        return iq, time_ns

    def get_waveform(
        self, count: int, offset: int, *, port: int = 0, xp, dtype='complex64'
    ):
        assert self._waveform is not None
        iq_size = self._waveform.shape[1]

        if iq_size < count + offset:
            raise ValueError(
                f'requested {count + offset} samples but file capture length is {iq_size} samples'
            )

        if port > self._waveform.shape[0]:
            raise ValueError(
                f'requested channel exceeds data channel count of {self._waveform.shape[0]}'
            )

        start = offset - self._sample_start_index

        iq = self._waveform.values[[port], start : count + start]

        if dtype is None or self._waveform.dtype == dtype:
            return iq.copy()
        else:
            return iq.astype(dtype)

    def acquire(self, *, analysis=None, correction=True, alias_func=None):
        iq = super().acquire(
            analysis=analysis, correction=correction, alias_func=alias_func
        )
        iq.info = self._capture_info
        return iq
