"""Fake radios for testing"""

from __future__ import annotations as __

import functools
from typing import Any, Literal, overload, TYPE_CHECKING

from ... import specs
from .. import util

import striqt.analysis as sa
import striqt.waveform as sw

from . import base, buffers
from ..typing import PS, PC

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr
    from ..typing import FileStream
else:
    np = util.lazy_import('numpy')


@base.bind_schema_types(specs.TDMSSource, specs.FileCapture)
class TDMSSource(base.VirtualSource[specs.TDMSSource, specs.FileCapture, PS, PC]):
    """a source of IQ waveforms from a TDMS file"""

    _file_info: specs.FileAcquisitionInfo

    def _connect(self, spec):
        from nptdms import TdmsFile  # type: ignore

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
    ) -> sw.ResamplerDesign:
        if capture is None:
            capture = self.capture_spec

        mcr = self.setup_spec.master_clock_rate
        return buffers.design_resampler(
            capture, mcr, backend_sample_rate=self._file_info.backend_sample_rate
        )


@base.bind_schema_types(specs.MATSource, specs.FileCapture)
class MATSource(base.VirtualSource[specs.MATSource, specs.FileCapture, PS, PC]):
    """returns IQ waveforms from a .mat file"""

    _file_info: specs.FileAcquisitionInfo
    _file_stream: FileStream

    def _connect(self, spec):
        meta = spec.file_metadata

        self._file_stream = sa.io.open_bare_iq(
            spec.path,
            format=spec.file_format,
            dtype='complex64',
            xp=buffers.get_array_namespace(self.setup_spec.array_backend),
            loop=spec.loop,
            backend_sample_rate=spec.master_clock_rate,
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
    ) -> sw.ResamplerDesign:
        if capture is None:
            capture = self.capture_spec
        mcr = self._file_info.backend_sample_rate
        fs_sdr = self._file_info.backend_sample_rate
        return buffers.design_resampler(capture, mcr, backend_sample_rate=fs_sdr)

    @functools.cached_property
    def info(self):
        return specs.structs.BaseSourceInfo(num_rx_ports=None)

    @functools.cached_property
    def id(self):
        return str(self.setup_spec.path)


@base.bind_schema_types(specs.ZarrIQSource, specs.FileCapture)
class ZarrIQSource(base.VirtualSource[specs.ZarrIQSource, specs.FileCapture, PS, PC]):
    """a sources of IQ samples from iq_waveform variables in a zarr store"""

    _waveform: 'xr.DataArray'
    _capture_info: specs.FileAcquisitionInfo

    @overload
    def _read_coord(self, name: str, single: Literal[True] = True) -> Any:
        pass

    @overload
    def _read_coord(self, name: str, single: Literal[False] = False) -> tuple[Any, ...]:
        pass

    def _read_coord(self, name: str, single: bool = True):
        assert self._waveform is not None
        result = np.atleast_1d(self._waveform[name])
        if single:
            return result[0]
        else:
            return tuple(result.tolist())

    def _connect(self, spec):
        """set the waveform from an xarray.DataArray containing a single capture of IQ samples"""

        waveform = sa.io.load(spec.path).iq_waveform

        if len(spec.select) > 0:
            waveform = waveform.set_xindex(list(spec.select.keys()))
            waveform = waveform.sel(**spec.select)

        if waveform.ndim != 2:
            raise ValueError('expected 2 dimensions (capture, iq_sample)')

        self._waveform = waveform

    @functools.cached_property
    def info(self):
        return specs.structs.BaseSourceInfo(num_rx_ports=self._waveform.shape[0])

    def _prepare_capture(self, capture):
        super()._prepare_capture(capture)

        try:
            port = self._read_coord('port', single=False)
        except KeyError:
            # legacy files
            port = self._read_coord('channel', single=False)

        self._capture_info = specs.FileAcquisitionInfo(
            center_frequency=self._read_coord('center_frequency'),
            gain=self._read_coord('gain', single=False),
            port=port,
            backend_sample_rate=self._read_coord('sample_rate'),
        )

    def get_resampler(self, capture=None) -> sw.ResamplerDesign:
        if capture is None:
            capture = self.capture_spec
        fs = self._read_coord('sample_rate')
        return buffers.design_resampler(capture, fs)

    def _read_stream(
        self,
        buffers,
        offset,
        count,
        timeout_sec=None,
        *,
        on_overflow: specs.types.OnOverflow = 'except',
    ) -> tuple[int, int]:
        assert self._waveform is not None
        iq, _ = super()._read_stream(
            buffers, offset, count, timeout_sec=timeout_sec, on_overflow=on_overflow
        )

        if offset == 0:
            time_ns = int(self._waveform.start_time[0].data)
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

        iq = self._waveform.data[[port], start : count + start]

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

    @functools.cached_property
    def id(self):
        return str(self.setup_spec.path)
