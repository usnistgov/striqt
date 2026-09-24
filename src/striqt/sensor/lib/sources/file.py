"""Fake radios for testing"""

from __future__ import annotations as __

import functools
from pathlib import Path
from typing import Any, Literal, overload, TYPE_CHECKING

from ... import specs

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis.lib.util import np, xr

from . import base, buffers
from ..typing import PS, PC

if TYPE_CHECKING:
    from types import ModuleType

    from striqt.waveform.lib.typing import DTypeLike

    from ..typing import Array, FileStream


def _split_preroll(start_index: int, count: int) -> tuple[int, int, int]:
    """(zero-fill count, first file index, file sample count) for a read request.

    File index 0 is corrected sample 0, so the part of the request before it is
    filled with zeros rather than read.
    """
    fill = min(max(-start_index, 0), count)
    return fill, max(start_index, 0), count - fill


class TDMSSource(base.VirtualSource[specs.TDMSSource, specs.FileCapture]):
    """a source of IQ waveforms from a TDMS file"""

    _file_info: specs.FileAcquisitionInfo

    def __init__(self, spec):
        super().__init__(spec)

        try:
            from nptdms import TdmsFile  # pyright: ignore # pyrefly: ignore
        except ImportError:
            raise ImportError('install nptdms to open TDMS files')

        fd = TdmsFile.read(spec.path)
        header_fd, iq_fd = fd.groups()
        self._handle = dict(header_fd=header_fd, iq_fd=iq_fd)

        self._file_info = specs.FileAcquisitionInfo(
            backend_sample_rate=header_fd['IQ_samples_per_second'][0],
            center_frequency=header_fd['carrier_frequency'][0],
        )

    def get_id(self) -> str:
        return str(self.setup_spec.path)

    def get_info(self) -> specs.SourceInfo:
        return specs.structs.SourceInfo(num_rx_ports=1)

    def get_waveform(
        self,
        count: int,
        start_index: int,
        *,
        port: int = 0,
        xp: ModuleType,
        dtype: DTypeLike = 'complex64',
    ) -> Array:
        size = int(self._handle['header_fd']['total_samples'][0])
        ref_level = self._handle['header_fd']['reference_level_dBm'][0]
        fill, file_start, file_count = _split_preroll(start_index, count)

        if size < file_start + file_count:
            raise ValueError(
                f'requested {count} samples but file capture length is {size} samples'
            )

        scale = 10 ** (float(ref_level) / 20.0) / np.iinfo(xp.int16).max
        i, q = self._handle['iq_fd'].channels()
        file_span = slice(file_start, file_start + file_count)
        iq = xp.zeros((2 * count,), dtype=xp.int16)
        iq[2 * fill :: 2] = xp.asarray(i[file_span])
        iq[2 * fill + 1 :: 2] = xp.asarray(q[file_span])

        float_dtype = np.finfo(np.dtype(dtype)).dtype

        return (iq * float_dtype.type(scale)).view(dtype).copy()

    def package_iq(
        self,
        iq: 'specs.AcquiredIQ',
        samples: Array,
        time_ns: int | None,
    ) -> 'specs.AcquiredIQ':
        iq.info = self._file_info
        return iq

    def get_resampler(self, capture: specs.FileCapture) -> sw.ResamplerDesign:
        from ..compute import design_resampler

        return design_resampler(
            capture,
            master_clock_rate=self.setup_spec.master_clock_rate,
            backend_sample_rate=self._file_info.backend_sample_rate,
        )

    def close(self) -> None:
        pass


class MATSource(base.VirtualSource[specs.MATSource, specs.FileCapture]):
    """returns IQ waveforms from a .mat file"""

    _file_info: specs.FileAcquisitionInfo
    _file_stream: FileStream

    def __init__(self, spec):
        super().__init__(spec)

        meta = spec.file_metadata or {}

        if not Path(spec.path).exists():
            raise IOError(f'file {str(spec.path)!r} does not exist')

        self._file_stream = sa.io.open_bare_iq(
            spec.path,
            format=spec.file_format,
            dtype='complex64',
            xp=buffers.get_array_namespace(spec.array_backend),
            loop=spec.loop,
            backend_sample_rate=spec.master_clock_rate,
            **({} if spec.key is None else {'key': spec.key}),
            **meta,
        )

        fields = self._file_stream.get_capture_fields()
        self._file_info = specs.FileAcquisitionInfo.from_dict(fields)
        self._file_stream.seek(0)

    def get_info(self):
        return specs.structs.SourceInfo(num_rx_ports=None)

    def get_id(self):  # pyright: ignore
        return str(self.setup_spec.path)

    def arm(self, capture):
        super().arm(capture)
        if self.setup_spec.loop:
            self._file_stream.seek(0)

    def close(self):
        self._file_stream.close()

    def get_waveform(
        self,
        count: int,
        start_index: int,
        *,
        port: int = 0,
        xp: ModuleType,
        dtype: DTypeLike = 'complex64',
    ) -> Array:
        fill, file_start, file_count = _split_preroll(start_index, count)

        # at least one sample, because the stream only reveals its port count by
        # returning data
        self._file_stream.seek(file_start)
        ret = self._file_stream.read(max(file_count, 1))
        ret = ret[:, :file_count]

        if fill == 0:
            return ret.copy()

        iq = sw.array_namespace(ret).zeros((ret.shape[0], count), dtype=ret.dtype)
        iq[:, fill:] = ret
        return iq

    def package_iq(
        self,
        iq: 'specs.AcquiredIQ',
        samples: Array,
        time_ns: int | None,
    ) -> 'specs.AcquiredIQ':
        iq.info = self._file_info
        return iq

    def get_resampler(self, capture) -> sw.ResamplerDesign:
        from ..compute import design_resampler

        return design_resampler(
            capture,
            master_clock_rate=self._file_info.backend_sample_rate,
            backend_sample_rate=self._file_info.backend_sample_rate,
        )


class ZarrIQSource(base.VirtualSource[specs.ZarrIQSource, specs.FileCapture]):
    """a sources of IQ samples from iq_waveform variables in a zarr store"""

    _waveform: 'xr.DataArray'
    _capture_info: specs.FileAcquisitionInfo

    def __init__(self, spec):
        """set the waveform from an xarray.DataArray containing a single capture of IQ samples"""
        super().__init__(spec)

        waveform = sa.io.load(spec.path).iq_waveform

        if len(spec.select) > 0:
            waveform = waveform.set_xindex(list(spec.select.keys()))
            waveform = waveform.sel(**spec.select)

        if waveform.ndim != 2:
            raise ValueError('expected 2 dimensions (capture, iq_sample)')

        self._waveform = waveform

    def get_id(self):  # pyright: ignore
        return str(self.setup_spec.path)

    def get_info(self):  # pyright: ignore
        return specs.structs.SourceInfo(num_rx_ports=self._waveform.shape[0])

    def close(self) -> None:
        pass

    def get_resampler(self, capture) -> sw.ResamplerDesign:
        from ..compute import design_resampler

        return design_resampler(capture, self._read_coord('sample_rate'))

    def arm(self, capture):
        super().arm(capture)

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

    def read(
        self,
        buffers,
        offset,
        count,
        timeout_sec=None,
        *,
        on_overflow: specs.types.OnOverflow = 'except',
    ) -> tuple[int, int]:
        assert self._waveform is not None
        iq, _ = super().read(
            buffers, offset, count, timeout_sec=timeout_sec, on_overflow=on_overflow
        )

        if offset == 0:
            time_ns = int(self._waveform.start_time[0].data)
        else:
            time_ns = 0

        return iq, time_ns

    def get_waveform(
        self,
        count: int,
        start_index: int,
        *,
        port: int = 0,
        xp: ModuleType,
        dtype: DTypeLike = 'complex64',
    ) -> Array:
        assert self._waveform is not None
        iq_size = self._waveform.shape[1]
        fill, file_start, file_count = _split_preroll(start_index, count)

        if iq_size < file_start + file_count:
            raise ValueError(
                f'requested {file_start + file_count} samples but file capture length is {iq_size} samples'
            )

        if port >= self._waveform.shape[0]:
            raise ValueError(
                f'requested channel exceeds data channel count of {self._waveform.shape[0]}'
            )

        out_dtype = self._waveform.dtype if dtype is None else dtype
        iq = xp.zeros((1, count), dtype=out_dtype)
        iq[:, fill:] = xp.asarray(
            self._waveform.data[[port], file_start : file_start + file_count]
        )
        return iq

    def package_iq(
        self,
        iq: 'specs.AcquiredIQ',
        samples: Array,
        time_ns: int | None,
    ) -> 'specs.AcquiredIQ':
        iq.info = self._capture_info
        return iq

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
