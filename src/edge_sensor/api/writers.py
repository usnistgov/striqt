from __future__ import annotations
from pathlib import Path
import typing
from . import io, structs, util, xarray_ops

import channel_analysis

if typing.TYPE_CHECKING:
    import xarray as xr
    import labbench as lb
else:
    xr = util.lazy_import('xarray')
    lb = util.lazy_import('labbench')


class WriterBase:
    """intake acquisitions one at a time, and parcel data store"""

    def __init__(
        self,
        sweep_spec: structs.Sweep | str | Path,
        *,
        output_path: str | None = None,
        store_backend: str | None = None,
        force: bool = False,
    ):
        self.sweep_spec = sweep_spec

        if output_path is None:
            self.output_path = self.sweep_spec.output.path
        else:
            self.output_path = output_path

        if store_backend is None:
            self.store_backend = self.sweep_spec.output.store.lower()
        else:
            self.store_backend = store_backend.lower()

        self.store = None
        self.force = force

        self.clear()

    def clear(self):
        self.pending_data: list['xr.Dataset'] = []

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.flush()

    def append(self, capture_data: 'xr.Dataset'):
        if capture_data is None:
            return
        else:
            self.pending_data.append(capture_data)

    def open(self):
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError


class CaptureAppender(WriterBase):
    """concatenates the data from each capture and dumps to a zarr data store"""

    def open(self):
        if self.store_backend == 'directory':
            fixed_path = Path(self.output_path).with_suffix('.zarr')
        elif self.store_backend == 'zip':
            fixed_path = Path(self.output_path).with_suffix('.zarr.zip')
        else:
            raise ValueError(f'unsupported store type {self.store_backend!r}')

        fixed_path.parent.mkdir(parents=True, exist_ok=True)

        self.store = io.open_store(fixed_path, mode='w' if self.force else 'a')

    def close(self):
        super().close()
        self.store.close()

    def flush(self):
        if len(self.pending_data) == 0:
            return

        with lb.stopwatch('build dataset'):
            data_captures = xr.concat(self.pending_data, xarray_ops.CAPTURE_DIM)

        with lb.stopwatch('dump data'):
            channel_analysis.dump(self.store, data_captures)

        self.clear()
