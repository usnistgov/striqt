from __future__ import annotations
from pathlib import Path
import sys
import time
import threading
import typing
from . import structs, util, xarray_ops
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import channel_analysis

if typing.TYPE_CHECKING:
    import xarray as xr
    import labbench as lb
else:
    xr = util.lazy_import('xarray')
    lb = util.lazy_import('labbench')


def _dump_captures(data: list['xarray_ops.DelayedAnalysisResult'], store, executor):
    """write the data to disk in a background process"""
    t0 = time.perf_counter()
    global _func # to prevent gc on _func
    def _func():
        # wait until now to do CPU-intensive xarray Dataset packaging
        # in order to leave cycles free for acquisition and analysis
        ds_seq = (r.to_xarray() for r in data)

        y = xr.concat(ds_seq, xarray_ops.CAPTURE_DIM)
        channel_analysis.dump(store, y)
        return time.perf_counter() - t0

    return executor.submit(_func)


class SinkBase:
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
        self._future = None
        self._lock = threading.RLock()
        self._pending_data: list['xr.Dataset'] = []
        context = multiprocessing.get_context('fork')
        self._executor = ProcessPoolExecutor(1, mp_context=context)

    def pop(self) -> list['xr.Dataset']:
        with self._lock:
            result = self._pending_data
            self._pending_data: list['xr.Dataset'] = []
        return result

    def __enter__(self):
        self.open()
        self._executor.__enter__()
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        try:
            self.flush()
        finally:
            self._executor.__exit__(*sys.exc_info())

        print('close!')
        

    def append(self, capture_data: 'xr.Dataset'):
        if capture_data is None:
            return
        else:
            with self._lock:
                self._pending_data.append(capture_data)

    def open(self):
        raise NotImplementedError

    def flush(self):
        print('no _future')
        if self._future is not None:
            time_elapsed = self._future.result(timeout=30)
            lb.logger.info(f'flush time elapsed: {time_elapsed:0.2f} s')
        self._future = None


class ZarrSinkBase(SinkBase):
    def open(self):
        if self.store is not None:
            return
        
        if self.store_backend == 'directory':
            fixed_path = Path(self.output_path).with_suffix('.zarr')
        elif self.store_backend == 'zip':
            fixed_path = Path(self.output_path).with_suffix('.zarr.zip')
        else:
            raise ValueError(f'unsupported store type {self.store_backend!r}')

        fixed_path.parent.mkdir(parents=True, exist_ok=True)

        self.store = channel_analysis.open_store(
            fixed_path, mode='w' if self.force else 'a'
        )

    def close(self):
        super().close()
        self.store.close()
        self.store = None


class CaptureAppender(ZarrSinkBase):
    """concatenates the data from each capture and dumps to a zarr data store"""

    def flush(self):
        super().flush()

        data_list = self.pop()

        if len(data_list) == 0:
            return

        self._future = _dump_captures(data_list, self.store, self._executor)
        # with lb.stopwatch('build dataset'):
        #     data_captures = xr.concat(data_list, xarray_ops.CAPTURE_DIM)

        # with lb.stopwatch('dump data'):
        #     channel_analysis.dump(self.store, data_captures)


class SpectrogramTimeAppender(ZarrSinkBase):
    def open(self):
        if 'spectrogram' not in self.sweep_spec.channel_analysis:
            raise ValueError(
                'channel_analysis must include "spectrogram" to append on spectrogram time axis'
            )

        super().open()

    def flush(self):
        super().flush()

        data_list = self.pop()

        if len(data_list) == 0:
            return
        
        with lb.stopwatch('build dataset'):
            by_spectrogram = xarray_ops.concat_time_dim(data_list, 'spectrogram_time')

        with lb.stopwatch('dump data'):
            channel_analysis.dump(by_spectrogram, data_list, compression=False)
