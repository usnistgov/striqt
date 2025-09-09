from __future__ import annotations
from pathlib import Path
import typing
from . import captures, datasets, specs, util
from concurrent.futures import ThreadPoolExecutor

from striqt import analysis

if typing.TYPE_CHECKING:
    import xarray as xr
    import labbench as lb
else:
    xr = util.lazy_import('xarray')
    lb = util.lazy_import('labbench')


class SinkBase:
    """intake acquisitions one at a time, and parcel data store"""

    def __init__(
        self,
        sweep_spec: specs.Sweep | str | Path,
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
        self._pending_data: list['xr.Dataset'] = []
        self._executor = ThreadPoolExecutor(1)
        self._group_sizes = captures.concat_group_sizes(sweep_spec.captures, min_size=2)

    def pop(self) -> list['xr.Dataset']:
        result = self._pending_data
        self._pending_data = []
        return result

    def submit(self, func, *args, **kws):
        self.wait()
        self._future = self._executor.submit(func, *args, **kws)

    def __enter__(self):
        self.open()
        self._executor.__enter__()
        return self

    def __exit__(self, *exc_info):
        self.close(*exc_info)

    def close(self, *exc_info):
        try:
            self.wait()
        finally:
            self._executor.__exit__(*exc_info)

    def append(self, capture_data: 'xr.Dataset' | None, capture: specs.RadioCapture):
        if capture_data is None:
            return

        self._pending_data.append(capture_data)

    def open(self):
        raise NotImplementedError

    def flush(self):
        pass

    def wait(self):
        if self._future is None:
            return
        self._future.result()
        self._future = None


class NullSink(SinkBase):
    def open(self):
        pass

    def close(self, *exc_info):
        pass

    def flush(self):
        pass

    def append(self, capture_data: 'xr.Dataset' | None, capture: specs.RadioCapture):
        pass

    def wait(self):
        pass


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

        self.store = analysis.open_store(fixed_path, mode='w' if self.force else 'a')

    def close(self, *exc_info):
        super().close(*exc_info)

        if getattr(self.store, '_is_open', True):
            self.store.close()

    def get_root_path(self):
        if hasattr(self.store, 'path'):
            return self.store.path
        else:
            return self.store.root


class CaptureAppender(ZarrSinkBase):
    """concatenates the data from each capture and dumps to a zarr data store"""

    def append(self, capture_data: 'xr.Dataset' | None, capture: specs.RadioCapture):
        super().append(capture_data, capture)

        if len(self._pending_data) == self._group_sizes[0]:
            self.flush()
            self._group_sizes.pop(0)
        else:
            util.get_logger('sink').debug('queued')

    def flush(self):
        self.wait()
        data_list = self.pop()

        if len(data_list) == 0:
            return

        self.submit(self._flush_thread, data_list)

    def _flush_thread(self, data_list):
        with util.stopwatch(
            'merge dataset', 'sink', logger_level=util.PERFORMANCE_INFO
        ):
            dataset = xr.concat(data_list, datasets.CAPTURE_DIM)

        with util.stopwatch(
            f'write {self.get_root_path()}', 'sink', logger_level=util.PERFORMANCE_INFO
        ):
            analysis.dump(
                self.store, dataset, max_threads=self.sweep_spec.output.max_threads
            )


class SpectrogramTimeAppender(ZarrSinkBase):
    def open(self):
        if 'spectrogram' not in self.sweep_spec.analysis:
            raise ValueError(
                '"analysis" spec must include "spectrogram" to append on spectrogram time axis'
            )

        super().open()

    def append(self, capture_data: 'xr.Dataset' | None, capture: specs.RadioCapture):
        super().append(capture_data, capture)

        if len(self._pending_data) == self._group_sizes[0]:
            self.flush()
            self._group_sizes.pop(0)

    def flush(self):
        self.wait()
        data_list = self.pop()

        if len(data_list) == 0:
            return

        self.submit(self._flush_thread, data_list)

    def _flush_thread(self, data_list):
        with util.stopwatch(
            'build dataset', 'sink', logger_level=util.PERFORMANCE_INFO
        ):
            by_spectrogram = datasets.concat_time_dim(data_list, 'spectrogram_time')

        with util.stopwatch('dump data', 'sink', logger_level=util.PERFORMANCE_INFO):
            analysis.dump(
                by_spectrogram,
                data_list,
                compression=False,
                max_threads=self.sweep_spec.output.max_threads,
            )
