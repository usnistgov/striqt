from __future__ import annotations

import typing
from concurrent.futures import ThreadPoolExecutor

from striqt import analysis

from . import captures, datasets, io, specs, util

if typing.TYPE_CHECKING:
    import xarray as xr
else:
    xr = util.lazy_import('xarray')


class SinkBase(typing.Generic[specs._TC]):
    """intake acquisitions one at a time, and parcel data store"""

    def __init__(
        self,
        sweep_spec: specs.Sweep[specs._TS, specs._TP, specs._TC],
        alias_func: captures.PathAliasFormatter | None = None,
        *,
        force: bool = False,
    ):
        self._spec = sweep_spec.sink
        self.captures_elapsed = 0
        self._alias_func = alias_func

        self.store = None
        self.force = force
        self._future = None
        self._pending_data: list['xr.Dataset'] = []
        self._executor = ThreadPoolExecutor(1)
        self._group_sizes = captures.concat_group_sizes(
            sweep_spec.loop_captures(), min_size=2
        )

    def pop(self) -> list['xr.Dataset']:
        result = self._pending_data
        self._pending_data = []
        self.captures_elapsed += len(result)
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

    def append(self, capture_result: datasets.DelayedDataset|None, capture: specs._TC):
        if capture_result is None:
            return

        ds = datasets.from_delayed(capture_result)
        self._pending_data.append(ds)

    def open(self):
        raise NotImplementedError

    def flush(self):
        pass

    def wait(self):
        if self._future is None:
            return
        self._future.result()
        self._future = None


class NoSink(SinkBase):
    def open(self):
        pass

    def close(self, *exc_info):
        pass

    def flush(self):
        count = self.captures_elapsed
        with util.log_capture_context(
            'sink', capture_index=count - 1, capture_count=count
        ):
            util.get_logger('sink').info(f'done')

    def append(self, capture_result, capture):
        self.captures_elapsed += 1
        if capture_result:
            datasets.from_delayed(capture_result)

    def wait(self):
        pass


class ZarrSinkBase(SinkBase[specs._TC]):
    def open(self):
        self.store = io.open_store(
            self._spec, alias_func=self._alias_func, force=self.force
        )

    def close(self, *exc_info):
        super().close(*exc_info)

        if getattr(self.store, '_is_open', True):
            self.store.close()

        path = self.get_root_path()
        count = self.captures_elapsed

        with util.log_capture_context('sink', capture_index=count - 1):
            util.get_logger('sink').info(f'closed "{str(path)}"')

    def get_root_path(self):
        if hasattr(self.store, 'path'):
            return self.store.path
        else:
            return self.store.root


class ZarrCaptureSink(ZarrSinkBase[specs._TC]):
    """concatenates the data from each capture and dumps to a zarr data store"""

    def append(self, capture_result, capture: specs._TC):
        super().append(capture_result, capture)

        if len(self._pending_data) == self._group_sizes[0]:
            self.flush()
            self._group_sizes.pop(0)
        else:
            util.get_logger('sink').debug('queued')

    def flush(self):
        super().flush()
        self.wait()
        data_list = self.pop()

        if len(data_list) == 0:
            return

        self.submit(self._flush_thread, data_list)

    def _flush_thread(self, data_list):
        with util.stopwatch('merge dataset', 'sink', threshold=0.25):
            dataset = xr.concat(data_list, datasets.CAPTURE_DIM)

        path = self.get_root_path()
        count = self.captures_elapsed
        logger = util.get_logger('sink')

        with (
            util.log_capture_context('sink', capture_index=count - 1),
            util.stopwatch(f'sync to {path}', 'sink'),
        ):
            analysis.dump(self.store, dataset, max_threads=self._spec.max_threads)

            for i in range(count - len(data_list), count):
                with util.log_capture_context('sink', capture_index=i):
                    logger.info('💾')


class SpectrogramTimeAppender(ZarrSinkBase):
    def __init__(
        self,
        sweep_spec: specs.Sweep,
        alias_func: captures.PathAliasFormatter | None = None,
        *,
        force: bool = False,
    ):
        if 'spectrogram' not in sweep_spec.analysis:
            raise ValueError(
                '"analysis" spec must include "spectrogram" to append on spectrogram time axis'
            )

        super().__init__(sweep_spec, alias_func, force=force)

    def append(
        self, capture_result, capture: specs.ResampledCapture
    ):
        super().append(capture_result, capture)

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
        with util.stopwatch('build dataset', 'sink', threshold=0.5):
            by_spectrogram = datasets.concat_time_dim(data_list, 'spectrogram_time')

        path = self.get_root_path()
        count = self.captures_elapsed
        logger = util.get_logger('sink')

        with (
            util.log_capture_context('sink', capture_index=count - 1),
            util.stopwatch(f'sync {path}', 'sink', threshold=0.5),
        ):
            analysis.dump(
                self.store,
                by_spectrogram,
                compression=False,
                max_threads=self._spec.max_threads,
            )

            for i in range(count - len(data_list), count):
                with util.log_capture_context('sink', capture_index=i):
                    logger.info('💾')
