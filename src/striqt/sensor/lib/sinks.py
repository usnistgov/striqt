from __future__ import annotations as __

__all__ = [
    'SinkBase',
    'NoSink',
    'ZarrSinkBase',
    'ZarrCaptureSink',
    'ZarrTimeAppendSink',
]

import typing as _typing

from . import compute as _compute
from . import io as _io
from . import util as _util
from .. import specs as _specs

if _typing.TYPE_CHECKING:
    import xarray as _xr
else:
    _xr = _util.lazy_import('xarray')


class SinkBase(_typing.Generic[_specs._TC]):
    """intake acquisitions one at a time, and parcel data store"""

    def __init__(
        self,
        sweep_spec: _specs.Sweep[_typing.Any, _typing.Any, _specs._TC],
        alias_func: _specs.helpers.PathAliasFormatter | None = None,
        *,
        force: bool = False,
    ):
        from concurrent.futures import ThreadPoolExecutor

        self._spec = sweep_spec.sink
        self.captures_elapsed = 0
        self._alias_func = alias_func

        self.store = None
        self.force = force
        self._future = None
        self._pending_data: list['_xr.Dataset'] = []
        self._executor = ThreadPoolExecutor(1)
        captures = _specs.helpers.loop_captures(sweep_spec)
        self._group_sizes = _specs.helpers.concat_group_sizes(captures, min_size=2)

    def pop(self) -> list['_xr.Dataset']:
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

    def append(
        self, capture_result: _compute.DelayedDataset
    ) -> '_xr.Dataset|_compute.DelayedDataset':
        if capture_result is None:
            return

        ds = _compute.from_delayed(capture_result)
        self._pending_data.append(ds)
        return ds

    def open(self) -> None:
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
        with _util.log_capture_context(
            'sink', capture_index=count - 1, capture_count=count
        ):
            _util.get_logger('sink').info(f'done')

    def append(self, capture_result) -> _compute.DelayedDataset:
        self.captures_elapsed += 1
        return capture_result

    def wait(self):
        pass


class ZarrSinkBase(SinkBase):
    def open(self):
        _util.safe_import('xarray')
        self.store = _io.open_store(
            self._spec, alias_func=self._alias_func, force=self.force
        )

    def close(self, *exc_info):
        super().close(*exc_info)

        if getattr(self.store, '_is_open', True):
            self.store.close()

        path = self.get_root_path()
        count = self.captures_elapsed

        _util.get_logger('sink').info(f'closed "{str(path)}"')

    def get_root_path(self):
        if hasattr(self.store, 'path'):
            return self.store.path
        else:
            return self.store.root


class ZarrCaptureSink(ZarrSinkBase):
    """concatenates the data from each capture and dumps to a zarr data store"""

    def append(self, capture_result: _compute.DelayedDataset):
        ret = super().append(capture_result)

        if len(self._pending_data) == self._group_sizes[0]:
            self.flush()
            self._group_sizes.pop(0)
        else:
            _util.get_logger('sink').debug('queued')

        return ret

    def flush(self):
        super().flush()
        self.wait()
        data_list = self.pop()

        if len(data_list) == 0:
            return

        self.submit(self._flush_thread, data_list)

    def _flush_thread(self, data_list):
        with _util.stopwatch('merge dataset', 'sink', threshold=0.25):
            dataset = _xr.concat(data_list, _compute.CAPTURE_DIM)

        path = self.get_root_path()
        count = self.captures_elapsed
        logger = _util.get_logger('sink')

        with (
            _util.log_capture_context('sink', capture_index=count - 1),
            _util.stopwatch(f'sync to {path}', 'sink'),
        ):
            _io.dump_data(self.store, dataset, max_threads=self._spec.max_threads)

            for i in range(count - len(data_list), count):
                with _util.log_capture_context('sink', capture_index=i):
                    logger.info('💾')


class ZarrTimeAppendSink(ZarrSinkBase):
    def __init__(
        self,
        sweep_spec: _specs.Sweep,
        alias_func: _specs.helpers.PathAliasFormatter | None = None,
        *,
        force: bool = False,
    ):
        if 'spectrogram' not in sweep_spec.analysis:
            raise ValueError(
                '"analysis" spec must include "spectrogram" to append on spectrogram time axis'
            )

        super().__init__(sweep_spec, alias_func, force=force)

    def append(self, capture_result):
        ret = super().append(capture_result)

        if len(self._pending_data) == self._group_sizes[0]:
            self.flush()
            self._group_sizes.pop(0)
        return ret

    def flush(self):
        self.wait()
        data_list = self.pop()

        if len(data_list) == 0:
            return

        self.submit(self._flush_thread, data_list)

    def _flush_thread(self, data_list):
        with _util.stopwatch('build dataset', 'sink', threshold=0.5):
            by_spectrogram = _compute.concat_time_dim(data_list, 'spectrogram_time')

        path = self.get_root_path()
        count = self.captures_elapsed
        logger = _util.get_logger('sink')

        with (
            _util.log_capture_context('sink', capture_index=count - 1),
            _util.stopwatch(f'sync {path}', 'sink', threshold=0.5),
        ):
            _io.dump_data(
                self.store,
                by_spectrogram,
                compression=False,
                max_threads=self._spec.max_threads,
            )

            for i in range(count - len(data_list), count):
                with _util.log_capture_context('sink', capture_index=i):
                    logger.info('💾')
