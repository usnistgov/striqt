from __future__ import annotations as __

import io
import itertools as itertools
from pathlib import Path
from typing import Any, cast, Generic, TYPE_CHECKING

from . import compute, io, sources, util
from .. import specs as specs

import os

import striqt.analysis as sa

if TYPE_CHECKING:
    import xarray as xr
    import zipfile
    import shutil
else:
    xr = sa.util.lazy_import('xarray')
    shutil = sa.util.lazy_import('shutil')
    zipfile = sa.util.lazy_import('zipfile')


class _BatchTracker:
    size: int

    def __init__(self, captures: tuple[specs.SensorCapture, ...], min_size: int):
        sizes = specs.helpers.concat_group_sizes(captures, min_size=min_size)

        self._cycler = itertools.cycle(sizes)
        if len(sizes) > 0 and sizes[0] > 0:
            self.next()

    def next(self) -> int:
        """step to the next batch size in the cycle"""
        self.size = next(self._cycler)
        return self.size


class _Zipper:
    BUFFER_SIZE = 10*1024*1024
    temp_dir: str
    temp_spec: specs.Sink

    def __init__(self, zip_path, sink: SinkBase):
        if sink._alias_func is not None:
            self.zip_path = Path(sink._alias_func(zip_path))
        else:
            self.zip_path = Path(zip_path)

        if self.zip_path.exists() and not sink.force:
            raise IOError(f'a zip archive already exists at "{self.zip_path!s}"')
        elif sink.force:
            logger = sa.util.get_logger('sink')
            logger.warning(f'will overwrite existing "{self.zip_path!s}"')

        self.temp_dir = str(self.zip_path.with_suffix(''))
        self.temp_spec = sink._spec.replace(path=self.temp_dir)

    def archive(self):
        """archive the .zarr directory and return the path to the zipfile"""
        if not Path(self.temp_dir).exists():
            return

        stopwatch = sa.util.stopwatch(f'zip {self.temp_dir!r}', 'sink')


        stream = open(self.zip_path, 'wb', self.BUFFER_SIZE)
        zf = zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED)

        with stopwatch, stream, zf:
            for root, _, files in os.walk(self.temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.temp_dir)
                    with open(file_path, 'rb', buffering=self.BUFFER_SIZE) as zstream:
                        zf.writestr(arcname, zstream.read())

        shutil.rmtree(self.temp_dir)

        return str(self.zip_path)


class SinkBase(Generic[specs.SC]):
    """intake acquisitions one at a time, and parcel data store"""

    def __init__(
        self,
        sweep_spec: specs.Sweep[Any, Any, specs.SC],
        alias_func: specs.helpers.PathAliasFormatter | None = None,
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
        self._pending_data: list['xr.Dataset'] = []
        self._executor = ThreadPoolExecutor(1)

        # decide group sizes
        source_id = sources.get_source_id(sweep_spec.source)
        captures = specs.helpers.loop_captures(sweep_spec, source_id)
        if len(sweep_spec.loops) > 0 and isinstance(sweep_spec.loops[0], specs.Repeat):
            captures = sweep_spec.loops[0].count * captures
        self._batch = _BatchTracker(
            captures, min_size=sweep_spec.sink.batched_write_count
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

    def append(
        self, capture_result: compute.DelayedDataset
    ) -> 'xr.Dataset|compute.DelayedDataset':
        if capture_result is None:
            return

        ds = compute.from_delayed(capture_result)
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
        with util.log_capture_context(
            'sink', capture_index=count - 1, capture_count=count
        ):
            sa.util.get_logger('sink').info(f'done')

    def append(self, capture_result) -> compute.DelayedDataset:
        self.captures_elapsed += 1
        return capture_result

    def wait(self):
        pass


class ZarrSinkBase(SinkBase):
    _zipper: _Zipper | None = None

    def open(self):
        path = Path(self._spec.path)

        if path.name.lower().endswith('.zarr.zip'):
            self._zipper = _Zipper(path, self)
            spec = self._zipper.temp_spec
        else:
            spec = self._spec

        self.store = io.open_store(spec, alias_func=self._alias_func, force=self.force)

    def close(self, *exc_info):
        super().close(*exc_info)

        if getattr(self.store, '_is_open', True):
            self.store.close()

        if self._zipper is not None:
            path = self._zipper.archive()
        else:
            path = self.get_root_path()

        if path is not None and Path(path).exists():
            sa.util.get_logger('sink').info(f'wrote "{str(path)}"')
        else:
            sa.util.get_logger('sink').info(f'no data was written')

    def get_root_path(self) -> str:
        if hasattr(self.store, 'path'):
            return self.store.path
        else:
            return self.store.root


class ZarrCaptureSink(ZarrSinkBase):
    """concatenates the data from each capture and dumps to a zarr data store"""

    def append(self, capture_result: compute.DelayedDataset):
        ret = super().append(capture_result)

        if len(self._pending_data) == self._batch.size:
            self.flush()
            self._batch.next()
        else:
            sa.util.get_logger('sink').debug('queued')

        return ret

    def flush(self):
        super().flush()
        self.wait()
        data_list = self.pop()

        if len(data_list) == 0:
            return

        self.submit(self._flush_thread, data_list)

    def _flush_thread(self, data_list):
        with sa.util.stopwatch('merge dataset', 'sink', threshold=0.25):
            dataset = xr.concat(
                data_list,
                compute.CAPTURE_DIM,
                join='outer',
                combine_attrs='drop_conflicts',
            )
            dataset = cast(xr.Dataset, dataset)

        path = self.get_root_path()
        count = self.captures_elapsed
        logger = sa.util.get_logger('sink')

        with (
            util.log_capture_context('sink', capture_index=count - 1),
            sa.util.stopwatch(f'sync to {path}', 'sink'),
        ):
            sa.dump(
                self.store,
                dataset,
                max_threads=self._spec.max_threads,
                chunk_bytes=self._spec.max_chunk_bytes,
            )

            for i in range(count - len(data_list), count):
                with util.log_capture_context('sink', capture_index=i):
                    logger.info('💾')


class ZarrTimeAppendSink(ZarrSinkBase):
    def __init__(
        self,
        sweep_spec: specs.Sweep,
        alias_func: specs.helpers.PathAliasFormatter | None = None,
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

        if len(self._pending_data) == self._batch.size:
            self.flush()
            self._batch.next()
        return ret

    def flush(self):
        self.wait()
        data_list = self.pop()

        if len(data_list) == 0:
            return

        self.submit(self._flush_thread, data_list)

    def _flush_thread(self, data_list):
        with sa.util.stopwatch('build dataset', 'sink', threshold=0.5):
            by_spectrogram = compute.concat_time_dim(data_list, 'spectrogram_time')

        path = self.get_root_path()
        count = self.captures_elapsed
        logger = sa.util.get_logger('sink')

        with (
            util.log_capture_context('sink', capture_index=count - 1),
            sa.util.stopwatch(f'sync {path}', 'sink', threshold=0.5),
        ):
            sa.dump(
                self.store,
                by_spectrogram,
                compression=False,
                max_threads=self._spec.max_threads,
            )

            for i in range(count - len(data_list), count):
                with util.log_capture_context('sink', capture_index=i):
                    logger.info('💾')

