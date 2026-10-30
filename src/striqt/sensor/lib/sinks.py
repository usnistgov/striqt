"""sinks that write or collect the analyzed dataset of each capture.

Every sink is constructed from a `Sweep` and reads the `sink` block of that sweep.
"""

from __future__ import annotations as __

import itertools as itertools
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, cast

import striqt.analysis as sa
from striqt.analysis.lib.util import np, xr

from .. import specs as specs
from . import compute, controller, io, util

if TYPE_CHECKING:
    import shutil
    import zipfile

    import zarr

    from striqt.analysis.lib.stores import StagedMetadataStore
else:
    shutil = sa.util.lazy_import('shutil')
    zipfile = sa.util.lazy_import('zipfile')
    zarr = sa.util.lazy_import('zarr')


class _BatchTracker:
    size: int
    total_size: int

    def __init__(self, captures: tuple[specs.SensorCapture, ...], min_size: int):
        sizes = specs.captures.concat_group_sizes(captures, min_size=min_size)
        self.total_size = sum(sizes)

        self._cycler = itertools.cycle(sizes)
        if len(sizes) > 0 and sizes[0] > 0:
            self.next()

    def next(self) -> int:
        """step to the next batch size in the cycle"""
        self.size = next(self._cycler)
        return self.size


class Zipper:
    """archive a ``.zarr`` directory into a ``.zarr.zip`` file of stored entries."""

    BUFFER_SIZE = 10 * 1024 * 1024
    temp_dir: str

    def __init__(self, zip_path: str | Path, force: bool = False):
        self.zip_path = Path(zip_path)

        if self.zip_path.exists() and not force:
            raise OSError(f'file "{self.zip_path!s}" already exists')
        elif force:
            logger = sa.util.get_logger('sink')
            logger.warning(f'will overwrite existing "{self.zip_path!s}"')

        self.temp_dir = str(self.zip_path.with_suffix(''))

    @classmethod
    def from_zarr(
        cls,
        input_dir: str | Path,
        out_file: str | Path | None = None,
        force: bool = False,
    ) -> Zipper:
        """return a `Zipper` for the existing zarr directory `input_dir`.

        Args:
            input_dir: the ``.zarr`` directory to archive
            out_file: the zip path; None selects `input_dir` with ``.zip`` appended
            force: if True, overwrite an existing `out_file`

        Raises:
            IOError: if `input_dir` does not exist, or if `out_file` exists and
                `force` is False
        """
        input_dir = Path(input_dir)

        if not input_dir.exists():
            raise OSError(f'no such directory {input_dir!s}')

        if out_file is None:
            out_file = input_dir.parent / (input_dir.name + '.zip')

        zipper = cls(out_file, force=force)
        zipper.temp_dir = str(input_dir)

        return zipper

    def archive(self, remove=True):
        """archive the .zarr directory and return the path to the zipfile"""
        if not Path(self.temp_dir).exists():
            return

        stopwatch = sa.util.stopwatch(f'zip {self.temp_dir!r}', 'sink')

        stream = open(self.zip_path, 'wb', self.BUFFER_SIZE)  # noqa: SIM115
        zf = zipfile.ZipFile(
            stream, 'w', compression=zipfile.ZIP_STORED, allowZip64=True
        )

        with stopwatch, stream, zf:
            for root, _, files in os.walk(self.temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.temp_dir)
                    with open(file_path, 'rb', buffering=self.BUFFER_SIZE) as zstream:
                        zf.writestr(arcname, zstream.read())
            stream.flush()
            os.fsync(stream.fileno())

        if remove:
            shutil.rmtree(self.temp_dir)

        return str(self.zip_path)


class SinkBase(Generic[specs.SC]):
    """base sink that accepts the analyzed dataset of one capture at a time.

    A sink is constructed from a `Sweep`, reads the `sink` block of that sweep, and
    is entered as a context manager before the first `append`. Writes run on 1
    worker thread.
    """

    def __init__(
        self,
        sweep_spec: specs.Sweep[Any, Any, specs.SC, Any],
        format_path: specs.paths.PathFormatter | None = None,
        *,
        force: bool = False,
    ):
        from concurrent.futures import ThreadPoolExecutor

        self._spec = sweep_spec.sink
        self.captures_elapsed = 0
        self._format_path = format_path

        self.store = None
        self.force = force
        self._future = None
        self._pending_data: list[xr.Dataset] = []
        self._executor = ThreadPoolExecutor(1)

    def pop(self) -> list[xr.Dataset]:
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
    ) -> xr.Dataset | compute.DelayedDataset:
        from . import compute

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
            sa.util.get_logger('sink').info('done')

    def append(self, capture_result) -> compute.DelayedDataset:
        self.captures_elapsed += 1
        return capture_result

    def wait(self):
        pass


class MemoryCacheSink(SinkBase):
    """collect the dataset of every capture in memory until `pop` is called.

    Nothing is written. `close` leaves the collected datasets in place.
    """

    def open(self) -> None:
        pass

    def close(self, *exc_info):
        pass


class ZarrSinkBase(SinkBase):
    """open the zarr store at `sink.path` and log the written path at close.

    Raises:
        ValueError: if `sweep_spec.sink.path` is None
    """

    def __init__(
        self,
        sweep_spec: specs.Sweep[Any, Any, specs.SC, Any],
        format_path: specs.paths.PathFormatter | None = None,
        *,
        force: bool = False,
    ):
        if sweep_spec.sink.path is None:
            raise ValueError('sink.path is null, but a zarr sink needs an output path')

        super().__init__(sweep_spec, format_path, force=force)
        self._sweep = sweep_spec

    def open(self):
        self.store = io.open_store(
            self._spec, format_path=self._format_path, force=self.force
        )

    def close(self, *exc_info):
        super().close(*exc_info)

        path = self.get_root_path() if self.store is not None else None
        if self.store is not None and getattr(self.store, '_is_open', True):
            self.store.close()

        if path is not None and Path(path).exists():
            sa.util.get_logger('sink').info(f'wrote "{path!s}"')
        else:
            sa.util.get_logger('sink').info('no data was written')

    def get_root_path(self) -> str:
        if self.store is None:
            raise OSError('store is not open')

        # a staging store wraps the store that owns the path
        for store in (
            self.store,
            getattr(self.store, '_inner', None),
            getattr(self.store, '_store', None),
        ):
            if store is None:
                continue
            if hasattr(store, 'path'):
                return str(store.path)
            if hasattr(store, 'root'):
                return str(store.root)
        raise OSError('the store has no path')


# %% Chunk geometry of the capture sink
def chunk_shape(
    rows: int, row_shape: tuple[int, ...], itemsize: int, budget: int
) -> tuple[int, ...]:
    """return a chunk shape of at most `budget` bytes for an array of `rows` rows.

    A row is one port of one capture with shape `row_shape`. When a row fits in
    `budget`, the chunk is as many whole rows as fit, and at least 1. Otherwise the
    chunk is one row, cut along the leading axes until the trailing block fits.
    """
    row_bytes = max(1, math.prod(row_shape) * itemsize)
    if row_bytes <= budget:
        return (max(1, min(rows, budget // row_bytes)), *row_shape)
    chunk = list(row_shape)
    for i in range(len(row_shape)):
        tail = max(1, math.prod(row_shape[i + 1 :]) * itemsize)
        chunk[i] = max(1, budget // tail)
        if tail <= budget:
            break
    return (1, *chunk)


def trailing_regions(
    row_shape: tuple[int, ...], chunk: tuple[int, ...]
) -> list[tuple[slice, ...]]:
    """return the chunk-aligned regions of the trailing axes of one row block."""
    grid = [range(0, n, c) for n, c in zip(row_shape, chunk[1:])]
    return [
        tuple(slice(s, min(s + c, n)) for s, c, n in zip(starts, chunk[1:], row_shape))
        for starts in itertools.product(*grid)
    ]


class _RowQueue:
    """rows of one variable not yet written, in arrival order"""

    def __init__(self) -> None:
        self.parts: list[np.ndarray] = []
        self.count = 0

    def push(self, rows: np.ndarray) -> None:
        self.parts.append(rows)
        self.count += rows.shape[0]

    def take(self, n: int) -> np.ndarray:
        taken = []
        got = 0
        while got < n:
            part = self.parts[0]
            need = n - got
            if part.shape[0] <= need:
                taken.append(self.parts.pop(0))
            else:
                taken.append(part[:need])
                self.parts[0] = part[need:]
            got += taken[-1].shape[0]
        self.count -= n
        return taken[0] if len(taken) == 1 else np.concatenate(taken, axis=0)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class ZarrCaptureSink(ZarrSinkBase):
    """write the dataset of each capture into a zarr store allocated for the whole sweep.

    `open` allocates every data variable and coordinate of the store from the sweep
    spec, with one row per port of every expanded capture along ``capture``, and
    keeps the metadata in memory until each `append` (directory store) or until
    `close` (zip store). `append` places the arrays of a `DelayedDataset` on the
    union grid of the sweep and writes every chunk that the capture completes, so
    each chunk is written once. `close` writes the partial chunks, trims the store
    to the rows written, and consolidates the metadata.
    """

    def __init__(
        self,
        sweep_spec: specs.Sweep[Any, Any, specs.SC, Any],
        format_path: specs.paths.PathFormatter | None = None,
        *,
        force: bool = False,
        action_spec: specs.ActionSpec | None = None,
    ):
        super().__init__(sweep_spec, format_path, force=force)
        self._action_spec = sweep_spec if action_spec is None else action_spec
        self._source_id = controller.lookup.id(sweep_spec.source)
        self._planned_rows = compute.datasets.sweep_row_count(
            sweep_spec, self._source_id
        )
        self._rows_total = 0
        self._row = 0
        self._grids: dict[str, compute.datasets.VariableGrid] = {}
        self._coord_names: list[str] = []
        self._encoders: dict[str, Any] = {}
        self._arrays: dict[str, Any] = {}
        self._chunks: dict[str, tuple[int, ...]] = {}
        self._regions: dict[str, list[tuple[slice, ...]]] = {}
        self._queues: dict[str, _RowQueue] = {}
        self._next_row: dict[str, int] = {}
        self._dropped_attrs: dict[str, set[str]] = {}
        self._root_attrs: dict[str, Any] = {}
        self._flush_each = True

    @property
    def rows_written(self) -> int:
        """the number of rows written so far"""
        return self._row

    def _acquisition_info_cls(self) -> type[specs.AcquisitionInfo]:
        from .sources.base import acquisition_info_cls

        sensor = getattr(type(self._sweep), 'sensor', None)
        if sensor is None:
            sensor = controller.lookup.instance(self._sweep.source).sensor
        return acquisition_info_cls(sensor.source_cls)

    def open(self):
        from striqt.analysis.lib import stores

        super().open()
        # a zip entry cannot be rewritten, so a zip store sees its metadata at close
        self._flush_each = not self.get_root_path().lower().endswith('.zip')
        self.store = stores.staged_metadata_store(self.store)

        if self._spec.max_threads is not None:
            import numcodecs

            numcodecs.blosc.set_nthreads(self._spec.max_threads)

        with sa.util.stopwatch('allocate store', 'sink', threshold=0.25):
            template, grids, encoding, encoders = compute.datasets.sweep_template(
                self._sweep,
                self._source_id,
                self._acquisition_info_cls(),
                attrs=self._action_spec.to_dict(True, allow_tuple_keys=False),
            )
            self._grids = grids
            self._encoders = encoders
            self._coord_names = list(encoders)
            self._root_attrs = dict(template.attrs)
            self._rows_total = max(self._planned_rows, 1)
            self._allocate(template, encoding)
        self._flush()

    def _allocate(self, template: xr.Dataset, encoding: dict[str, dict[str, Any]]):
        """allocate the arrays of the zero-row `template` at the current row count."""
        chunks: dict[str, tuple[int, ...]] = {}
        row_shapes: dict[str, tuple[int, ...]] = {}
        for key, var in template.variables.items():
            name = str(key)
            if compute.CAPTURE_DIM not in var.dims:
                continue
            row_shape = tuple(var.shape[1:])
            itemsize = 8 if var.dtype.kind == 'M' else var.dtype.itemsize
            chunks[name] = chunk_shape(
                self._rows_total, row_shape, itemsize, self._spec.max_chunk_bytes
            )
            row_shapes[name] = row_shape
        arrays = sa.lib.io.allocate_store(
            self.store,
            template,
            rows=self._rows_total,
            chunks=chunks,
            encoding=encoding,
        )
        for name, array in arrays.items():
            self._arrays[name] = array
            self._chunks[name] = chunks[name]
            self._regions[name] = trailing_regions(row_shapes[name], chunks[name])
            self._queues[name] = _RowQueue()
            self._next_row[name] = 0

    def _grow(self, rows: int):
        """resize every array so that `rows` rows fit."""
        new_total = max(rows, self._rows_total + max(self._planned_rows, 1))
        for array in self._arrays.values():
            array.resize((new_total, *array.shape[1:]))
        self._rows_total = new_total

    def _trim(self):
        """resize every array to the rows written."""
        if self._row == self._rows_total:
            return
        for array in self._arrays.values():
            array.resize((self._row, *array.shape[1:]))
        self._rows_total = self._row

    def _push(self, name: str, rows: np.ndarray):
        self._queues[name].push(rows)
        self._drain(name, final=False)

    def _drain(self, name: str, final: bool):
        queue = self._queues[name]
        k = self._chunks[name][0]
        array = self._arrays[name]
        while queue.count >= k or (final and queue.count > 0):
            n = min(k, queue.count)
            block = queue.take(n)
            r0 = self._next_row[name]
            rows = slice(r0, r0 + n)
            for region in self._regions[name]:
                array[(rows, *region)] = block[(slice(None), *region)].astype(
                    array.dtype, copy=False
                )
            self._next_row[name] += n

    def _update_attrs(self, name: str, incoming: dict[str, Any]):
        """add the run-time attrs of one capture, dropping a key that conflicts."""
        array = self._arrays[name]
        dropped = self._dropped_attrs.setdefault(name, set())
        current = dict(array.attrs)
        changes: dict[str, Any] = {}
        removals: list[str] = []
        for key, value in incoming.items():
            if key in dropped:
                continue
            value = _json_value(value)
            if key not in current:
                changes[key] = value
            elif not compute.datasets._equivalent(current[key], value):
                removals.append(key)
                dropped.add(key)
        if removals:
            for key in removals:
                del current[key]
            array.attrs.put(current)
        if changes:
            array.attrs.update(changes)

    def _add_extra_data(self, extra: dict[str, xr.DataArray]):
        """allocate the `extra_data` variables not yet in the store."""
        new = {k: v for k, v in extra.items() if k not in self._arrays}
        if not new:
            return
        # xarray replaces the root attrs on every write, so the template carries them
        template = xr.Dataset(attrs=self._root_attrs)
        encoding: dict[str, dict[str, Any]] = {}
        for name, var in new.items():
            dtype, enc, encoder = compute.datasets.capture_coord_storage(var.values)
            self._encoders[name] = encoder
            if enc:
                encoding[name] = enc
            template[name] = var.isel({compute.CAPTURE_DIM: slice(0, 0)}).astype(dtype)
        self._allocate(template, encoding)

    def _flush(self):
        if self._flush_each:
            cast('StagedMetadataStore', self.store).flush_metadata()

    def append(self, capture_result: compute.DelayedDataset) -> compute.DelayedDataset:
        dd = capture_result
        ports = len(specs.captures.split_capture_ports(dd.capture))
        if self._row + ports > self._rows_total:
            self._grow(self._row + ports)

        for name, grid in self._grids.items():
            delayed = dd.delayed[name]
            result = np.asarray(delayed.result)
            if grid.mixed:
                index = compute.datasets.row_index(grid, dd.capture, delayed.spec)
                result = compute.datasets.scatter_rows(grid, result, index)
            self._push(name, result)
            self._update_attrs(name, dict(delayed.attrs))

        coords = compute.datasets.build_capture_coords(
            dd.capture, dd.extra_coords, self._sweep.loops
        )
        if coords is not None:
            for name in self._coord_names:
                self._push(name, self._encoders[name](coords[name].values))

        if dd.extra_data:
            extra = compute.datasets.extra_data_arrays(dd.extra_data, ports)
            self._add_extra_data(extra)
            for name, var in extra.items():
                self._push(name, self._encoders[name](var.values))

        self._row += ports
        self.captures_elapsed += 1
        self._flush()

        with util.log_capture_context('sink', capture_index=self.captures_elapsed - 1):
            sa.util.get_logger('sink').info('💾')
        return dd

    def close(self, *exc_info):
        if self.store is not None and self._arrays:
            for name in self._arrays:
                self._drain(name, final=True)
            self._trim()
            zarr.consolidate_metadata(self.store)
            cast('StagedMetadataStore', self.store).flush_metadata()
        super().close(*exc_info)


class ZarrTimeAppendSink(ZarrSinkBase):
    """append the spectrogram of each capture along the spectrogram time axis.

    The captures of one write batch share one `capture` index, and each batch is
    appended along `capture`. The store is written without compression.

    Raises:
        ValueError: if `sweep_spec.analysis` has no `spectrogram`, or if the sink
            writes a ``.zarr.zip`` file
    """

    _batch: _BatchTracker

    def __init__(
        self,
        sweep_spec: specs.Sweep,
        format_path: specs.paths.PathFormatter | None = None,
        *,
        force: bool = False,
        batch_size: int | None = None,
    ):
        if getattr(sweep_spec.analysis, 'spectrogram', None) is None:
            raise ValueError(
                '"analysis" spec must include "spectrogram" to append on spectrogram time axis'
            )
        if sweep_spec.sink.store == 'zip' or str(sweep_spec.sink.path).endswith('.zip'):
            raise ValueError('ZarrTimeAppendSink writes directory stores only')

        super().__init__(sweep_spec, format_path, force=force)

        if batch_size is None:
            batch_size = sweep_spec.sink.batched_write_count
        source_id = controller.lookup.id(sweep_spec.source)
        captures = specs.sequencing.repeat_captures(sweep_spec, source_id)
        self._batch = _BatchTracker(captures, min_size=batch_size)

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
