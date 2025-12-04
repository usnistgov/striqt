from __future__ import annotations as __

import functools
import time
import typing

from ... import specs
from .. import util
from . import _base

if typing.TYPE_CHECKING:
    import numpy as np
    from striqt.waveform._typing import ArrayType
else:
    np = util.lazy_import('numpy')

_TS = typing.TypeVar('_TS', bound=specs.NoSource)
_TC = typing.TypeVar('_TC', bound=specs.ResampledCapture)


class NoSource(_base.SourceBase[_TS, _TC]):
    """fast paths to acquire empty buffers"""

    _samples_elapsed = 0

    @functools.cached_property
    def info(self):
        return _base.BaseSourceInfo(num_rx_ports=self.setup_spec.num_rx_ports)

    @functools.cached_property
    def id(self) -> str:
        return 'null'

    def reset_sample_counter(self, value=0):
        self._sync_time_source()
        self._samples_elapsed = value
        self._sample_start_index = value

    def _sync_time_source(self):
        self._sync_time_ns = time.time_ns()

    def _apply_setup(self, spec: _TS):
        self.reset_sample_counter()

    def _prepare_capture(self, capture) -> _TC | None:
        self.reset_sample_counter()

    def _read_stream(
        self, buffers, offset, count, timeout_sec=None, *, on_overflow='except'
    ) -> tuple[int, int]:
        fs = float(self.get_resampler()['fs_sdr'])
        sample_period_ns = 1_000_000_000 / fs
        timestamp_ns = self._sync_time_ns + self._samples_elapsed * sample_period_ns

        self._samples_elapsed += count

        return count, round(timestamp_ns)

    def get_resampler(self, capture: _TC | None = None) -> _base.ResamplerDesign:
        if capture is None:
            capture = self.capture_spec

        return _base.design_capture_resampler(self.setup_spec.base_clock_rate, capture)
