from __future__ import annotations as __

import functools
import time
import typing

import striqt.waveform as sw

from ... import specs
from . import base, buffers
from ..typing import SourceBackend, PS, PC


SS = typing.TypeVar('SS', bound=specs.NoSource)
SC = typing.TypeVar('SC', bound=specs.SensorCapture)


class NoSource(SourceBackend[SS, specs.SensorCapture]):
    """fast paths to acquire empty buffers"""

    _samples_elapsed = 0
    _capture: specs.SensorCapture

    def __init__(self, spec: SS):
        self.spec = spec

    @functools.cached_property
    def info(self):
        return specs.structs.BaseSourceInfo(num_rx_ports=self.spec.num_rx_ports)

    @functools.cached_property
    def id(self) -> str:
        return 'null'

    def reset_sample_counter(self, value=0):
        self._sync_time_source()
        self._samples_elapsed = value
        self._sample_start_index = value

    def _sync_time_source(self):
        self._sync_time_ns = time.time_ns()

    def setup(self, *, captures=None, loops=None):
        self.reset_sample_counter()

    def arm(self, capture) -> SC | None:
        self._capture = capture
        self.reset_sample_counter()

    def read_buffer(
        self, buffers, offset, count, timeout_sec=None, *, on_overflow='except'
    ) -> tuple[int, int]:
        fs = float(self.get_resampler(self._capture)['fs_sdr'])
        sample_period_ns = 1_000_000_000 / fs
        timestamp_ns = self._sync_time_ns + self._samples_elapsed * sample_period_ns

        self._samples_elapsed += count

        return count, round(timestamp_ns)

    def get_resampler(self, capture: SC) -> sw.ResamplerDesign:
        from ..compute import design_resampler

        mcr = self.spec.master_clock_rate
        return design_resampler(capture, mcr)
