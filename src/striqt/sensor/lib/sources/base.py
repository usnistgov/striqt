from __future__ import annotations as __

import functools
import time
from typing import TYPE_CHECKING

from ... import specs
from .. import util
from ..typing import SS, SC, SourceBackend

if TYPE_CHECKING:
    from ..typing import Array, Self
    import numpy as np
    import striqt.waveform as sw

else:
    np = util.lazy_import('numpy')


class NoSource(SourceBackend[specs.NoSource, specs.SensorCapture]):
    """fast paths to acquire empty buffers"""

    _samples_elapsed = 0

    def __init__(self, spec: specs.NoSource):
        self.spec = spec

    @util.cached_property
    def about(self):
        return specs.structs.AboutSource(num_rx_ports=self.spec.num_rx_ports)

    @util.cached_property
    def id(self) -> str:  # pyright: ignore
        return 'null'

    def reset_sample_counter(self, value=0):
        self._sync_time_source()
        self._samples_elapsed = value
        self._sample_start_index = value

    def _sync_time_source(self):
        self._sync_time_ns = time.time_ns()

    def setup(self, *, captures=None, loops=None):
        self.reset_sample_counter()

    def arm(self, capture):
        self._capture = capture
        self.reset_sample_counter()

    def read(
        self, buffers, offset, count, timeout_sec=None, *, on_overflow='except'
    ) -> tuple[int, int]:
        assert self._capture is not None

        fs = float(self.get_resampler(self._capture)['fs_sdr'])
        sample_period_ns = 1_000_000_000 / fs
        timestamp_ns = self._sync_time_ns + self._samples_elapsed * sample_period_ns

        self._samples_elapsed += count

        return count, round(timestamp_ns)

    def get_resampler(self, capture: specs.SensorCapture) -> sw.ResamplerDesign:
        from ..compute import design_resampler

        mcr = self.spec.master_clock_rate
        return design_resampler(capture, mcr)


class VirtualSource(SourceBackend[SS, SC]):
    spec: SS
    _samples_elapsed = 0
    _overlaps: tuple[int, int] = (0, 0)
    _capture: SC

    def __init__(self, spec: SS):
        self.setup_spec = spec

    def reset_sample_counter(self, value=0):
        self._sync_time_source()
        self._samples_elapsed = value
        self._sample_start_index = value

    def get_waveform(
        self,
        count: int,
        start: int,
        offset: int,
        *,
        port: int = 0,
        xp,
        dtype='complex64',
    ) -> Array:
        raise NotImplementedError

    def setup(self, captures=None, loops=None):
        self.reset_sample_counter()

    def arm(self, capture: SC):
        self._capture = capture
        self.reset_sample_counter()

    def trigger(self, overlaps=(0, 0)):
        self._overlaps = overlaps

    def read(
        self,
        buffers,
        offset,
        count,
        timeout_sec=None,
        *,
        on_overflow: specs.types.OnOverflow = 'except',
    ):
        assert self._capture is not None

        if not isinstance(self._capture.port, tuple):
            ports = (self._capture.port,)
        else:
            ports = self._capture.port

        for port, buf in zip(ports, buffers):
            values = self.get_waveform(
                count,
                start=self._overlaps[0],
                offset=self._samples_elapsed,
                port=port,
                xp=getattr(self, 'xp', np),
            )
            buf[offset : (offset + count)] = values

        fs = float(self.get_resampler(self._capture)['fs_sdr'])
        sample_period_ns = 1_000_000_000 / fs
        timestamp_ns = self._sync_time_ns + self._samples_elapsed * sample_period_ns

        self._samples_elapsed += count

        return count, round(timestamp_ns)

    def _sync_time_source(self):
        self._sync_time_ns = round(1_000_000_000 * self._samples_elapsed)

    def get_resampler(self, capture) -> 'sw.ResamplerDesign':
        from ..compute import design_resampler

        return design_resampler(capture, self.setup_spec.master_clock_rate)
