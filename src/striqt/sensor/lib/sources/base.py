from __future__ import annotations as __

import functools
import time
from typing import TYPE_CHECKING

from striqt.analysis.lib.util import np

from ... import specs
from .. import util
from ..typing import SS, SC, SourceBackend

if TYPE_CHECKING:
    from types import ModuleType

    from striqt.waveform.lib.typing import DTypeLike

    from ..typing import Array, Self
    import striqt.waveform as sw


class ReceiveStreamError(IOError):
    pass


class NoSource(SourceBackend[specs.NoSource, specs.SensorCapture]):
    """fast paths to acquire empty buffers"""

    _samples_elapsed = 0

    def __init__(self, spec: specs.NoSource):
        self.spec = spec

    @util.cached_property
    def about(self):
        return specs.structs.SourceInfo(num_rx_ports=self.spec.num_rx_ports)

    def close(self):
        pass

    def get_id(self) -> str:  # pyright: ignore
        return 'null'

    def get_info(self):
        return specs.SourceInfo(num_rx_ports=self.spec.num_rx_ports)

    def reset_sample_counter(self, value=0):
        self._sync_time_source()
        self._samples_elapsed = value
        self._sample_start_index = value

    def _sync_time_source(self):
        self._sync_time_ns = time.time_ns()

    def setup(self, rx_ports: tuple[int, ...] | None = None):
        self.reset_sample_counter()

    def trigger(self, overlaps=(0, 0)):
        self._overlaps = overlaps

    def arm(self, capture):
        self._capture = capture
        self.reset_sample_counter()

    def read(
        self, buffers, offset, count, timeout_sec=None, *, on_overflow='except'
    ) -> tuple[int, int]:
        assert self._capture is not None

        fs = float(self.get_resampler(self._capture)['fs_sdr'])
        sample_period_ns = 1_000_000_000 / fs
        timestamp_ns = self._sync_time_ns + round(
            self._samples_elapsed * sample_period_ns
        )

        self._samples_elapsed += count

        return count, round(timestamp_ns)

    def get_resampler(self, capture: specs.SensorCapture) -> sw.ResamplerDesign:
        from ..compute import design_resampler

        mcr = self.spec.master_clock_rate
        return design_resampler(capture, mcr)


class VirtualSource(SourceBackend[SS, SC]):
    setup_spec: SS
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
        start_index: int,
        *,
        port: int = 0,
        xp: ModuleType,
        dtype: DTypeLike = 'complex64',
    ) -> Array:
        """`count` samples of `port` starting at absolute sample `start_index`.

        Indices are at the source sample rate and are referenced to the corrected
        capture: index 0 is the first sample of the output of `correct_iq`, so
        negative indices are the leading-overlap pre-roll that it trims. A source
        that has no samples there (a file) fills them with zeros.
        """
        raise NotImplementedError

    def setup(self, rx_ports: tuple[int, ...] | None = None):
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

        start_index = self._samples_elapsed - self._overlaps[0]

        for port, buf in zip(ports, buffers):
            values = self.get_waveform(
                count, start_index, port=port, xp=getattr(self, 'xp', np)
            )
            buf[offset : (offset + count)] = values

        fs = float(self.get_resampler(self._capture)['fs_sdr'])
        sample_period_ns = 1_000_000_000 / fs
        timestamp_ns = self._sync_time_ns + round(
            self._samples_elapsed * sample_period_ns
        )

        self._samples_elapsed += count

        return count, round(timestamp_ns)

    def _sync_time_source(self):
        self._sync_time_ns = round(1_000_000_000 * self._samples_elapsed)

    def get_resampler(self, capture) -> 'sw.ResamplerDesign':
        from ..compute import design_resampler

        return design_resampler(capture, self.setup_spec.master_clock_rate)
