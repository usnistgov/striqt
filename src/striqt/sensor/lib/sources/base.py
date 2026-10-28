from __future__ import annotations as __

import time
from typing import TYPE_CHECKING

from striqt.analysis.lib.util import np

from ... import specs
from .. import util
from ..typing import SS, SC, SourceBackend

if TYPE_CHECKING:
    from types import ModuleType

    from striqt.waveform.lib.typing import DTypeLike

    from ..typing import Array
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

    def get_id(self) -> str:
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
    """a source that reads samples from `get_waveform` on a synthetic time base.

    The time base is 0 at the first `arm`, and `_sync_time_ns` is the time of
    corrected sample 0 of the armed capture. A re-arm continues the time base: the
    first sample read after `arm` is stamped at the end time of the samples read
    before the re-arm.
    """

    setup_spec: SS
    _samples_elapsed = 0
    _sync_time_ns = 0
    _stream_end_ns = 0
    _origin_pending = False
    _overlaps: tuple[int, int] = (0, 0)
    _capture: SC

    def __init__(self, spec: SS):
        self.setup_spec = spec

    def reset_sample_counter(self, value: int = 0) -> None:
        """restart the sample counter at `value`.

        The next read is stamped at the end time of the samples read so far.
        """
        self._sync_time_source()
        self._samples_elapsed = value

    def get_waveform(
        self,
        count: int,
        start_index: int,
        *,
        port: int = 0,
        xp: ModuleType,
        dtype: DTypeLike = 'complex64',
    ) -> Array:
        """return `count` samples of `port` starting at source sample `start_index`.

        Index 0 is the first sample of the corrected capture, and negative indices
        are the lead-overlap pre-roll that conditioning trims. A source with no
        samples before index 0 fills the pre-roll with zeros.

        Args:
            count: the number of samples, at the source sample rate
            start_index: the source-rate index of the first sample; negative in the
                pre-roll
            port: the port number of the armed capture
            xp: the array namespace of the result
            dtype: the dtype of the result

        Returns:
            a 1-D array of `count` samples in the `xp` namespace with dtype `dtype`

        Raises:
            NotImplementedError: in the base class
        """
        raise NotImplementedError

    def setup(self, rx_ports: tuple[int, ...] | None = None):
        self.reset_sample_counter()

    def arm(self, capture: SC) -> None:
        # the elapsed samples convert to time at the source rate of the capture that
        # produced the samples, so the reset precedes the change of `_capture`
        self.reset_sample_counter()
        self._capture = capture

    def trigger(self, overlaps=(0, 0)):
        self._overlaps = overlaps

    def read(
        self,
        buffers: list[Array],
        offset: int,
        count: int,
        timeout_sec: float | None = None,
        *,
        on_overflow: specs.types.OnOverflow = 'except',
    ) -> tuple[int, int]:
        assert self._capture is not None

        if not isinstance(self._capture.port, tuple):
            ports = (self._capture.port,)
        else:
            ports = self._capture.port

        start_index = self._samples_elapsed - self._overlaps[0]

        if self._origin_pending:
            # the lead overlap of the re-armed capture arrives in `trigger`, after
            # `arm`, so the origin is placed here rather than at the reset
            self._sync_time_ns = self._stream_end_ns - self._sample_time_ns(
                -self._overlaps[0]
            )
            self._origin_pending = False

        for port, buf in zip(ports, buffers):
            values = self.get_waveform(
                count, start_index, port=port, xp=getattr(self, 'xp', np)
            )
            buf[offset : (offset + count)] = values

        timestamp_ns = self._sync_time_ns + self._sample_time_ns(start_index)

        self._samples_elapsed += count

        return count, timestamp_ns

    def _sample_time_ns(self, index: int) -> int:
        """return the time (in ns) of source sample `index` after the time origin.

        Index 0 is the first sample of the corrected waveform. The result is negative
        in the lead-overlap pre-roll and is rounded to 1 ns.
        """
        fs = float(self.get_resampler(self._capture)['fs_sdr'])
        return round(index * 1_000_000_000 / fs)

    def _sync_time_source(self) -> None:
        """record the end time (in ns) of the samples read since the last reset.

        The next `read` stamps its first sample at this end time. Nothing changes
        when no capture is armed or no samples were read since the last reset.
        """
        capture = getattr(self, '_capture', None)
        if capture is None or self._samples_elapsed == 0:
            return
        index = self._samples_elapsed - self._overlaps[0]
        self._stream_end_ns = self._sync_time_ns + self._sample_time_ns(index)
        self._origin_pending = True

    def get_resampler(self, capture) -> 'sw.ResamplerDesign':
        from ..compute import design_resampler

        return design_resampler(capture, self.setup_spec.master_clock_rate)
