import functools
import time
import typing

from . import base
from .. import specs


class NullSetup(specs.RadioSetup, forbid_unknown_fields=True, frozen=True, cache_hash=True, kw_only=True):
    # make these configurable, to support matching hardware for warmup sweeps
    num_rx_ports: int
    stream_all_rx_ports: bool = False


_TS = typing.TypeVar('_TS', bound=NullSetup)
_TC = typing.TypeVar('_TC', bound=specs.RadioCapture)


class NullSource(base.SourceBase[_TS, _TC]):
    """emulate a radio with fake data"""

    _setup_cls: _TS = NullSetup
    _samples_elapsed = 0

    @functools.cached_property
    def source_info(self):
        return base.BaseSourceInfo(
            num_rx_ports=self.get_setup_spec().num_rx_ports
        )

    @functools.cached_property
    def id(self):
        return 'null'

    def reset_sample_counter(self, value=0):
        self._sync_time_source()
        self._samples_elapsed = value
        self._sample_start_index = value

    def _sync_time_source(self):
        self._sync_time_ns = time.time_ns()

    def _apply_setup(self, setup: _TS | None = None):
        self._source_info = base.BaseSourceInfo(num_rx_ports=setup.num_rx_ports)
        self.reset_sample_counter()

    def _prepare_capture(self, capture):
        self.reset_sample_counter()

    def _read_stream(
        self, buffers, offset, count, timeout_sec=None, *, on_overflow='except'
    ) -> tuple[int, int]:
        fs = float(self._capture.backend_sample_rate)
        sample_period_ns = 1_000_000_000 / fs
        timestamp_ns = self._sync_time_ns + self._samples_elapsed * sample_period_ns

        self._samples_elapsed += count

        return count, round(timestamp_ns)



class WarmupSource(NullSource):
    pass
