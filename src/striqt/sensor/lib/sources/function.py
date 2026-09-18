"""Function generator virtual IQ sources for testing"""

from __future__ import annotations as __

from typing import TYPE_CHECKING

from striqt.analysis import testing

from ... import specs
from .. import util
from . import base

from ..typing import TypeVar

if TYPE_CHECKING:
    from ..typing import Array


SS = TypeVar('SS', bound=specs.FunctionSource)
SC = TypeVar('SC', bound=specs.SensorCapture)


class TestSourceBase(base.VirtualSource[SS, SC]):
    def close(self):
        pass

    def get_id(self):
        return ''

    def get_info(self):
        return specs.SourceInfo(num_rx_ports=self.setup_spec.num_rx_ports)

    def _generator_kws(self, count: int, start_index: int, xp) -> dict:
        """the `striqt.analysis.testing` arguments shared by every source here.

        The sample rate is the source's rather than the capture's, because host
        resampling happens after acquisition. `duration` goes unused because `count`
        pins the sample window.
        """
        return dict(
            duration=None,
            sample_rate=self.get_resampler(self._capture)['fs_sdr'],
            start_index=start_index,
            count=count,
            xp=xp,
            dtype=self.setup_spec.transport_dtype,
        )

    def _port_position(self, port: int) -> tuple[int, int]:
        """the row index of `port` in the capture, and the capture's port count.

        The generators return rows in capture order, so a source that is called one
        port at a time needs the position of that port rather than its number, which
        may differ (`port: [1, 0]`).

        Callers generate all `ports` rows and keep one, rather than asking for a
        single port, so that a multi-port acquisition reproduces one generator call
        with the same `ports`: the noise draws interleave port-by-port within each
        sample, so one row of an N-port call is not the result of any smaller call.
        The cost is that each port of an N-port capture draws all N ports.
        """
        ports = self._capture.port

        if not isinstance(ports, tuple):
            return 0, 1
        else:
            return ports.index(port), len(ports)

    @util.cached_property
    def id(self):  # pyright: ignore
        return '00'

    @util.cached_property
    def about(self) -> specs.SourceInfo:  # pyright: ignore
        return specs.SourceInfo(num_rx_ports=self.setup_spec.num_rx_ports)


class SingleToneSource(TestSourceBase[specs.FunctionSource, specs.SingleToneCapture]):
    def get_waveform(
        self,
        count: int,
        start_index: int,
        *,
        port: int = 0,
        xp,
        dtype='complex64',
    ) -> Array:
        capture = self._capture
        index, ports = self._port_position(port)

        x = testing.single_tone(
            frequency_offset=capture.frequency_offset,
            snr=capture.snr,
            lo_offset=self.get_resampler(capture)['lo_offset'],
            ports=ports,
            **self._generator_kws(count, start_index, xp),
        )

        return x[index : index + 1]


class DiracDeltaSource(TestSourceBase[specs.FunctionSource, specs.DiracDeltaCapture]):
    def get_waveform(
        self,
        count: int,
        start_index: int,
        *,
        port: int = 0,
        xp,
        dtype='complex64',
    ) -> Array:
        capture = self._capture

        return testing.dirac_delta(
            time=capture.time,
            power=capture.power,
            **self._generator_kws(count, start_index, xp),
        )


class SawtoothSource(TestSourceBase[specs.FunctionSource, specs.SawtoothCapture]):
    def get_waveform(
        self,
        count: int,
        start_index: int,
        *,
        port: int = 0,
        xp,
        dtype='complex64',
    ) -> Array:
        capture = self._capture

        return testing.sawtooth(
            period=capture.period,
            power=capture.power,
            **self._generator_kws(count, start_index, xp),
        )


class NoiseSource(TestSourceBase[specs.FunctionSource, specs.NoiseCapture]):
    def get_waveform(
        self,
        count: int,
        start_index: int,
        *,
        port: int = 0,
        xp,
        dtype='complex64',
    ) -> Array:
        capture = self._capture
        index, ports = self._port_position(port)

        x = testing.noise(
            noise_psd=capture.noise_psd,
            ports=ports,
            **self._generator_kws(count, start_index, xp),
        )

        return x[index : index + 1]
