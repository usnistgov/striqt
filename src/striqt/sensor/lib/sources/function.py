"""Function generator virtual IQ sources for testing"""

from __future__ import annotations as __

import functools
from typing import TYPE_CHECKING

from striqt.analysis import Capture, source

from ... import specs
from .. import util
from . import base

from ..typing import PS, PC, SourceBackend, TypeVar

if TYPE_CHECKING:
    import numpy as np
    import striqt.waveform as sw
else:
    np = util.lazy_import('numpy')


SS = TypeVar('SS', bound=specs.FunctionSource)
SC = TypeVar('SC', bound=specs.SensorCapture)


def _lo_shift_tone(inds, resampler: 'sw.ResamplerDesign', xp, lo_offset=None):
    if lo_offset is None:
        lo_offset = resampler['lo_offset']
    phase_scale = (2j * np.pi * lo_offset) / resampler['fs_sdr']
    return xp.exp(phase_scale * inds).astype('complex64')


class TestSourceBase(base.VirtualSource[SS, SC]):
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

        return super().read(
            buffers, offset, count, timeout_sec=timeout_sec, on_overflow=on_overflow
        )

    def _sync_time_source(self):
        self._sync_time_ns = round(1_000_000_000 * self._samples_elapsed)

    @util.cached_property
    def id(self):  # pyright: ignore
        return '00'

    @util.cached_property
    def about(self) -> specs.AboutSource: # pyright: ignore
        return specs.AboutSource(num_rx_ports=self.setup_spec.num_rx_ports)


class SingleToneSource(TestSourceBase[specs.FunctionSource, specs.SingleToneCapture]):
    def get_waveform(
        self,
        count: int,
        start: int,
        offset: int,
        *,
        port: int = 0,
        xp,
        dtype='complex64',
    ):
        capture = self._capture
        resampler = self.get_resampler(self._capture)
        fs = resampler['fs_sdr']
        i = xp.arange(start + offset, start + count + offset, dtype='int64')

        lo = _lo_shift_tone(i, resampler, xp)

        phi = (2 * np.pi * capture.frequency_offset) / fs * i + np.pi / 2
        x = lo * xp.exp(1j * phi)
        x = x.astype(self.spec.transport_dtype)

        if capture.snr is not None:
            power = 10 ** (-capture.snr / 10)
            noise = source.simulated_awgn(
                capture.replace(duration=x.shape[-1], sample_rate=1),
                xp=xp,
                seed=0,
                power_spectral_density=power,
            )
            noise = noise[i % noise.size]
            x += noise

        return x


class DiracDeltaSource(TestSourceBase[specs.FunctionSource, specs.DiracDeltaCapture]):
    def get_waveform(
        self,
        count: int,
        start: int,
        offset: int,
        *,
        port: int = 0,
        xp,
        dtype='complex64',
    ):
        capture = self._capture
        fs = self.get_resampler(capture)['fs_sdr']

        abs_pulse_index = round(capture.time * fs) + start
        rel_pulse_index = abs_pulse_index - offset
        ret = xp.full(count, 1e-20, dtype=self.spec.transport_dtype)

        if rel_pulse_index >= 0 and rel_pulse_index < count:
            ret[rel_pulse_index] = 10 ** (capture.power / 20)

        return ret[np.newaxis,]


class SawtoothSource(TestSourceBase[specs.FunctionSource, specs.SawtoothCapture]):
    def get_waveform(
        self,
        count: int,
        start: int,
        offset: int,
        *,
        port: int = 0,
        xp,
        dtype='complex64',
    ):
        capture = self._capture
        fs = self.get_resampler(capture)['fs_sdr']

        ret = xp.empty(count, dtype='complex64')
        ii = xp.arange(start + offset, start + count + offset, dtype='uint64')
        t = ii / fs
        magnitude = 10 ** (capture.power / 20)
        ret.real[:] = (t % capture.period) * (magnitude / capture.period)
        ret.imag[:] = 0
        return ret


class NoiseSource(TestSourceBase[specs.FunctionSource, specs.NoiseCapture]):
    def get_waveform(
        self,
        count: int,
        start: int,
        offset: int,
        *,
        port: int = 0,
        xp,
        dtype='complex64',
    ):
        capture = self._capture
        fs = self.get_resampler(capture)['fs_sdr']

        backend_capture = Capture(
            duration=(count + offset) / fs,
            sample_rate=fs,
            analysis_bandwidth=capture.analysis_bandwidth,
        )

        x = source.simulated_awgn(
            backend_capture,
            xp=xp,
            seed=0,
            power_spectral_density=capture.noise_psd,
        )
        # x /= np.sqrt(self._capture.backend_sample_rate / self.sample_rate())

        if offset < 0:
            pad = -offset
            start_index = 0
            count = count - pad
        else:
            pad = 0

        ret = x[offset : count + offset]

        if pad:
            return xp.pad(ret, [[pad, 0]], mode='constant')
        else:
            return ret
