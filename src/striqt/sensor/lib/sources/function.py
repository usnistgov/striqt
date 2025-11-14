"""Function generator virtual IQ sources for testing"""

from __future__ import annotations

import typing
from pathlib import Path

from striqt.analysis import CaptureBase, io, simulated_awgn

from .. import specs, util
from . import base, null

if typing.TYPE_CHECKING:
    import numpy as np
else:
    np = util.lazy_import('numpy')


def lo_shift_tone(inds, radio: base.SourceBase, xp, lo_offset=None):
    design = radio.get_resampler()
    if lo_offset is None:
        lo_offset = design['lo_offset']
    phase_scale = (2j * np.pi * lo_offset) / design['fs_sdr']
    return xp.exp(phase_scale * inds).astype('complex64')


class SingleToneCapture(
    specs.CaptureSpec,
    forbid_unknown_fields=True,
    frozen=True,
    cache_hash=True,
    kw_only=True,
):
    frequency_offset: specs.Annotated[
        float, specs.meta('Input tone frequency offset to band center', 'Hz')
    ] = 0
    snr: typing.Optional[
        specs.Annotated[float, specs.meta('SNR with added noise ', 'dB')]
    ] = None


class DiracDeltaCapture(
    specs.CaptureSpec,
    forbid_unknown_fields=True,
    frozen=True,
    cache_hash=True,
    kw_only=True,
):
    time: specs.Annotated[
        float, specs.meta('pulse start time relative to the start of the waveform', 's')
    ] = 0

    power: typing.Optional[
        specs.Annotated[
            float,
            specs.meta('instantaneous power level of the impulse function', 'dB', gt=0),
        ]
    ] = 0


class SawtoothCapture(
    specs.CaptureSpec,
    forbid_unknown_fields=True,
    frozen=True,
    cache_hash=True,
    kw_only=True,
):
    period: specs.Annotated[
        float,
        specs.meta('pulse start time relative to the start of the waveform', 's', ge=0),
    ] = 0.01
    power: specs.Annotated[
        float,
        specs.meta('instantaneous power level of the impulse function', 'dB', gt=0),
    ] = 1


class NoiseCapture(
    specs.CaptureSpec,
    forbid_unknown_fields=True,
    frozen=True,
    cache_hash=True,
    kw_only=True,
):
    power_spectral_density: specs.Annotated[
        float, specs.meta('noise total channel power', 'mW/Hz', ge=0)
    ] = 1e-17


_TS = typing.TypeVar('_TS', bound=specs.NullSourceSpec)
_TC = typing.TypeVar('_TC', bound=specs.CaptureSpec)


class TestSourceBase(base.VirtualSourceBase[_TS, _TC]):
    def _read_stream(
        self,
        buffers,
        offset,
        count,
        timeout_sec=None,
        *,
        on_overflow: base.OnOverflowType = 'except',
    ):
        assert self._capture is not None

        if not isinstance(self._capture.port, tuple):
            ports = (self._capture.port,)
        else:
            ports = self._capture.port

        for port, buf in zip(ports, buffers):
            values = self.get_waveform(
                count,
                self._samples_elapsed,
                port=port,
                xp=getattr(self, 'xp', np),
            )
            buf[offset : (offset + count)] = values

        return super()._read_stream(
            buffers, offset, count, timeout_sec=timeout_sec, on_overflow=on_overflow
        )

    def get_waveform(
        self, count: int, offset: int, *, port: int = 0, xp, dtype='complex64'
    ):
        raise NotImplementedError

    def _sync_time_source(self):
        self._sync_time_ns = round(1_000_000_000 * self._samples_elapsed)


class SingleToneSource(TestSourceBase[specs.NullSourceSpec, SingleToneCapture]):
    def get_waveform(
        self, count: int, offset: int, *, port: int = 0, xp, dtype='complex64'
    ):
        capture = self.capture_spec

        fs = self.get_resampler()['fs_sdr']
        i = xp.arange(offset, count + offset, dtype='int64')

        lo = lo_shift_tone(i, self, xp)

        phi = (2 * np.pi * capture.frequency_offset) / fs * i + np.pi / 2
        ret = lo * xp.exp(1j * phi)
        ret = ret.astype(self.__setup__.transport_dtype)

        if capture.snr is not None:
            power = 10 ** (-capture.snr / 10)
            noise = simulated_awgn(
                capture.replace(sample_rate=fs),
                xp=xp,
                seed=0,
                power_spectral_density=power,
            )
            noise = noise[i % noise.size]
            ret += noise

        return ret


class DiracDeltaSource(TestSourceBase):
    def get_waveform(
        self, count: int, offset: int, *, port: int = 0, xp, dtype='complex64'
    ):
        capture = self.capture_spec

        abs_pulse_index = round(capture.time * capture.backend_sample_rate)
        rel_pulse_index = abs_pulse_index - offset

        ret = xp.zeros(count, dtype=self.__setup__.transport_dtype)

        if rel_pulse_index >= 0 and rel_pulse_index < count:
            ret[rel_pulse_index] = 10 ** (capture.power / 20)

        return ret[np.newaxis,]


class SawtoothSource(TestSourceBase[specs.NullSourceSpec, SawtoothCapture]):
    def get_waveform(
        self, count: int, offset: int, *, port: int = 0, xp, dtype='complex64'
    ):
        capture = self.capture_spec

        ret = xp.empty(count, dtype='complex64')
        ii = xp.arange(offset, count + offset, dtype='uint64')
        t = ii / capture.backend_sample_rate
        magnitude = 2 * np.sqrt(capture.power)
        ret.real[:] = (t % capture.period) * (magnitude / capture.period)
        ret.imag[:] = 0
        return ret


class NoiseSource(TestSourceBase[specs.NullSourceSpec, NoiseCapture]):
    def get_waveform(
        self, count: int, offset: int, *, port: int = 0, xp, dtype='complex64'
    ):
        capture = self.capture_spec
        fs = self.get_resampler()['fs_sdr']

        backend_capture = CaptureBase(
            duration=(count + offset) / fs,
            sample_rate=fs,
            analysis_bandwidth=capture.analysis_bandwidth,
        )

        x = simulated_awgn(
            backend_capture,
            xp=xp,
            seed=0,
            power_spectral_density=capture.power_spectral_density,
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
