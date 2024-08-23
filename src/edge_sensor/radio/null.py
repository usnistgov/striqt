import typing

import labbench as lb
from labbench import paramattr as attr

from .soapy import SoapyRadioDevice
from .base import RadioDevice
from ..util import import_cupy_with_fallback

if typing.TYPE_CHECKING:
    import numpy as np
else:
    np = lb.util.lazy_import('numpy')

channel_kwarg = attr.method_kwarg.int('channel', min=0, help='hardware port number')


class NullRadio(RadioDevice):
    """emulate a radio with fake data"""

    _inbuf = None

    resource: str = attr.value.str(
        default='tone', only=('empty', 'tone', 'noise'), help='waveform to return'
    )

    on_overflow = attr.value.str(
        'ignore',
        only=['ignore', 'except', 'log'],
        help='configure behavior on receive buffer overflow',
    )

    _downsample = attr.value.float(1.0, inherit=True)

    lo_offset = attr.value.float(
        0.0,
        label='Hz',
        help='digital frequency shift of the RX center frequency',
    )
    analysis_bandwidth = attr.value.float(
        None,
        min=1,
        label='Hz',
        help='bandwidth of the digital bandpass filter (or None to bypass)',
    )

    @attr.method.int(
        min=0,
        allow_none=True,
        cache=True,
        help='RX input port index',
    )
    def channel(self):
        # return none until this is set, then the cached value is returned
        return None

    @channel.setter
    def _(self, channel: int):
        pass

    @attr.method.float(
        min=0,
        label='Hz',
        help='direct conversion LO frequency of the RX',
    )
    def lo_frequency(self):
        # there is only one RX LO, shared by both channels
        return self.backend.setdefault('lo_frequency', 3e9)

    @lo_frequency.setter
    def _(self, value):
        # there is only one RX LO, shared by both channels
        self.backend['lo_frequency'] = value

    center_frequency = lo_frequency.corrected_from_expression(
        lo_frequency + lo_offset,
        help='RF frequency at the center of the RX baseband',
        label='Hz',
    )

    @attr.method.float(
        min=0,
        cache=True,
        label='Hz',
        help='sample rate before resampling',
    )
    def backend_sample_rate(self):
        return self.backend.setdefault('backend_sample_rate', 25e6)

    @backend_sample_rate.setter
    def _(self, value):
        self.backend['backend_sample_rate'] = value

    sample_rate = backend_sample_rate.corrected_from_expression(
        backend_sample_rate / _downsample,
        label='Hz',
        help='sample rate of acquired waveform',
    )

    @attr.method.float(sets=False, label='Hz')
    def realized_sample_rate(self):
        """the self-reported "actual" sample rate of the radio"""
        return self.sample_rate()

    @attr.method.bool(cache=True)
    def channel_enabled(self):
        # this is only called at most once, due to cache=True
        raise ValueError('must set channel_enabled once before reading')

    @channel_enabled.setter
    def _(self, enable: bool):
        self.backend['channel_enabled'] = enable

    @attr.method.float(label='dB', help='SDR hardware gain')
    def gain(self):
        return self.backend.setdefault('gain', 0)

    @gain.setter
    def _(self, gain: float):
        self.backend['gain'] = gain

    @attr.method.float(label='dB', help='SDR TX hardware gain')
    @channel_kwarg
    def tx_gain(self, gain: float = lb.Undefined, /, *, channel: int = 0):
        if gain is lb.Undefined:
            return self.backend.setdefault('tx_gain', default=0)
        else:
            self.backend['tx_gain'] = gain

    def open(self):
        self._logger.propagate = False
        self.backend = {}
        self.channel(0)

    @attr.property.str(inherit=True)
    def id(self):
        return 'null'

    @property
    def master_clock_rate(self):
        return 125e6

    arm = SoapyRadioDevice.arm

    def _read_stream(self, N) -> np.ndarray:
        xp = import_cupy_with_fallback()
        ret = self._inbuf.view('complex64')[:N]
        if self.resource == 'empty':
            pass
        elif self.resource == 'noise':
            ret[:] = (
                xp.random.normal(scale=1e-3, size=2 * N)
                .astype('float32')
                .view('complex64')
            )
        elif self.resource == 'tone':
            ret[:] = (
                xp.random.normal(scale=1e-3, size=2 * N)
                .astype('float32')
                .view('complex64')
            )
            t = np.arange(N) / self.sample_rate()
            f_cw = self.sample_rate() / 5
            ret[:] += xp.sin((2 * np.pi * f_cw) * t)
        return self._inbuf.view('complex64')[:N]
