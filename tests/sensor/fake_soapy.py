"""a fake SoapySDR module with an Airstack-shaped device, for installation at the
`striqt.sensor.lib.sources.soapy.SoapySDR` seam.

The device streams a physically meaningful waveform: complex Gaussian noise from a
receiver with a per-port noise figure after a noise source that a "noise diode" can
switch on, plus optional CW tones, all through the receiver model in
`sweep_strategies` so that calibration sweeps run against it have closed-form
expected results.
"""

from __future__ import annotations

import dataclasses
import time
import types
from typing import NamedTuple

import numpy as np
from sweep_strategies import (
    T_REF,
    YFACTOR_ENR_DB,
    YFACTOR_NF_DB,
    noise_figure_lookup,
    receiver_gain,
    yfactor_rms_power,
)

# from SoapySDR/Constants.h and SoapySDR/Errors.h
CONSTANTS = {
    'SOAPY_SDR_TX': 0,
    'SOAPY_SDR_RX': 1,
    'SOAPY_SDR_CF32': 'CF32',
    'SOAPY_SDR_CS16': 'CS16',
    'SOAPY_SDR_END_BURST': 1 << 1,
    'SOAPY_SDR_HAS_TIME': 1 << 2,
    'SOAPY_SDR_TIMEOUT': -1,
    'SOAPY_SDR_STREAM_ERROR': -2,
    'SOAPY_SDR_CORRUPTION': -3,
    'SOAPY_SDR_OVERFLOW': -4,
    'SOAPY_SDR_NOT_SUPPORTED': -5,
    'SOAPY_SDR_TIME_ERROR': -6,
    'SOAPY_SDR_UNDERFLOW': -7,
}
_ERROR_CODES = {k: v for k, v in CONSTANTS.items() if isinstance(v, int) and v < 0}
ERROR_NAMES = {v: k.replace('SOAPY_SDR_', '') for k, v in _ERROR_CODES.items()}

RX = CONSTANTS['SOAPY_SDR_RX']
DRIVER_KEY = 'SoapyAIRT'
HARDWARE_KEY = 'AIR7101B'
NUM_RX_PORTS = 2
GAIN_RANGE = (-30.0, 0.0, 0.5)
FREQUENCY_RANGE = (300e6, 6e9)
SAMPLE_RATE_RANGE = (3.9e6, 125e6)
MASTER_CLOCK_RATE = 125e6
INT16_FULL_SCALE = 32767


class StreamResult(NamedTuple):
    ret: int
    flags: int = 0
    timeNs: int = 0
    chanMask: int = 0


class Range:
    def __init__(self, minimum, maximum, step=0.0):
        self._values = (float(minimum), float(maximum), float(step))

    def minimum(self):
        return self._values[0]

    def maximum(self):
        return self._values[1]

    def step(self):
        return self._values[2]


class ArgInfo:
    """attribute-only, like the SWIG object: indexing it raises TypeError"""

    def __init__(self, key, *, units='', value='', options=()):
        self.key = key
        self.name = key.replace('_', ' ').title()
        self.description = f'the {key}'
        self.units = units
        self.type = 2
        self.value = value
        self.range = Range(0.0, 0.0)
        self.options = tuple(options)


@dataclasses.dataclass
class SignalModel:
    """receiver and stimulus parameters shared by every device of one fake module"""

    T0: float = T_REF
    enr_dB: float = YFACTOR_ENR_DB
    nf_dB: dict = dataclasses.field(default_factory=lambda: dict(YFACTOR_NF_DB))
    front_end_gain_dB: float = 50.0
    fs_mW: float = 1.0
    diode_on: dict = dataclasses.field(default_factory=dict)
    # port -> (RF frequency in Hz, power in dBm at the receiver input)
    tones: dict = dataclasses.field(default_factory=dict)
    seed: int = 0
    pps_present: bool = True

    def noise_variance(self, port, gain_dB, sample_rate, center_frequency):
        """total noise power per complex sample, in full-scale units"""
        return yfactor_rms_power(
            gain_dB,
            noise_figure_lookup(self.nf_dB, port, center_frequency),
            sample_rate,
            diode_on=self.diode_on.get(port, False),
            enr_dB=self.enr_dB,
            T0=self.T0,
            front_end_gain_dB=self.front_end_gain_dB,
            fs_mW=self.fs_mW,
        )

    def waveform(self, port, start, count, *, gain_dB, lo_frequency, sample_rate, rng):
        """`count` samples starting `start` samples after stream activation"""
        sigma2 = self.noise_variance(port, gain_dB, sample_rate, lo_frequency)
        x = rng.normal(scale=np.sqrt(sigma2 / 2), size=(2, count))
        x = (x[0] + 1j * x[1]).astype('complex64')

        if port in self.tones:
            f_rf, p_dBm = self.tones[port]
            gain = receiver_gain(gain_dB, self.front_end_gain_dB, self.fs_mW)
            amplitude = np.sqrt(gain * 10 ** (p_dBm / 10))
            n = np.arange(start, start + count)
            phase = 2 * np.pi * (f_rf - lo_frequency) * n / sample_rate
            x += (amplitude * np.exp(1j * phase)).astype('complex64')

        return x


@dataclasses.dataclass
class FakeStream:
    channels: tuple
    format: str
    active: bool = False
    activate_time_ns: int = 0
    delivered: int = 0


class FakeDevice:
    """the subset of SoapySDR.Device that striqt touches, shaped like an Airstack"""

    def __init__(self, module: FakeSoapySDR, **kwargs):
        self.module = module
        self.model = module.model
        self.kwargs = kwargs
        self.calls: list[tuple] = []
        self.rng = np.random.default_rng(self.model.seed)

        self.gains = {}
        self.gain_modes = {}
        self.frequencies = {}
        self.sample_rate = MASTER_CLOCK_RATE
        self.time_source = 'internal'
        self.clock_source = 'internal'
        self.master_clock_rate = MASTER_CLOCK_RATE
        self.streams: list[FakeStream] = []
        self.closed = False

        # the hardware clock boots at zero, so the first host sync must set it
        self.clock_offset_ns = -time.time_ns()
        self.pps_count = 0
        self.pps_time_set_ns = None

        self.registers = {}
        self.channel_sensors: dict[str, str] = {}
        self.global_sensors = {'xcvr_temp': '41.5'}

        # one entry is popped per readStream call; an int is returned as its ret
        self.fault_queue: list = []
        self.max_read: int | None = None

    def _record(self, *call):
        self.calls.append(call)

    def calls_named(self, *names) -> list[tuple]:
        return [c for c in self.calls if c[0] in names]

    # %% identification and capabilities
    def getDriverKey(self):
        return DRIVER_KEY

    def getHardwareKey(self):
        return HARDWARE_KEY

    def getHardwareInfo(self):
        return {'firmware': '1.0.0', 'fpga': 'fake'}

    def getNumChannels(self, direction):
        return NUM_RX_PORTS if direction == RX else 0

    def hasHardwareTime(self, what=''):
        return True

    def listClockSources(self):
        return ['internal', 'external']

    def listTimeSources(self):
        return ['internal', 'external', 'gps']

    def listRegisterInterfaces(self):
        return ['FPGA']

    def listGPIOBanks(self):
        return []

    def listUARTs(self):
        return []

    def listSensors(self, *args):
        if args:
            return list(self.channel_sensors)
        return list(self.global_sensors)

    def getSensorInfo(self, *args):
        key = args[-1]
        return ArgInfo(key, units='C' if key == 'xcvr_temp' else '')

    def readSensor(self, *args):
        key = args[-1]
        if len(args) > 1:
            return self.channel_sensors[key]
        return self.global_sensors[key]

    def readSensorFloat(self, key):
        return float(self.global_sensors[key])

    def getSettingInfo(self, *args):
        return []

    def hasDCOffsetMode(self, direction, channel):
        return True

    def hasDCOffset(self, direction, channel):
        return False

    def hasIQBalance(self, direction, channel):
        return False

    def listGains(self, direction, channel):
        return ['PGA']

    def getGainRange(self, direction, channel, name=None):
        return Range(*GAIN_RANGE)

    def listFrequencies(self, direction, channel):
        return ['RF']

    def getFrequencyRange(self, direction, channel, name=None):
        return [Range(*FREQUENCY_RANGE)]

    def getFrequencyArgsInfo(self, direction, channel):
        return []

    def getChannelInfo(self, direction, channel):
        return {'channel': str(channel)}

    def getFullDuplex(self, direction, channel):
        return False

    def hasGainMode(self, direction, channel):
        return True

    def getStreamFormats(self, direction, channel):
        return ['CF32', 'CS16']

    def getStreamArgsInfo(self, direction, channel):
        return []

    def listAntennas(self, direction, channel):
        return ['RX']

    def getSampleRateRange(self, direction, channel):
        return [Range(*SAMPLE_RATE_RANGE)]

    def getMasterClockRates(self):
        return [MASTER_CLOCK_RATE]

    def getBandwidthRange(self, direction, channel):
        return [Range(*SAMPLE_RATE_RANGE)]

    # %% configuration
    def setGainMode(self, direction, channel, automatic):
        self._record('setGainMode', direction, channel, automatic)
        self.gain_modes[direction, channel] = automatic

    def setGain(self, direction, channel, value):
        self._record('setGain', direction, channel, value)
        self.gains[direction, channel] = value

    def getGain(self, direction, channel):
        return self.gains.get((direction, channel), 0.0)

    def setFrequency(self, direction, channel, value):
        self._record('setFrequency', direction, channel, value)
        self.frequencies[direction, channel] = value

    def setSampleRate(self, direction, channel, value):
        self._record('setSampleRate', direction, channel, value)
        # one clock drives every RX channel (specs.SoapySource.shared_rx_sample_clock)
        self.sample_rate = value

    def setTimeSource(self, source):
        self._record('setTimeSource', source)
        self.time_source = source

    def setClockSource(self, source):
        self._record('setClockSource', source)
        self.clock_source = source

    def setMasterClockRate(self, rate):
        self._record('setMasterClockRate', rate)
        self.master_clock_rate = rate

    # %% hardware time
    def getHardwareTime(self, what=''):
        if what == 'pps':
            if self.model.pps_present:
                self.pps_count += 1
            return self.pps_count * 1_000_000_000
        return time.time_ns() + self.clock_offset_ns

    def setHardwareTime(self, time_ns, what=''):
        self._record('setHardwareTime', time_ns, what)
        if what == 'pps':
            self.pps_time_set_ns = time_ns
        else:
            self.clock_offset_ns = time_ns - time.time_ns()

    # %% registers
    def readRegister(self, name, addr):
        return self.registers.get((name, addr), 0)

    def writeRegister(self, name, addr, value):
        self._record('writeRegister', name, addr, value)
        self.registers[name, addr] = value

    # %% streaming
    def setupStream(self, direction, format, channels=(), args=None):
        self._record('setupStream', direction, format, tuple(channels))
        stream = FakeStream(channels=tuple(channels), format=format)
        self.streams.append(stream)
        return stream

    def _check_stream(self, stream):
        if stream not in self.streams:
            raise ValueError('invalid parameter')

    def activateStream(self, stream, flags=0, timeNs=0, numElems=0):
        self._record('activateStream', flags, timeNs)
        self._check_stream(stream)
        stream.active = True
        stream.delivered = 0
        stream.activate_time_ns = timeNs or self.getHardwareTime('now')

    def deactivateStream(self, stream, flags=0, timeNs=0):
        self._record('deactivateStream')
        self._check_stream(stream)
        stream.active = False

    def closeStream(self, stream):
        self._record('closeStream')
        self._check_stream(stream)
        stream.active = False
        self.streams.remove(stream)

    def readStream(self, stream, buffs, numElems, flags=0, timeoutUs=100000):
        self._record('readStream', numElems, timeoutUs)

        if self.fault_queue:
            fault = self.fault_queue.pop(0)
            if fault is not None:
                return StreamResult(fault, 0, self._stream_time(stream))
        if not stream.active:
            return StreamResult(CONSTANTS['SOAPY_SDR_TIMEOUT'], 0, 0)

        count = numElems if self.max_read is None else min(numElems, self.max_read)
        count = min([count] + [len(buf) // 2 for buf in buffs])
        time_ns = self._stream_time(stream)

        for buf, channel in zip(buffs, stream.channels):
            x = self.model.waveform(
                channel,
                stream.delivered,
                count,
                gain_dB=self.gains.get((RX, channel), 0.0),
                lo_frequency=self.frequencies.get((RX, channel), 0.0),
                sample_rate=self.sample_rate,
                rng=self.rng,
            )
            interleaved = x.view('float32')
            if stream.format == 'CF32':
                buf[: 2 * count] = interleaved
            elif stream.format == 'CS16':
                buf[: 2 * count] = np.round(interleaved * INT16_FULL_SCALE)
            else:
                raise ValueError(f'unsupported stream format {stream.format!r}')

        stream.delivered += count
        mask = sum(1 << ch for ch in stream.channels)
        return StreamResult(count, CONSTANTS['SOAPY_SDR_HAS_TIME'], time_ns, mask)

    def _stream_time(self, stream):
        return stream.activate_time_ns + round(
            stream.delivered * 1e9 / self.sample_rate
        )

    # %% lifetime
    def close(self):
        self._record('close')
        self.closed = True

    def __del__(self):
        pass


class FakeSoapySDR:
    """stands in for the SoapySDR module"""

    Range = Range
    ArgInfo = ArgInfo
    StreamResult = StreamResult

    def __init__(self, model: SignalModel | None = None):
        self.model = model if model is not None else SignalModel()
        self.devices: list[FakeDevice] = []
        for name, value in CONSTANTS.items():
            setattr(self, name, value)
        # SoapySource.close checks this guard against a torn-down module
        self._SoapySDR = types.SimpleNamespace(Device_deactivateStream=object())

    def Device(self, args=None):
        # the sequence form (SoapySDR issue 472 workaround) returns a list of devices
        if isinstance(args, (list, tuple)):
            return [self._open(dict(a)) for a in args]
        return self._open(dict(args or {}))

    def _open(self, kwargs):
        device = FakeDevice(self, **kwargs)
        self.devices.append(device)
        return device

    @staticmethod
    def errToStr(code):
        return ERROR_NAMES.get(code, 'UNKNOWN')


def install_fake_soapy(monkeypatch, model: SignalModel | None = None) -> FakeSoapySDR:
    from striqt.sensor.lib.sources import soapy

    fake = FakeSoapySDR(model)
    monkeypatch.setattr(soapy, 'SoapySDR', fake)
    return fake
