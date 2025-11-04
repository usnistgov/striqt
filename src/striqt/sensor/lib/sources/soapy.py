from __future__ import annotations

import contextlib
import functools
import time
import typing

from . import base
from .. import captures, specs, util

if typing.TYPE_CHECKING:
    import numpy as np
    import SoapySDR
else:
    np = util.lazy_import('numpy')
    SoapySDR = util.lazy_import('SoapySDR')


_TS = typing.TypeVar('_TS', bound=specs.SoapySetup)
_TC = typing.TypeVar('_TC', bound=specs.SoapyCapture)


def _tuplize_port(obj: specs.PortType) -> tuple[int, ...]:
    if isinstance(obj, (tuple, list)):
        return obj
    else:
        return (int(obj),)


class Range(specs.SpecBase, kw_only=True, frozen=True, cache_hash=True):
    """Represents a range with a minimum, maximum, and step."""

    minimum: float
    maximum: float
    step: float

    @classmethod
    def from_soapy(cls, r: 'SoapySDR.Range') -> typing.Self:
        return cls(minimum=r.minimum(), maximum=r.maximum(), step=r.step())

    @classmethod
    def from_soapy_tuple(cls, seq: typing.Any) -> tuple[typing.Self, ...]:
        return tuple([cls.from_soapy(r) for r in seq])


class ArgInfo(specs.SpecBase, kw_only=True, frozen=True, cache_hash=True):
    """Represents information about a configurable device argument."""

    name: str
    description: str
    units: str
    type: int
    value: str
    range: Range
    options: tuple[str, ...]

    @classmethod
    def from_soapy(cls, arg: 'SoapySDR.ArgInfo') -> typing.Self:
        return cls(
            name=arg.name,
            description=arg.description,
            units=arg.units,
            type=arg.type,
            value=arg.value,
            range=Range.from_soapy(arg.range),
            options=tuple(arg.options),
        )

    @classmethod
    def from_soapy_map(
        cls, soapy_args: list[SoapySDR.ArgInfo]
    ) -> dict[str, typing.Self]:
        return {arg.key: cls.from_soapy(arg) for arg in soapy_args}


class SensorReading(specs.SpecBase, kw_only=True, frozen=True, cache_hash=True):
    """Represents a sensor and its current reading."""

    name: str
    info: ArgInfo
    reading: str


# class NativeFormat(NamedTuple):
#     """Represents the native stream format and its full-scale value."""
#     format: str
#     full_scale: float


class PortInfo(specs.SpecBase, kw_only=True, frozen=True, cache_hash=True):
    """Holds all capability metadata for a single RX or TX channel."""

    port_info: dict[str, str]
    full_duplex: bool
    agc: bool
    stream_formats: tuple[str, ...]
    # native_format: NativeFormat
    stream_args: dict[str, ArgInfo]
    antennas: tuple[str, ...]
    corrections: tuple[str, ...]
    gains: dict[str, Range]
    full_gain_range: Range
    frequencies: dict[str, tuple[Range, ...]]
    full_freq_range: Range
    tune_args: dict[str, ArgInfo]
    backend_sample_rate_range: Range
    base_clock_rates: tuple[float, ...]
    bandwidths: tuple[Range, ...]
    sensors: dict[str, SensorReading]
    settings: dict[str, ArgInfo]


class SoapyInfo(base.BaseSourceInfo, kw_only=True, frozen=True, cache_hash=True):
    """Top-level container for all device capabilities metadata."""

    driver: str
    hardware: str
    hardware_info: dict[str, str]
    num_rx_ports: int
    num_tx_ports: int
    has_timestamps: bool
    clock_sources: tuple[str, ...]
    time_sources: tuple[str, ...]
    global_sensors: dict[str, SensorReading]
    registers: tuple[str, ...]
    settings: dict[str, ArgInfo]
    gpios: tuple[str, ...]
    uarts: tuple[str, ...]
    rx_ports: tuple[PortInfo, ...]
    tx_ports: tuple[PortInfo, ...]

    # def to_capture_cls(self, base_cls: type[_TC]=specs.SoapyCapture) -> type[_TC]:
    #     PortType = specs.replace_meta(specs.PortType, ge=0, lt=self.num_rx_ports)

    #     return base_cls

    # def to_setup_cls(self, base_cls: type[_TS]=specs.SoapySetup) -> type[_TS]:
    #     clock_rates = self.rx_ports[0].base_clock_rates
    #     BaseClockRateType = specs.replace_type(specs.BaseClockRateType, typing.Union[*clock_rates])

    #     return base_cls


def _probe_channel(device: SoapySDR.Device, direction: int, port: int) -> PortInfo:
    """Probes a single channel and returns its capabilities."""

    args = (direction, port)

    corrections_list = []
    if device.hasDCOffsetMode(*args):
        corrections_list.append('DC removal')
    if device.hasDCOffset(*args):
        corrections_list.append('DC offset')
    if device.hasIQBalance(*args):
        corrections_list.append('IQ balance')

    gains = {}
    for name in device.listGains(*args):
        soapy_gain_range = device.getGainRange(*args, name)
        gain_range = Range(
            minimum=soapy_gain_range.minimum(),
            maximum=soapy_gain_range.maximum(),
            step=soapy_gain_range.step(),
        )
        gains[name] = gain_range

    freqs = {
        name: Range.from_soapy_tuple(device.getFrequencyRange(*args, name))
        for name in device.listFrequencies(*args)
    }

    sensors = {
        key: SensorReading(
            name=device.getSensorInfo(*args, key).name,
            info=ArgInfo.from_soapy(device.getSensorInfo(*args, key)[0]),
            reading=device.readSensor(*args, key),
        )
        for key in device.listSensors(*args)
    }

    return PortInfo(
        port_info=dict(device.getChannelInfo(*args)),
        full_duplex=device.getFullDuplex(*args),
        agc=device.hasGainMode(*args),
        stream_formats=tuple(device.getStreamFormats(*args)),
        # native_format=NativeFormat(format=native_fmt, full_scale=full_scale),
        stream_args=ArgInfo.from_soapy_map(device.getStreamArgsInfo(*args)),
        antennas=tuple(device.listAntennas(*args)),
        corrections=tuple(corrections_list),
        gains=gains,
        full_gain_range=Range.from_soapy(device.getGainRange(*args)),
        frequencies=freqs,
        full_freq_range=Range.from_soapy(device.getFrequencyRange(*args)),
        tune_args=ArgInfo.from_soapy_map(device.getFrequencyArgsInfo(*args)),
        backend_sample_rate_range=Range.from_soapy(device.getSampleRateRange(*args)),
        base_clock_rates=tuple(device.getMasterClockRates()),
        bandwidths=Range.from_soapy_tuple(device.getBandwidthRange(*args)),
        sensors=sensors,
        settings=ArgInfo.from_soapy_map(device.getSettingInfo(*args)),
    )


def probe_soapy_info(device: SoapySDR.Device) -> SoapyInfo:
    """
    Probes a SoapySDR device and returns its capabilities as a nested NamedTuple.

    This function mirrors the C++ probing utility, but instead of printing
    the information, it packages it into a structured Python object for
    programmatic access.

    Args:
        device: An open SoapySDR.Device

    Returns:
        A RadioInfo named tuple containing capability metadata for the device.
    """
    global_sensors = {
        key: SensorReading(
            name=device.getSensorInfo(key).name,
            info=ArgInfo.from_soapy(device.getSensorInfo(key)),
            reading=device.readSensor(key),
        )
        for key in device.listSensors()
    }

    num_rx = device.getNumChannels(SoapySDR.SOAPY_SDR_RX)
    num_tx = device.getNumChannels(SoapySDR.SOAPY_SDR_TX)

    rx_channels = tuple(
        _probe_channel(device, SoapySDR.SOAPY_SDR_RX, i) for i in range(num_rx)
    )
    tx_channels = tuple(
        _probe_channel(device, SoapySDR.SOAPY_SDR_TX, i) for i in range(num_tx)
    )

    return SoapyInfo(
        driver=device.getDriverKey(),
        hardware=device.getHardwareKey(),
        hardware_info=device.getHardwareInfo(),
        num_rx_ports=num_rx,
        num_tx_ports=num_tx,
        has_timestamps=device.hasHardwareTime(),
        clock_sources=tuple(device.listClockSources()),
        time_sources=tuple(device.listTimeSources()),
        global_sensors=global_sensors,
        registers=tuple(device.listRegisterInterfaces()),
        settings=ArgInfo.from_soapy_map(device.getSettingInfo()),
        gpios=tuple(device.listGPIOBanks()),
        uarts=tuple(device.listUARTs()),
        rx_ports=rx_channels,
        tx_ports=tx_channels,
    )


@contextlib.contextmanager
def read_retries(radio: SoapySourceBase) -> typing.Generator[None]:
    """in this context, retry radio.read_iq on stream errors"""

    if radio._setup.receive_retries == 0:
        yield
        return

    def prepare_retry(*args, **kws):
        assert radio._rx_stream is not None, 'soapy stream is not open'
        assert radio._device is not None, 'soapy device is not open'

        radio._rx_stream.enable(False)
        radio._buffers.hold_buffer_swap = True
        if not radio._setup.time_sync_every_capture:
            radio._sync_time_source(radio._device)

    decorate = util.retry(
        (base.ReceiveStreamError, OverflowError),
        tries=radio._setup.receive_retries + 1,
        exception_func=prepare_retry,
    )

    radio.read_iq, original = decorate(radio.read_iq), radio.read_iq

    try:
        yield
    finally:
        radio.read_iq = original


class RxStream:
    """manage the state of the RX stream"""

    def __init__(
        self,
        radio: SoapySourceBase[_TS, _TC],
        on_overflow: base.OnOverflowType = 'except',
    ):
        self.device = radio._device
        self.setup = radio._setup
        self.info = radio.info

        assert self.device is not None

        self.checked_timestamp = False
        self.stream = None
        self._enabled: bool = False
        self._on_overflow: base.OnOverflowType = on_overflow
        self._ports: specs.PortType = ()

    @util.stopwatch('stream initialization', 'source')
    def open(self, ports: specs.PortType | None = None):
        assert self.device is not None, 'soapy source is not Open'

        if self.stream is not None and self._ports is not None:
            return

        if ports is not None:
            ports = _tuplize_port(ports)
        elif self.setup.stream_all_rx_ports:
            ports = tuple(range(self.info.num_rx_ports))
        else:
            ports = self._ports

        self._ports = ports

        if self.setup.transport_dtype == 'int16':
            stype = SoapySDR.SOAPY_SDR_CS16
        elif self.setup.transport_dtype == 'float32':
            stype = SoapySDR.SOAPY_SDR_CF32
        else:
            raise ValueError(
                f'unsupported transport type {self.setup.transport_dtype!r}'
            )

        with self._minimized_rx_gain():
            self.stream = self.device.setupStream(SoapySDR.SOAPY_SDR_RX, stype, ports)

    @contextlib.contextmanager
    def _minimized_rx_gain(self):
        """minimize the gain setting while in the context"""
        assert self.device is not None, 'soapy source is not Open'

        start_gains = {}

        for port, info in enumerate(self.info.rx_ports):
            start_gains[port] = self.device.getGain(SoapySDR.SOAPY_SDR_RX, port)
            min_gain = info.full_gain_range.minimum
            self.device.setGain(SoapySDR.SOAPY_SDR_RX, port, min_gain)

        yield

        # restore the initial gains
        for port, gain in start_gains.items():
            self.device.setGain(SoapySDR.SOAPY_SDR_RX, port, gain)

    def close(self):
        if getattr(self, '_rx_stream', None) is None:
            return

        try:
            self.enable(False)
        except ValueError:
            # channel not yet set
            pass

        try:
            if self.device is not None:
                self.device.closeStream(self.stream)
        except ValueError as ex:
            if 'invalid parameter' in str(ex):
                # already closed
                pass
            else:
                raise

        self.stream = None
        self._ports = ()

    def enable(self, enable: bool):
        assert self.device is not None, 'soapy source is not open'
        if enable == self._enabled:
            return

        if enable:
            if self.device.hasHardwareTime():
                kws = {'flags': SoapySDR.SOAPY_SDR_HAS_TIME}
            else:
                kws = {}

            if self.setup.rx_enable_delay is not None:
                delay_ns = round(self.setup.rx_enable_delay * 1e9)
                time_ns = self.device.getHardwareTime('now') + delay_ns
                kws['timeNs'] = time_ns

            self.checked_timestamp = False
            self.device.activateStream(self.stream, **kws)

        elif self.stream is not None:
            self.device.deactivateStream(self.stream)

        self._enabled = enable

    def read(
        self,
        buffers,
        offset,
        count,
        timeout_sec,
        *,
        last_sync_time: int | None,
        on_overflow: base.OnOverflowType | None = None,
    ) -> tuple[int, int]:
        assert self.device is not None, 'soapy source is not open'

        total_timeout = self.setup.rx_enable_delay + timeout_sec + 0.5

        rx_result = self.device.readStream(
            self.stream,
            [buf[offset * 2 :] for buf in buffers],
            count,
            timeoutUs=round(total_timeout * 1e6),
        )

        if not self.checked_timestamp and self.device.hasHardwareTime():
            # require a valid timestamp only for the first read after channel enables.
            # subsequent reads are treated as contiguous unless TimeoutError is raised
            sync_time_ns = last_sync_time
            self.checked_timestamp = True
        else:
            sync_time_ns = None

        return self.validate_stream_read(
            rx_result,
            on_overflow=on_overflow or self._on_overflow,
            sync_time_ns=sync_time_ns,
        )

    @staticmethod
    def validate_stream_read(
        sr: SoapySDR.StreamResult,
        on_overflow: base.OnOverflowType,
        sync_time_ns: int | None = None,
    ) -> tuple[int, int]:
        """track the number of samples received and remaining in a read stream.

        Args:
            sr: the return value from self.backend.readStream
            count: the expected number of samples (1 (I,Q) pair each)

        Returns:
            (samples received, start clock timestamp (ns))
        """
        msg = None

        # ensure the proper number of waveform samples was read
        if sr.ret >= 0:
            result = sr.ret, sr.timeNs
        elif sr.ret == SoapySDR.SOAPY_SDR_OVERFLOW:
            if on_overflow == 'except':
                msg = f'{time.perf_counter()}: overflow'
                raise OverflowError(msg)
            result = 0, sr.timeNs
        else:
            err_str = SoapySDR.errToStr(sr.ret)
            raise base.ReceiveStreamError(f'{err_str} (error code {sr.ret})')

        if sync_time_ns is not None and sync_time_ns > sr.timeNs:
            raise base.ReceiveStreamError(f'invalid timestamp from before last sync')

        return result

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def capture_changes_port(self, capture: specs.SoapyCapture) -> bool:
        return _tuplize_port(capture.port) == self._ports

    def set_ports(self, ports: specs.PortType):
        if self.setup.stream_all_rx_ports:
            # in this case, the stream is controlled only on open
            return

        elif getattr(self, 'stream', None) is not None:
            if self._ports == ports:
                # already set up
                return
            else:
                self.close()

        # if we make it this far, we need to build and enable the RX stream
        self.open(ports)


class HardwareTimeSync:
    def __init__(self, time_source: specs.TimeSourceType):
        self.last_sync_time: int | None = None
        self.time_source = time_source

    def __call__(self, device: 'SoapySDR.Device') -> int | None:
        if self.time_source in ('internal', 'host'):
            self.last_sync_time = self.to_host_os(device)
        elif self.time_source in ('gps', 'external'):
            self.last_sync_time = self.to_external_pps(device)
        else:
            raise TypeError('unsupported time source {self.time_source!r}')

    def to_host_os(self, device: 'SoapySDR.Device') -> int | None:
        if not device.hasHardwareTime():
            raise IOError('device does not expose hardware time')

        hardware_time = device.getHardwareTime('now') / 1e9
        if abs(hardware_time - time.time()) >= 0.2:
            sync_time = round(time.time() * 1e9)
            device.setHardwareTime(sync_time, 'now')
            return sync_time
        else:
            return self.last_sync_time

    def to_external_pps(self, device: 'SoapySDR.Device') -> int | None:
        if not device.hasHardwareTime():
            raise IOError('device does not expose hardware time')

        # let a PPS transition pass to avoid race conditions involving
        # applying the time of the next PPS
        init_pps_time = device.getHardwareTime('pps')
        start_time = time.perf_counter()
        while init_pps_time == device.getHardwareTime('pps'):
            if time.perf_counter() - start_time > 1.5:
                raise RuntimeError('no pps input detected for external time source')
            else:
                time.sleep(10e-3)

        # PPS transition occurred, should be safe to snag system time and apply it
        sys_time_now = time.time()
        full_secs = int(sys_time_now)
        frac_secs = sys_time_now - full_secs
        if frac_secs > 0.8:
            # System time is lagging behind the PPS transition
            full_secs += 1
        elif frac_secs > 0.2:
            # System time and PPS are off, warn caller
            util.get_logger('acquisition').warning(
                f'system time and PPS out of sync by {frac_secs:0.3f}s, check NTP'
            )
        time_to_set_ns = int((full_secs + 1) * 1e9)
        device.setHardwareTime(time_to_set_ns, 'pps')
        return time_to_set_ns


@functools.lru_cache
def split_this_prev_captures(
    c1: _TC, c2: _TC | None, is_new: bool
) -> tuple[list[_TC], list[_TC] | list[None]]:
    # a list with 1 capture per port
    c1_split = captures.split_capture_ports(c1)

    # any changes to the port index
    if c2 is None or is_new:
        c2_split = len(c1_split) * [None]
    else:
        c2_split = captures.split_capture_ports(c2)

    return c1_split, c2_split


def set_gain(source: SoapySourceBase, capture: specs.SoapyCapture, ports_changed: bool):
    assert source._device is not None, 'soapy source is not open'

    this_split, prev_split = split_this_prev_captures(
        capture, source._capture, ports_changed
    )

    for prev, this in zip(prev_split, this_split):
        if prev is None or this.gain != this.gain:
            source._device.setGain(SoapySDR.SOAPY_SDR_RX, this.port, this.gain)


def set_center_frequency(
    source: SoapySourceBase, capture: specs.SoapyCapture, ports_changed: bool
):
    assert source._device is not None, 'soapy source is not open'

    this_split, prev_split = split_this_prev_captures(
        capture, source._capture, ports_changed
    )
    fs_base = source._setup.base_sample_rate

    # set center frequency with lo_shift
    for prev, this in zip(prev_split, this_split):
        this_resamp = base.design_capture_resampler(fs_base, this)

        if prev is None:
            prev_backend_fc = None
        else:
            prev_resamp = base.design_capture_resampler(fs_base, prev)
            prev_backend_fc = prev.center_frequency - prev_resamp['lo_offset']
        backend_fc = this.center_frequency - this_resamp['lo_offset']

        if backend_fc != prev_backend_fc:
            source._device.setFrequency(SoapySDR.SOAPY_SDR_RX, this.port, backend_fc)


def set_sample_rate(
    source: SoapySourceBase, capture: specs.SoapyCapture, ports_changed: bool
):
    assert source._device is not None

    fs_base = source._setup.base_sample_rate

    this_fs = base.design_capture_resampler(fs_base, capture)['fs_sdr']

    if source._capture is None:
        prev_fs = None
    else:
        prev_fs = base.design_capture_resampler(fs_base, source._capture)['fs_sdr']

    if this_fs == prev_fs:
        return

    if source._setup.shared_rx_sample_clock:
        capture_per_port = [capture]
    else:
        capture_per_port = captures.split_capture_ports(capture)

    for c in capture_per_port:
        source._device.setSampleRate(SoapySDR.SOAPY_SDR_RX, c.port, this_fs)


class SoapySourceBase(
    base.SourceBase, base.HasSetupType[_TS], base.HasCaptureType[_TC]
):
    """Applies SoapySDR for device control and acquisition"""

    _device: 'SoapySDR.Device | None' = None
    _rx_stream: RxStream | None = None

    def close(self):
        if self._device is None:
            return

        if (
            SoapySDR is None
            or SoapySDR._SoapySDR is None
            or SoapySDR._SoapySDR.Device_deactivateStream is None
            or SoapySDR.Device is None
        ):
            # soapy's underlying libraries have been deconstructed
            # too far to proceed
            return

        self._device.__del__ = lambda: None

        if self._rx_stream is not None:
            self._rx_stream.close()

        self._device.close()
        util.get_logger('acquisition').info('closed soapy radio')
        super().close()

    @util.stopwatch('connected to radio', 'source', util.PERFORMANCE_INFO)
    def _connect(self, spec: _TS):
        if spec.device_kwargs:
            # prevent race conditions in threaded accesses to the Soapy driver
            self._device = SoapySDR.Device(spec.device_kwargs)
        else:
            self._device = SoapySDR.Device()

    @util.stopwatch('configured soapy radio streaming', 'source', util.PERFORMANCE_INFO)
    def _apply_setup(self, spec: _TS) -> _TS | None:
        assert self._device is not None, 'soapy device is not open'

        for p in range(self.info.num_rx_ports):
            self._device.setGainMode(SoapySDR.SOAPY_SDR_RX, p, False)

        self._sync_time_source: HardwareTimeSync = HardwareTimeSync(spec.time_source)

        if spec.time_source == 'host':
            self._device.setTimeSource('internal')
            on_overflow = 'log'
            periodic_trigger = getattr(spec, 'periodic_trigger', None)
            if periodic_trigger is not None:
                util.get_logger('acquisition').warning(
                    'periodic trigger with host time will suffer from inaccuracy on overflow'
                )
        else:
            self._device.setTimeSource(spec.time_source)
            on_overflow = 'except'

        self._rx_stream = RxStream(self, on_overflow=on_overflow)

        self._device.setClockSource(spec.clock_source)
        self._device.setMasterClockRate(spec.base_clock_rate)

        if not spec.time_sync_every_capture:
            self._sync_time_source(self._device)

        self._rx_stream.open()

    def _prepare_capture(self, capture: _TC) -> _TC | None:
        if (
            capture == self._capture
            and self._setup.gapless_repeats
            or self._rx_stream is None
        ):
            # the one case where we leave it running
            return

        assert self._device is not None

        self._rx_stream.enable(False)

        # manage changes to the ports
        ports_changed = self._rx_stream.capture_changes_port(capture)
        if self._capture is None or ports_changed:
            self._rx_stream.set_ports(capture.port)

        # gain before center frequency to accommodate attenuator settling time
        set_gain(self, capture, ports_changed)
        set_center_frequency(self, capture, ports_changed)
        set_sample_rate(self, capture, ports_changed)

    def _read_stream(
        self,
        buffers,
        offset,
        count,
        timeout_sec,
        *,
        on_overflow: base.OnOverflowType = 'except',
    ) -> tuple[int, int]:
        assert self._rx_stream is not None
        # assert self._sync_time_source is not None

        return self._rx_stream.read(
            buffers,
            offset,
            count,
            timeout_sec,
            last_sync_time=self._sync_time_source.last_sync_time,
            on_overflow=on_overflow,
        )

    def get_temperatures(self) -> dict[str, float]:
        return {}

    @functools.cached_property
    def id(self):
        raise NotImplementedError

    @functools.cached_property
    def info(self) -> SoapyInfo:
        assert self._device is not None, 'soapy device is not open'

        return probe_soapy_info(self._device)

    @util.stopwatch('read_iq', 'source')
    def read_iq(self):
        assert self._rx_stream is not None, 'soapy device is not open'
        assert self._device is not None, 'soapy device is not open'

        if self._setup.time_sync_every_capture:
            self._rx_stream.enable(False)
            self._sync_time_source(self._device)

        if not self._rx_stream.is_enabled:
            self._rx_stream.enable(True)

        return super().read_iq()

    def acquire(self, capture=None, next_capture=None, correction=True):
        with read_retries(self):
            return super().acquire(capture, next_capture, correction)
