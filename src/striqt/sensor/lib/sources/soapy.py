from __future__ import annotations

import contextlib
import functools
import time
import numbers
import typing

from . import base
from .. import captures, specs, util
from striqt.analysis import registry

if typing.TYPE_CHECKING:
    import numpy as np
    import SoapySDR

else:
    np = util.lazy_import('numpy')
    SoapySDR = util.lazy_import('SoapySDR')


_TS = typing.TypeVar('_TS', bound=SoapyRadioSetup)
_TC = typing.TypeVar('_TC', bound=specs.RadioCapture)


ReceiveRetriesType = specs.Annotated[
    int,
    specs.meta(
        'number of attempts to retry acquisition on a stream error',
        ge=0,
    ),
]

TimeSyncEveryCaptureType = specs.Annotated[
    bool, specs.meta('whether to sync to PPS before each capture in a sweep')
]


TimeSourceType = specs.Annotated[
    specs.Literal['host', 'internal', 'external', 'gps'],
    specs.meta('Hardware source for timestamps'),
]

ClockSourceType = specs.Annotated[
    specs.Literal['internal', 'external', 'gps'],
    specs.meta('Hardware source for the frequency reference'),
]

ContinuousTriggerType = specs.Annotated[
    bool,
    specs.meta(
        'Whether to trigger immediately after each call to acquire() when armed'
    ),
]

OnOverflowType = (
    typing.Literal['ignore'] | typing.Literal['except'] | typing.Literal['log']
)


def _tuplize_if_number(obj: numbers.Number | tuple | None):
    if isinstance(obj, numbers.Number):
        return (obj,)
    else:
        return obj


class Range(typing.NamedTuple):
    """Represents a range with a minimum, maximum, and step."""

    minimum: float
    maximum: float
    step: float


class ArgInfo(typing.NamedTuple):
    """Represents information about a configurable device argument."""

    key: str
    name: str
    description: str
    units: str
    type: int
    value: str
    range: Range
    options: tuple[str, ...]


class SensorReading(typing.NamedTuple):
    """Represents a sensor and its current reading."""

    key: str
    name: str
    info: ArgInfo
    reading: str


class GainInfo(typing.NamedTuple):
    """Represents a specific gain element and its value range."""

    name: str
    range: Range


class FrequencyInfo(typing.NamedTuple):
    """Represents a specific frequency component and its value range."""

    name: str
    range: tuple[Range, ...]


# class NativeFormat(NamedTuple):
#     """Represents the native stream format and its full-scale value."""
#     format: str
#     full_scale: float


class PortInfo(typing.NamedTuple):
    """Holds all capability metadata for a single RX or TX channel."""

    port_info: dict[str, str]
    full_duplex: bool
    agc: bool
    stream_formats: tuple[str, ...]
    # native_format: NativeFormat
    stream_args: tuple[ArgInfo, ...]
    antennas: tuple[str, ...]
    corrections: tuple[str, ...]
    gains: tuple[GainInfo, ...]
    full_gain_range: Range
    frequencies: tuple[FrequencyInfo, ...]
    full_freq_range: tuple[Range, ...]
    tune_args: tuple[ArgInfo, ...]
    sample_rates: tuple[Range, ...]
    base_clock_rates: tuple[float, ...]
    bandwidths: tuple[Range, ...]
    sensors: tuple[SensorReading, ...]
    settings: tuple[ArgInfo, ...]


class SoapySourceInfo(typing.NamedTuple):
    """Top-level container for all device capabilities metadata."""

    driver: str
    hardware: str
    hardware_info: dict[str, str]
    num_rx_ports: int
    num_tx_ports: int
    has_timestamps: bool
    clock_sources: tuple[str, ...]
    time_sources: tuple[str, ...]
    global_sensors: tuple[SensorReading, ...]
    registers: tuple[str, ...]
    settings: tuple[ArgInfo, ...]
    gpios: tuple[str, ...]
    uarts: tuple[str, ...]
    rx_ports: tuple[PortInfo, ...]
    tx_ports: tuple[PortInfo, ...]


def _to_range_tuple(soapy_ranges: list[SoapySDR.Range]) -> tuple[Range, ...]:
    """Converts a list of SoapySDR.Range objects to a tuple of our Range named tuples."""
    return tuple(
        Range(minimum=r.minimum(), maximum=r.maximum(), step=r.step())
        for r in soapy_ranges
    )


def _to_arginfo_tuple(soapy_args: list[SoapySDR.ArgInfo]) -> tuple[ArgInfo, ...]:
    """Converts a list of SoapySDR.ArgInfo objects to a tuple of our ArgInfo named tuples."""
    results = []
    for arg in soapy_args:
        arg_range = Range(
            minimum=arg.range.minimum(),
            maximum=arg.range.maximum(),
            step=arg.range.step(),
        )
        results.append(
            ArgInfo(
                key=arg.key,
                name=arg.name,
                description=arg.description,
                units=arg.units,
                type=arg.type,
                value=arg.value,
                range=arg_range,
                options=tuple(arg.options),
            )
        )
    return tuple(results)


def _probe_channel(device: SoapySDR.Device, direction: int, port: int) -> PortInfo:
    """Probes a single channel and returns its capabilities."""
    corrections_list = []
    if device.hasDCOffsetMode(direction, port):
        corrections_list.append('DC removal')
    if device.hasDCOffset(direction, port):
        corrections_list.append('DC offset')
    if device.hasIQBalance(direction, port):
        corrections_list.append('IQ balance')

    gains_list = []
    for name in device.listGains(direction, port):
        soapy_gain_range = device.getGainRange(direction, port, name)
        gain_range = Range(
            minimum=soapy_gain_range.minimum(),
            maximum=soapy_gain_range.maximum(),
            step=soapy_gain_range.step(),
        )
        gains_list.append(GainInfo(name=name, range=gain_range))

    freqs_list = [
        FrequencyInfo(
            name=name,
            range=_to_range_tuple(device.getFrequencyRange(direction, port, name)),
        )
        for name in device.listFrequencies(direction, port)
    ]
    sensors_list = [
        SensorReading(
            key=key,
            name=device.getSensorInfo(direction, port, key).name,
            info=_to_arginfo_tuple([device.getSensorInfo(direction, port, key)])[0],
            reading=device.readSensor(direction, port, key),
        )
        for key in device.listSensors(direction, port)
    ]

    soapy_full_gain_range = device.getGainRange(direction, port)
    full_gain_range_tuple = Range(
        minimum=soapy_full_gain_range.minimum(),
        maximum=soapy_full_gain_range.maximum(),
        step=soapy_full_gain_range.step(),
    )

    # native_fmt, full_scale = device.getNativeStreamFormat(direction, channel)

    return PortInfo(
        port_info=dict(device.getChannelInfo(direction, port)),
        full_duplex=device.getFullDuplex(direction, port),
        agc=device.hasGainMode(direction, port),
        stream_formats=tuple(device.getStreamFormats(direction, port)),
        # native_format=NativeFormat(format=native_fmt, full_scale=full_scale),
        stream_args=_to_arginfo_tuple(device.getStreamArgsInfo(direction, port)),
        antennas=tuple(device.listAntennas(direction, port)),
        corrections=tuple(corrections_list),
        gains=tuple(gains_list),
        full_gain_range=full_gain_range_tuple,
        frequencies=tuple(freqs_list),
        full_freq_range=_to_range_tuple(device.getFrequencyRange(direction, port)),
        tune_args=_to_arginfo_tuple(device.getFrequencyArgsInfo(direction, port)),
        sample_rates=_to_range_tuple(device.getSampleRates(direction, port)),
        base_clock_rates=tuple(device.getMasterClockRates()),
        bandwidths=_to_range_tuple(device.getBandwidthRange(direction, port)),
        sensors=tuple(sensors_list),
        settings=_to_arginfo_tuple(device.getSettingInfo(direction, port)),
    )


def probe_soapy_info(device: SoapySDR.Device) -> SoapySourceInfo:
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
    global_sensors = [
        SensorReading(
            key=key,
            name=device.getSensorInfo(key).name,
            info=_to_arginfo_tuple([device.getSensorInfo(key)])[0],
            reading=device.readSensor(key),
        )
        for key in device.listSensors()
    ]

    num_rx = device.getNumChannels(SoapySDR.SOAPY_SDR_RX)
    num_tx = device.getNumChannels(SoapySDR.SOAPY_SDR_TX)

    rx_channels = tuple(
        _probe_channel(device, SoapySDR.SOAPY_SDR_RX, i) for i in range(num_rx)
    )
    tx_channels = tuple(
        _probe_channel(device, SoapySDR.SOAPY_SDR_TX, i) for i in range(num_tx)
    )

    return SoapySourceInfo(
        driver=device.getDriverKey(),
        hardware=device.getHardwareKey(),
        hardware_info=device.getHardwareInfo(),
        num_rx_ports=num_rx,
        num_tx_ports=num_tx,
        has_timestamps=device.hasHardwareTime(),
        clock_sources=tuple(device.listClockSources()),
        time_sources=tuple(device.listTimeSources()),
        global_sensors=tuple(global_sensors),
        registers=tuple(device.listRegisterInterfaces()),
        settings=_to_arginfo_tuple(device.getSettingInfo()),
        gpios=tuple(device.listGPIOBanks()),
        uarts=tuple(device.listUARTs()),
        rx_ports=rx_channels,
        tx_ports=tx_channels,
    )


class SoapyRadioSetup(specs.RadioSetup):
    transport_dtype: typing.Literal['int16'] | typing.Literal['float32'] = 'int16'
    time_source: TimeSourceType = 'host'
    time_sync_every_capture: TimeSyncEveryCaptureType = False
    clock_source: ClockSourceType = 'internal'
    receive_retries: ReceiveRetriesType = 0

    # True if the same clock drives acquisition on all RX ports
    shared_rx_sample_clock = True
    rx_enable_delay = 0.0

    def __post_init__(self):
        if not self.gapless_repeats:
            pass
        elif self.time_sync_every_capture:
            raise ValueError(
                'time_sync_every_capture and gapless_repeats are mutually exclusive'
            )
        elif self.receive_retries > 0:
            raise ValueError(
                'receive_retries must be 0 when gapless_repeats is enabled'
            )

        if self.channel_sync_source is None:
            pass
        elif self.channel_sync_source not in registry.channel_sync_source:
            registered = set(registry.channel_sync_source)
            raise ValueError(
                f'channel_sync_source "{self.channel_sync_source!r}" is not one of the registered functions {registered!r}'
            )


class _SoapyRadioSetupKeywords(specs._RadioSetupKeywords, total=False):
    receive_retries: typing.NotRequired[ReceiveRetriesType]
    time_source: typing.NotRequired[TimeSourceType]
    clock_source: typing.NotRequired[ClockSourceType]
    time_sync_every_capture: typing.NotRequired[TimeSyncEveryCaptureType]


@contextlib.contextmanager
def read_retries(radio: SoapySourceBase) -> typing.Generator[None]:
    """in this context, retry radio.read_iq on stream errors"""

    if radio._setup.receive_retries == 0:
        yield
        return

    def prepare_retry(*args, **kws):
        radio._rx_stream.enable(False)
        radio._buffers.hold_buffer_swap = True
        if not radio._setup.time_sync_every_capture:
            radio._sync_time_source()

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

    def __init__(self, radio: SoapySourceBase, on_overflow: OnOverflowType = 'except'):
        self.radio = radio
        self.checked_timestamp = False
        self.stream = None
        self._enabled: bool = False
        self._on_overflow: OnOverflowType = on_overflow
        self._ports: specs.PortType = ()

    @util.stopwatch('stream initialization', 'source')
    def open(self, ports=None):
        if self.stream is not None and self._ports is not None:
            return

        if isinstance(ports, numbers.Number):
            ports = (ports,)

        if ports is not None:
            ports = _tuplize_if_number(ports)
        elif self.radio._setup.stream_all_channels:
            ports = tuple(range(self.radio.source_info.num_rx_ports))
        else:
            ports = self._ports
        self._ports = ports

        if self.radio._setup.transport_dtype == 'int16':
            soapy_type = SoapySDR.SOAPY_SDR_CS16
        elif self.radio._setup.transport_dtype == 'float32':
            soapy_type = SoapySDR.SOAPY_SDR_CF32
        else:
            raise ValueError(f'unsupported transport type {self._transport_type}')

        with minimized_rx_gain(self.radio):
            self.stream = self.radio._device.setupStream(
                SoapySDR.SOAPY_SDR_RX, soapy_type, list(ports)
            )

    def close(self):
        if getattr(self, 'stream', None) is None:
            return

        try:
            self.enable(False)
        except ValueError:
            # channel not yet set
            pass

        try:
            self.radio._device.closeStream(self.stream)
        except ValueError as ex:
            if 'invalid parameter' in str(ex):
                # already closed
                pass
            else:
                raise

        self.stream = None
        self._ports = None

    def enable(self, enable: bool):
        if enable == self._enabled:
            return

        if enable:
            if self.backend.hasHardwareTime():
                kws = {'flags': SoapySDR.SOAPY_SDR_HAS_TIME}
            else:
                kws = {}

            if self.radio._setup.rx_enable_delay is not None:
                delay_ns = round(self._enable_delay * 1e9)
                time_ns = self.backend.getHardwareTime('now') + delay_ns
                kws['timeNs'] = time_ns

            self.checked_timestamp = False
            self.backend.activateStream(self.stream, **kws)

        elif self.stream is not None:
            self.backend.deactivateStream(self.stream)

        self._enabled = enable

    def read(
        self,
        buffers,
        offset,
        count,
        timeout_sec,
        *,
        on_overflow: OnOverflowType | None = None,
    ) -> tuple[int, int]:
        total_timeout = self._enable_delay + timeout_sec + 0.5

        rx_result = self.radio._device.readStream(
            self.stream,
            [buf[offset * 2 :] for buf in buffers],
            count,
            timeoutUs=round(total_timeout * 1e6),
        )

        if not self.checked_timestamp and self.radio._device.hasHardwareTime():
            # require a valid timestamp only for the first read after channel enables.
            # subsequent reads are treated as contiguous unless TimeoutError is raised
            sync_time_ns = self.radio._sync_time_source.last_sync_time
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
        sr: SoapySDR.StreamResult, on_overflow: OnOverflowType, sync_time_ns: int = None
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

    def capture_changes_port(self, capture: specs.RadioCapture) -> bool:
        return _tuplize_if_number(capture.port) == self._ports

    def set_ports(self, ports: specs.PortType):
        if self._setup.stream_all_rx_ports:
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


@contextlib.contextmanager
def minimized_rx_gain(radio: SoapySourceBase):
    """minimize the gain setting while in the context"""
    start_gains = {}

    for port, info in enumerate(radio.source_info.rx_ports):
        start_gains[port] = radio._device.getGain(SoapySDR.SOAPY_SDR_RX, port)
        min_gain = info.full_gain_range.minimum
        radio._device.setGain(SoapySDR.SOAPY_SDR_RX, port, min_gain)

    yield

    # restore the initial gains
    for port, gain in start_gains.items():
        radio._device.setGain(SoapySDR.SOAPY_SDR_RX, port, gain)


class HardwareTimeSync:
    def __init__(self, source: SoapySourceBase):
        self.last_sync_time: int | None = None
        self.source = source

    def __call__(self):
        time_source = self.source._setup.time_source
        if time_source in ('internal', 'host'):
            self.last_sync_time = self.to_host_os(self.source)
        elif time_source in ('gps', 'external'):
            self.last_sync_time = self.to_external_pps(self.source)
        else:
            pass

    def to_host_os(self, source: SoapySourceBase) -> int:
        if not self.source._device.hasHardwareTime():
            raise IOError('device does not expose hardware time')

        hardware_time = source._device.getHardwareTime('now') / 1e9
        if abs(hardware_time - time.time()) >= 0.2:
            sync_time = round(time.time() * 1e9)
            source._device.setHardwareTime(sync_time, 'now')
            return sync_time
        else:
            return self.last_sync_time

    def to_external_pps(self, source: SoapySourceBase) -> int:
        if not self.source._device.hasHardwareTime():
            raise IOError('device does not expose hardware time')

        # let a PPS transition pass to avoid race conditions involving
        # applying the time of the next PPS
        init_pps_time = source._device.getHardwareTime('pps')
        start_time = time.perf_counter()
        while init_pps_time == source._device.getHardwareTime('pps'):
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
            self.source._logger.warning(
                f'system time and PPS out of sync by {frac_secs:0.3f}s, check NTP'
            )
        time_to_set_ns = int((full_secs + 1) * 1e9)
        source._device.setHardwareTime(time_to_set_ns, 'pps')
        return time_to_set_ns


class SoapySourceBase(base.SourceBase[_TS, _TC]):
    """Applies SoapySDR for device control and acquisition"""

    _device: 'SoapySDR.Device' | None = None
    _rx_stream: RxStream | None = None
    _setup_cls: _TS = SoapyRadioSetup
    _capture_cls: _TC = specs.RadioCapture

    def __init__(self, device_args: dict = {}):
        self._device_kwargs: dict = device_args

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

        if getattr(self, '_rx_stream', None) is not None:
            self._rx_stream.close()

        self._device.close()
        self._logger.info('closed')
        super().close()

    @util.stopwatch('connected to radio', 'source', util.PERFORMANCE_INFO)
    def _connect(self, spec: _TS):
        if self._device_kwargs:
            # prevent race conditions in threaded accesses to the Soapy driver
            self._device = SoapySDR.Device(self._device_kwargs)
        else:
            self._device = SoapySDR.Device()

        super()._connect(spec)

    @util.stopwatch('configured soapy radio streaming', 'source', util.PERFORMANCE_INFO)
    def _apply_setup(self, spec: _TS) -> _TS | None:
        for p in range(self.source_info.num_rx_ports):
            self._device.setGainMode(SoapySDR.SOAPY_SDR_RX, p, False)

        self._sync_time_source: HardwareTimeSync = HardwareTimeSync(self)
        self._rx_stream = RxStream(self)

        if spec.time_source == 'host':
            self._device.setTimeSource('internal')
            on_overflow = 'log'
            periodic_trigger = getattr(spec, 'periodic_trigger', None)
            if periodic_trigger is not None:
                self._logger.warning(
                    'periodic trigger with host time will suffer from inaccuracy on overflow'
                )
        else:
            self._device.setTimeSource(spec.time_source)
            on_overflow = 'except'

        self._device.setClockSource(spec.clock_source)
        self._device.setMasterClockRate(spec.base_clock_rate)

        if not spec.time_sync_every_capture:
            self._sync_time_source()

        self._rx_stream.open(on_overflow=on_overflow)

    def _prepare_capture(self, capture: specs.RadioCapture) -> specs.RadioCapture:
        if capture == self._capture and self._setup.gapless_repeats:
            # the one case where we leave it running
            return super()._prepare_capture(capture)

        self._rx_stream.enable(False)

        # a list with 1 capture per port
        split_captures = captures.split_capture_ports(capture)

        prev_capture = self._capture
        # self._forced_backend_sample_rate = capture.backend_sample_rate

        # any changes to the port index
        if prev_capture is None or self._rx_stream.capture_changes_port(capture):
            self._rx_stream.set_ports(capture.port)
            prev_splits = len(split_captures) * [None]
        else:
            prev_splits = captures.split_capture_ports(prev_capture)

        # gain before center frequency to accommodate attenuator settling time
        for prev, this in zip(prev_splits, split_captures):
            if prev is None or this.gain != this.gain:
                self._device.setGain(SoapySDR.SOAPY_SDR_RX, this.port, this.gain)

        # work the backend sample rate, accounting for lo_shift
        for prev, this in zip(prev_splits, split_captures):
            backend_fc = this.center_frequency - this.lo_offset

            if prev is None:
                prev_backend_fc = None
            else:
                prev_backend_fc = prev.center_frequency - prev.lo_offset

            if backend_fc != prev_backend_fc:
                self._device.setFrequency(SoapySDR.SOAPY_SDR_RX, this.port, backend_fc)

        # TODO: support different sample rates per port, accommodating different downsampling rates
        if (
            prev_capture is None
            or capture.backend_sample_rate != prev_capture.backend_sample_rate
        ):
            if self._setup.shared_rx_sample_clock:
                sample_rate_list = split_captures[:1]
            else:
                sample_rate_list = split_captures

            for c in sample_rate_list:
                self._device.setSampleRate(
                    SoapySDR.SOAPY_SDR_RX, c.port, c.backend_sample_rate
                )

        if (
            self._setup.periodic_trigger is not None
            and capture.duration < self._setup.periodic_trigger
        ):
            self._logger.warning(
                'periodic trigger duration exceeds capture duration, '
                'which creates a large buffer of unused samples'
            )

        return super()._prepare_capture(capture)

    def _read_stream(
        self, buffers, offset, count, timeout_sec, *, on_overflow='except'
    ) -> tuple[int, int]:
        return self._rx_stream.read(
            buffers, offset, count, timeout_sec, on_overflow=on_overflow
        )

    def get_temperatures(self) -> dict[str, float]:
        return {}

    @functools.cached_property
    def id(self):
        raise NotImplementedError

    @functools.cached_property
    def source_info(self) -> SoapySourceInfo:
        return probe_soapy_info(self._device)

    @util.stopwatch('read_iq', 'source')
    def read_iq(
        self,
        capture: specs.RadioCapture,
    ) -> tuple['np.ndarray[np.complex64]', int]:
        if self._setup.time_sync_every_capture:
            self._rx_stream.enable(False)
            self._sync_time_source()

        if not self._rx_stream.is_enabled:
            self._rx_stream.enable(True)

        return super().read_iq(capture)

    def acquire(self, capture=None, next_capture=None, correction=True):
        with read_retries(self):
            return super().acquire(capture, next_capture, correction)
