from __future__ import annotations as __

import contextlib
import functools
import time
import typing

from ... import specs
from ...specs.structs import (
    _SoapyRange,
    _SoapyPortInfo,
    _SoapyArgInfo,
    _SoapySensorReading,
)

from .. import calibration, util
from . import _base

import striqt.analysis as sa
import striqt.waveform as sw

if typing.TYPE_CHECKING:
    import SoapySDR  # type: ignore
    import pandas as pd
    from striqt.waveform.util import ArrayType
else:
    try:
        SoapySDR = util.lazy_import('SoapySDR')
    except ImportError:
        SoapySDR = None

    pd = util.lazy_import('pandas')


_TS = typing.TypeVar('_TS', bound=specs.SoapySource)
_TC = typing.TypeVar('_TC', bound=specs.SoapyCapture)


def _get_adc_peak(
    x: 'ArrayType', capture: specs.SoapyCapture, source: specs.SoapySource
):
    xp = sw.util.array_namespace(x)

    adc_scale = _base._get_dtype_scale(source.transport_dtype)

    peak_counts = xp.abs(x).max(axis=-1)
    unscaled_peak = 20 * xp.log10(peak_counts * adc_scale) - 3
    return unscaled_peak


def _probe_channel(
    device: SoapySDR.Device, direction: int, port: int
) -> _SoapyPortInfo:
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
        gain_range = _SoapyRange(
            minimum=soapy_gain_range.minimum(),
            maximum=soapy_gain_range.maximum(),
            step=soapy_gain_range.step(),
        )
        gains[name] = gain_range

    freqs = {
        name: _SoapyRange.from_soapy_tuple(device.getFrequencyRange(*args, name))
        for name in device.listFrequencies(*args)
    }

    sensors = {
        key: _SoapySensorReading(
            name=device.getSensorInfo(*args, key).name,
            info=_SoapyArgInfo.from_soapy(device.getSensorInfo(*args, key)[0]),
            reading=device.readSensor(*args, key),
        )
        for key in device.listSensors(*args)
    }

    return _SoapyPortInfo(
        port_info=dict(device.getChannelInfo(*args)),
        full_duplex=device.getFullDuplex(*args),
        agc=device.hasGainMode(*args),
        stream_formats=tuple(device.getStreamFormats(*args)),
        # native_format=NativeFormat(format=native_fmt, full_scale=full_scale),
        stream_args=_SoapyArgInfo.from_soapy_map(device.getStreamArgsInfo(*args)),
        antennas=tuple(device.listAntennas(*args)),
        corrections=tuple(corrections_list),
        gains=gains,
        full_gain_range=_SoapyRange.from_soapy(device.getGainRange(*args)),
        frequencies=freqs,
        full_freq_range=_SoapyRange.from_soapy_tuple(device.getFrequencyRange(*args)),  # type: ignore
        tune_args=_SoapyArgInfo.from_soapy_map(device.getFrequencyArgsInfo(*args)),
        backend_sample_rate_range=_SoapyRange.from_soapy_tuple(
            device.getSampleRateRange(*args)
        ),  # type: ignore
        master_clock_rates=tuple(device.getMasterClockRates()),
        bandwidths=_SoapyRange.from_soapy_tuple(device.getBandwidthRange(*args)),
        sensors=sensors,
        settings=_SoapyArgInfo.from_soapy_map(device.getSettingInfo(*args)),
    )


def probe_soapy_info(device: SoapySDR.Device) -> specs.SoapySourceInfo:
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
        key: _SoapySensorReading(
            name=device.getSensorInfo(key).name,
            info=_SoapyArgInfo.from_soapy(device.getSensorInfo(key)),
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

    return specs.SoapySourceInfo(
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
        settings=_SoapyArgInfo.from_soapy_map(device.getSettingInfo()),
        gpios=tuple(device.listGPIOBanks()),
        uarts=tuple(device.listUARTs()),
        rx_ports=rx_channels,
        tx_ports=tx_channels,
    )


def compute_overload_info(
    samples: ArrayType, setup: specs.SoapySource, capture: specs.SoapyCapture
):
    adc_limit = setup.adc_overload_limit
    if_limit = setup.if_overload_limit
    if adc_limit is None and if_limit is None:
        return {}

    info = {}
    peak = _get_adc_peak(samples, capture, setup)
    xp = sw.util.array_namespace(peak)

    if adc_limit is not None:
        adc_headroom = xp.floor(adc_limit - peak)
        info['adc_headroom'] = xp.clip(adc_headroom, -100, 100).astype('int8')
    if if_limit is not None:
        xp = sw.util.array_namespace(samples)
        gains = [c.gain for c in specs.helpers.split_capture_ports(capture)]
        peak_im3 = peak + (2 / 3) * xp.array(gains)  # 2/3 arises from intermod
        if_headroom = xp.floor(if_limit - peak_im3)
        info['if_headroom'] = xp.clip(if_headroom, -100, 100).astype('int8')

    return info


@contextlib.contextmanager
def read_retries(source: SoapySourceBase) -> typing.Generator[None]:
    """in this context, retry source.read_iq on stream errors"""

    EXC_TYPES = (_base.ReceiveStreamError, OverflowError)

    max_count = source.setup_spec.receive_retries

    if max_count == 0:
        yield
        return

    def prepare_retry(*args, **kws):
        source._rx_stream.enable(source._device, False)
        source._buffers.clear()
        source._buffers.skip_next_buffer_swap()

    decorate = util.retry(EXC_TYPES, tries=max_count + 1, exception_func=prepare_retry)

    source.read_iq, original = decorate(source.read_iq), source.read_iq

    try:
        yield
    finally:
        source.read_iq = original


class RxStream:
    """manage the state of the RX stream"""

    ports: specs.types.Port

    def __init__(
        self,
        setup: specs.SoapySource,
        info: specs.SoapySourceInfo,
        *,
        ports: tuple[int, ...] = (),
        on_overflow: _base.OnOverflowType = 'except',
    ):
        self.setup = setup
        self.info = info

        self.checked_timestamp = False
        self.stream = None
        self._enabled: bool = False
        self._on_overflow: _base.OnOverflowType = on_overflow
        self.ports = ports

    @sa.util.stopwatch('stream initialization', 'source')
    def open(self, device: 'SoapySDR.Device', ports: specs.types.Port | None = None):
        if self.stream is not None and self.ports is not None:
            return

        if ports is not None:
            ports = specs.helpers.ensure_tuple(ports)
        elif self.setup.stream_all_rx_ports:
            ports = tuple(range(self.info.num_rx_ports))
        else:
            ports = self.ports

        assert ports not in (None, ()), 'failed to resolve port list for stream'

        self.ports = ports

        if self.setup.transport_dtype == 'int16':
            stype = SoapySDR.SOAPY_SDR_CS16
        elif self.setup.transport_dtype == 'float32':
            stype = SoapySDR.SOAPY_SDR_CF32
        else:
            raise ValueError(
                f'unsupported transport type {self.setup.transport_dtype!r}'
            )

        with self._minimized_rx_gain(device):
            self.stream = device.setupStream(SoapySDR.SOAPY_SDR_RX, stype, ports)

    @contextlib.contextmanager
    def _minimized_rx_gain(self, device: 'SoapySDR.Device'):
        """minimize the gain setting while in the context"""
        start_gains = {}

        for port, info in enumerate(self.info.rx_ports):
            start_gains[port] = device.getGain(SoapySDR.SOAPY_SDR_RX, port)
            min_gain = info.full_gain_range.minimum
            device.setGain(SoapySDR.SOAPY_SDR_RX, port, min_gain)

        yield

        # restore the initial gains
        for port, gain in start_gains.items():
            device.setGain(SoapySDR.SOAPY_SDR_RX, port, gain)

    def close(self, device: 'SoapySDR.Device'):
        if getattr(self, '_rx_stream', None) is None:
            return

        try:
            self.enable(device, False)
        except ValueError:
            # channel not yet set
            pass

        try:
            if device is not None:
                device.closeStream(self.stream)
        except ValueError as ex:
            if 'invalid parameter' in str(ex):
                # already closed
                pass
            else:
                raise

        self.stream = None
        self.ports = ()

    def enable(self, device: 'SoapySDR.Device', enable: bool):
        if enable == self._enabled:
            return

        if enable:
            if device.hasHardwareTime():
                kws = {'flags': SoapySDR.SOAPY_SDR_HAS_TIME}
            else:
                kws = {}

            if self.setup.rx_enable_delay is not None:
                delay_ns = round(self.setup.rx_enable_delay * 1e9)
                time_ns = device.getHardwareTime('now') + delay_ns
                kws['timeNs'] = time_ns

            self.checked_timestamp = False
            device.activateStream(self.stream, **kws)

        elif self.stream is not None:
            device.deactivateStream(self.stream)

        self._enabled = enable

    def read(
        self,
        device: 'SoapySDR.Device',
        buffers,
        offset,
        count,
        timeout_sec,
        *,
        last_sync_time: int | None,
        on_overflow: _base.OnOverflowType | None = None,
    ) -> tuple[int, int]:
        total_timeout = self.setup.rx_enable_delay + timeout_sec + 0.5

        offset_bufs = [buf[offset * 2 :] for buf in buffers]
        rx_result = device.readStream(
            self.stream,
            offset_bufs,
            count,
            timeoutUs=round(total_timeout * 1e6),
        )

        if not self.checked_timestamp and device.hasHardwareTime():
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
        on_overflow: _base.OnOverflowType,
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
            raise _base.ReceiveStreamError(f'{err_str} (error code {sr.ret})')

        if sync_time_ns is None or sr.timeNs == 0:
            pass
        elif sync_time_ns > sr.timeNs:
            raise _base.ReceiveStreamError(f'invalid timestamp from before last sync')

        return result

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def capture_changes_port(self, capture: specs.SoapyCapture) -> bool:
        return specs.helpers.ensure_tuple(capture.port) == self.ports

    def set_ports(self, device: 'SoapySDR.Device', ports: specs.types.Port):
        if self.setup.stream_all_rx_ports:
            assert self.stream is not None, (
                'expected open stream since stream_all_rx_ports=True'
            )

            # the stream is controlled on self.open(...)
            return

        elif getattr(self, 'stream', None) is not None:
            if self.ports == ports:
                # already set up
                return
            else:
                self.close(device)

        # if we make it this far, we need to build and enable the RX stream
        self.open(device, ports)


class HardwareTimeSync:
    def __init__(self, time_source: specs.types.TimeSource):
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
            raise IOError('this soapy device does not expose hardware time')

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
            sa.util.get_logger('source').warning(
                f'system time and PPS out of sync by {frac_secs:0.2f}s'
            )
        time_to_set_ns = int((full_secs + 1) * 1e9)
        device.setHardwareTime(time_to_set_ns, 'pps')
        return time_to_set_ns


def device_time_source(spec: specs.SoapySource):
    if spec.time_source == 'host':
        return 'internal'
    else:
        return spec.time_source


@_base.bind_schema_types(specs.SoapySource, specs.SoapyCapture)
class SoapySourceBase(_base.SourceBase[_TS, _TC, _base._PS, _base._PC]):
    """Applies SoapySDR for device control and acquisition"""

    _device: 'SoapySDR.Device'
    _rx_stream: RxStream

    def close(self):
        if not self.is_open():
            return

        if (
            SoapySDR is None
            or SoapySDR._SoapySDR is None
            or SoapySDR._SoapySDR.Device_deactivateStream is None
            or SoapySDR.Device is None
        ):
            # handle partially-deconstructed soapy module observed after
            # exceptions
            return

        self._device.__del__ = lambda: None
        try:
            self._rx_stream.close(self._device)
        finally:
            self._device.close()
        sa.util.get_logger('source').info('closed')
        super().close()

    @sa.util.stopwatch('open soapy radio', 'source', threshold=1)
    def _connect(self, spec: _TS, **device_kwargs):
        if SoapySDR is None:
            raise ImportError('could not import SoapySDR')

        # passing in a tuple works around an insantiation bug on some platforms
        # https://github.com/pothosware/SoapySDR/issues/472
        devices = SoapySDR.Device((device_kwargs,))

        if isinstance(devices, (list, tuple)) and len(devices) == 1:
            self._device = devices[0]
        else:
            raise RuntimeError('SoapySDR instantiated an unexpected type')

    @sa.util.stopwatch('setup radio', 'source', threshold=1)
    def _apply_setup(self, spec, *, captures=None, loops=None):
        for p in range(self.info.num_rx_ports):
            self._device.setGainMode(SoapySDR.SOAPY_SDR_RX, p, False)

        self._sync_time_source: HardwareTimeSync = HardwareTimeSync(spec.time_source)

        if spec.time_source == 'host':
            self._device.setTimeSource('internal')
            on_overflow = 'log'
            trigger_strobe = getattr(spec, 'trigger_strobe', None)
            if trigger_strobe is not None:
                sa.util.get_logger('source').warning(
                    'periodic trigger with host time will suffer from inaccuracy on overflow'
                )
        else:
            self._device.setTimeSource(spec.time_source)
            on_overflow = 'except'

        if captures is not None:
            ports = specs.helpers.get_unique_ports(captures, loops)
        else:
            ports = ()

        self._rx_stream = RxStream(
            spec, self.info, ports=ports, on_overflow=on_overflow
        )

        self._device.setClockSource(spec.clock_source)
        self._device.setMasterClockRate(spec.master_clock_rate)

        if spec.time_sync_at == 'open':
            self._sync_time_source(self._device)

        if len(ports) > 0:
            self._rx_stream.open(self._device)

    def _prepare_capture(self, capture: _TC) -> _TC | None:
        if (
            capture == self._capture and self.setup_spec.gapless
        ) or self._rx_stream is None:
            # the one case where we leave it running
            return

        self._rx_stream.enable(self._device, False)

        # manage changes to the ports
        ports_changed = self._rx_stream.capture_changes_port(capture)
        if self._capture is None or ports_changed:
            self._rx_stream.set_ports(self._device, capture.port)

        rs = self.get_resampler(capture)

        # gain before center frequency to accommodate attenuator settling time
        for c in specs.helpers.split_capture_ports(capture):
            assert not isinstance(c.center_frequency, tuple)
            freq = c.center_frequency - rs['lo_offset']
            self._device.setGain(SoapySDR.SOAPY_SDR_RX, c.port, c.gain)
            self._device.setFrequency(SoapySDR.SOAPY_SDR_RX, c.port, freq)
            self._device.setSampleRate(SoapySDR.SOAPY_SDR_RX, c.port, rs['fs_sdr'])

    def _read_stream(
        self,
        buffers,
        offset,
        count,
        timeout_sec,
        *,
        on_overflow: _base.OnOverflowType = 'except',
    ) -> tuple[int, int]:
        return self._rx_stream.read(
            self._device,
            buffers,
            offset,
            count,
            timeout_sec,
            last_sync_time=self._sync_time_source.last_sync_time,
            on_overflow=on_overflow,
        )

    def _package_acquisition(
        self,
        samples: ArrayType,
        time_ns: int | None,
        *,
        analysis=None,
        correction=True,
        alias_func: specs.helpers.PathAliasFormatter | None = None,
    ) -> _base.AcquiredIQ:
        iq = super()._package_acquisition(
            samples,
            time_ns,
            analysis=analysis,
            correction=correction,
            alias_func=alias_func,
        )

        capture = self.capture_spec

        # timestamps
        if time_ns is None:
            ts = None
        else:
            ts = pd.Timestamp(time_ns, unit='ns')

        if self._sweep_time is None:
            self._sweep_time = ts
            self._sweep_index = self.capture_spec._sweep_index
        elif self._sweep_index != self.capture_spec._sweep_index:
            self._sweep_time = ts
            self._sweep_index = self.capture_spec._sweep_index

        iq.info = specs.SoapyAcquisitionCoordinates(
            sweep_start_time=self._sweep_time,
            start_time=ts,
            backend_sample_rate=self.get_resampler()['fs_sdr'],
            **iq.info.to_dict(),
        )

        iq.extra_data.update(compute_overload_info(samples, self.setup_spec, capture))
        iq.extra_data.update(self.read_peripherals())

        # calibration data
        power_scale = calibration.lookup_power_correction(
            self.setup_spec.calibration,
            self.capture_spec,
            self.setup_spec.master_clock_rate,
            alias_func=alias_func,
            xp=sw.util.array_namespace(samples),
        )
        if power_scale is not None:
            iq.voltage_scale = iq.voltage_scale * (power_scale**0.5)

            iq.extra_data['system_noise'] = calibration.lookup_system_noise_power(
                self.setup_spec.calibration,
                self.capture_spec,
                self.setup_spec.master_clock_rate,
                alias_func=alias_func,
            )

        return iq

    def read_peripherals(self) -> dict[str, typing.Any]:
        return {}

    @functools.cached_property
    def id(self) -> str:
        if not self.is_open():
            raise ConnectionError('Device is closed')

        raise self._device.getHardwareKey()

    @functools.cached_property
    def info(self) -> specs.SoapySourceInfo:
        return probe_soapy_info(self._device)

    @sa.util.stopwatch('read_iq', 'source')
    def read_iq(self, analysis=None):
        assert self._rx_stream is not None, 'soapy device is not open'
        assert self._device is not None, 'soapy device is not open'

        if self.setup_spec.time_sync_at == 'acquire':
            self._rx_stream.enable(self._device, False)
            self._sync_time_source(self._device)

        if not self._rx_stream.is_enabled:
            self._rx_stream.enable(self._device, True)

        return super().read_iq(analysis)

    def acquire(self, *, analysis=None, correction=True, alias_func=None):
        with read_retries(self):
            return super().acquire(
                analysis=analysis, correction=correction, alias_func=alias_func
            )
