import SoapySDR
from typing import NamedTuple, List, Tuple, Dict

#
# 1. Define the nested NamedTuple structures for the metadata
# ----------------------------------------------------------------


class Range(NamedTuple):
    """Represents a range with a minimum, maximum, and step."""

    minimum: float
    maximum: float
    step: float


class ArgInfo(NamedTuple):
    """Represents information about a configurable device argument."""

    key: str
    name: str
    description: str
    units: str
    type: int
    value: str
    range: Range
    options: Tuple[str, ...]


class SensorReading(NamedTuple):
    """Represents a sensor and its current reading."""

    key: str
    name: str
    info: ArgInfo
    reading: str


class GainInfo(NamedTuple):
    """Represents a specific gain element and its value range."""

    name: str
    range: Range


class FrequencyInfo(NamedTuple):
    """Represents a specific frequency component and its value range."""

    name: str
    range: Tuple[Range, ...]


# class NativeFormat(NamedTuple):
#     """Represents the native stream format and its full-scale value."""
#     format: str
#     full_scale: float


class PortInfo(NamedTuple):
    """Holds all capability metadata for a single RX or TX channel."""

    port_info: Dict[str, str]
    full_duplex: bool
    agc: bool
    stream_formats: Tuple[str, ...]
    # native_format: NativeFormat
    stream_args: Tuple[ArgInfo, ...]
    antennas: Tuple[str, ...]
    corrections: Tuple[str, ...]
    gains: Tuple[GainInfo, ...]
    full_gain_range: Range
    frequencies: Tuple[FrequencyInfo, ...]
    full_freq_range: Tuple[Range, ...]
    tune_args: Tuple[ArgInfo, ...]
    sample_rates: Tuple[Range, ...]
    bandwidths: Tuple[Range, ...]
    sensors: Tuple[SensorReading, ...]
    settings: Tuple[ArgInfo, ...]


class RadioInfo(NamedTuple):
    """Top-level container for all device capabilities metadata."""

    driver: str
    hardware: str
    hardware_info: Dict[str, str]
    num_rx_ports: int
    num_tx_ports: int
    has_timestamps: bool
    clock_sources: Tuple[str, ...]
    time_sources: Tuple[str, ...]
    global_sensors: Tuple[SensorReading, ...]
    registers: Tuple[str, ...]
    settings: Tuple[ArgInfo, ...]
    gpios: Tuple[str, ...]
    uarts: Tuple[str, ...]
    rx_ports: Tuple[PortInfo, ...]
    tx_ports: Tuple[PortInfo, ...]


def _to_range_tuple(soapy_ranges: List[SoapySDR.Range]) -> Tuple[Range, ...]:
    """Converts a list of SoapySDR.Range objects to a tuple of our Range named tuples."""
    return tuple(
        Range(minimum=r.minimum(), maximum=r.maximum(), step=r.step())
        for r in soapy_ranges
    )


def _to_arginfo_tuple(soapy_args: List[SoapySDR.ArgInfo]) -> Tuple[ArgInfo, ...]:
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


def _probe_channel(device: SoapySDR.Device, direction: int, channel: int) -> PortInfo:
    """Probes a single channel and returns its capabilities."""
    corrections_list = []
    if device.hasDCOffsetMode(direction, channel):
        corrections_list.append('DC removal')
    if device.hasDCOffset(direction, channel):
        corrections_list.append('DC offset')
    if device.hasIQBalance(direction, channel):
        corrections_list.append('IQ balance')

    gains_list = []
    for name in device.listGains(direction, channel):
        soapy_gain_range = device.getGainRange(direction, channel, name)
        gain_range = Range(
            minimum=soapy_gain_range.minimum(),
            maximum=soapy_gain_range.maximum(),
            step=soapy_gain_range.step(),
        )
        gains_list.append(GainInfo(name=name, range=gain_range))

    freqs_list = [
        FrequencyInfo(
            name=name,
            range=_to_range_tuple(device.getFrequencyRange(direction, channel, name)),
        )
        for name in device.listFrequencies(direction, channel)
    ]
    sensors_list = [
        SensorReading(
            key=key,
            name=device.getSensorInfo(direction, channel, key).name,
            info=_to_arginfo_tuple([device.getSensorInfo(direction, channel, key)])[0],
            reading=device.readSensor(direction, channel, key),
        )
        for key in device.listSensors(direction, channel)
    ]

    soapy_full_gain_range = device.getGainRange(direction, channel)
    full_gain_range_tuple = Range(
        minimum=soapy_full_gain_range.minimum(),
        maximum=soapy_full_gain_range.maximum(),
        step=soapy_full_gain_range.step(),
    )

    # native_fmt, full_scale = device.getNativeStreamFormat(direction, channel)

    return PortInfo(
        port_info=dict(device.getChannelInfo(direction, channel)),
        full_duplex=device.getFullDuplex(direction, channel),
        agc=device.hasGainMode(direction, channel),
        stream_formats=tuple(device.getStreamFormats(direction, channel)),
        # native_format=NativeFormat(format=native_fmt, full_scale=full_scale),
        stream_args=_to_arginfo_tuple(device.getStreamArgsInfo(direction, channel)),
        antennas=tuple(device.listAntennas(direction, channel)),
        corrections=tuple(corrections_list),
        gains=tuple(gains_list),
        full_gain_range=full_gain_range_tuple,
        frequencies=tuple(freqs_list),
        full_freq_range=_to_range_tuple(device.getFrequencyRange(direction, channel)),
        tune_args=_to_arginfo_tuple(device.getFrequencyArgsInfo(direction, channel)),
        sample_rates=_to_range_tuple(device.getSampleRateRange(direction, channel)),
        bandwidths=_to_range_tuple(device.getBandwidthRange(direction, channel)),
        sensors=tuple(sensors_list),
        settings=_to_arginfo_tuple(device.getSettingInfo(direction, channel)),
    )


def probe_radio_capabilities(device: SoapySDR.Device) -> RadioInfo:
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

    return RadioInfo(
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
