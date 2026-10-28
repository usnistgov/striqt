"""factories for SoapySDR-backed spec instances and the fake device, shared by the
sensor tests."""

from __future__ import annotations

import functools

import msgspec

import striqt.sensor as ss

MCR = 125e6

CAPTURE_DEFAULTS = {
    'port': 0,
    'center_frequency': 1e9,
    'gain': 0,
    'sample_rate': MCR,
    'duration': 1e-3,
    'analysis_bandwidth': 40e6,
    'host_resample': False,
}

# class attributes that SoapySource reads but does not expose as fields, so a preset
# with a different value has to be a subclass
_CLASS_ATTRS = (
    'rx_enable_delay',
    'stream_all_rx_ports',
    'transport_dtype',
    'shared_rx_sample_clock',
    'transient_holdoff_time',
)


def soapy_capture(**kws) -> ss.specs.SoapyCapture:
    """return a `SoapyCapture` at the master clock rate of the fake device, with `kws`
    overriding the defaults"""
    return ss.specs.SoapyCapture(**{**CAPTURE_DEFAULTS, **kws})


@functools.cache
def _source_preset_cls(class_attrs: tuple):
    return msgspec.defstruct(
        'SoapySourcePreset',
        [],
        bases=(ss.specs.SoapySource,),
        namespace=dict(class_attrs),
        kw_only=True,
        frozen=True,
    )


def source_spec(**overrides) -> ss.specs.SoapySource:
    """return a `SoapySource` at the master clock rate of the fake device.

    Field overrides are passed to the constructor. Overrides of the class attributes
    `rx_enable_delay`, `stream_all_rx_ports`, `transport_dtype`,
    `shared_rx_sample_clock`, and `transient_holdoff_time` select a cached
    `SoapySource` subclass that defines those attributes.
    """
    class_attrs = tuple(
        sorted((k, overrides.pop(k)) for k in list(overrides) if k in _CLASS_ATTRS)
    )
    cls = _source_preset_cls(class_attrs) if class_attrs else ss.specs.SoapySource
    return cls(**{'master_clock_rate': MCR, **overrides})


def call_names(device, *names) -> list[str]:
    """return the names of the recorded calls of the fake `device` that match `names`,
    in call order"""
    return [c[0] for c in device.calls_named(*names)]
