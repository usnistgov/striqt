"""a site-style sensor binding over the synthetic single-tone source

Mirrors the shape of the out-of-tree extension modules that sweep YAML files load
through their `extensions:` block: capture and peripheral spec subclasses, a source
with a fixed hardware id, and a y-factor calibration binding derived from the sensor.
"""

# ruff: noqa: UP007, UP045  msgspec resolves these annotations at runtime on py3.9
from __future__ import annotations

from math import nan
from typing import Annotated, Optional, Union

from striqt.sensor import bindings, peripherals, sources, specs

RADIO_ID = '48b02d17a587'

MastHeight = Annotated[float, specs.Meta('Mast height', 'm', ge=0)]
Elevation = Annotated[
    float, specs.Meta('Elevation pointing angle', 'deg', ge=-10, le=90)
]
Azimuth = Annotated[float, specs.Meta('Azimuth pointing angle', 'deg', ge=-180, le=180)]
SwitchInput = Annotated[int, specs.Meta('RF input switch index', ge=0, le=1)]
Label = Union[str, tuple[str, ...], None]


class SiteCapture(specs.SingleToneCapture, frozen=True, kw_only=True):
    center_frequency: specs.types.CenterFrequency
    gain: specs.types.Gain

    mast_height: Optional[MastHeight] = None
    elevation: Optional[Elevation] = None
    azimuth: Optional[Azimuth] = None
    switch_input: Optional[SwitchInput] = 0
    azimuth_offset: float = 0
    elevation_offset: float = 0

    radio_name: Label = ''
    site_name: Label = ''
    channel_name: Label = ''
    antenna_name: Label = ''
    antenna_model: Label = ''
    antenna_polarization: Label = ''
    antenna_index: Union[int, tuple[int, ...], None] = None

    def __post_init__(self):
        super().__post_init__()
        if (self.azimuth is None) != (self.elevation is None):
            raise ValueError('azimuth and elevation must be set together')


class SiteSurveyCapture(SiteCapture, frozen=True, kw_only=True):
    latitude: float = nan
    longitude: float = nan
    azimuth_repeat: int = 0
    tx1_power: float = nan
    test_case: int = 0


class SitePeripherals(specs.Peripherals, frozen=True, kw_only=True):
    control_orientation: bool = False
    connect_x410: bool = True


class SiteToneSource(sources.SingleToneSource):
    def get_id(self):
        return RADIO_ID


def _schema(capture_cls):
    return bindings.Schema(
        capture=capture_cls,
        arm_like=capture_cls,
        source=specs.FunctionSource,
        init_like=specs.FunctionSource,
        peripherals=SitePeripherals,
    )


_site_sensor = bindings.Sensor(
    source_cls=SiteToneSource, peripherals_cls=peripherals.NoPeripherals
)

site_single_tone = bindings.bind_sensor(
    'site_single_tone', _site_sensor, _schema(SiteCapture)
)

site_survey = bindings.bind_sensor(
    'site_survey', _site_sensor, _schema(SiteSurveyCapture)
)

site_single_tone_calibration = bindings.bind_manual_yfactor_calibration(
    'site_single_tone_calibration', site_single_tone
)
