from typing import Annotated, Union
import msgspec as _msgspec


def Meta(standard_name: str, units: str | None = None, **kws) -> _msgspec.Meta:
    """annotation that is used to generate 'standard_name' and 'units' fields of xarray attrs objects"""
    extra = {'standard_name': standard_name}
    if units is not None:
        # xarray objects with units == None cannot be saved to netcdf;
        # in this case, omit
        extra['units'] = units
    return _msgspec.Meta(description=standard_name, extra=extra, **kws)


DurationType = Annotated[float, Meta('Duration of the analysis waveform', 's')]
SampleRateType = Annotated[float, Meta('Analysis sample rate', 'S/s')]
AnalysisBandwidthType = Annotated[float, Meta('Analysis bandwidth', 'Hz')]
WindowType = Union[str, tuple[str, float]]