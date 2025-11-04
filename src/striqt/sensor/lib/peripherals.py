from __future__ import annotations
from . import specs, sources
import typing


class PeripheralsBase(typing.Generic[specs._TSW, specs._TC]):
    """base class defining the object protocol peripheral hardware support.

    This is implemented primarily through connection management and callback
    methods for arming and acquisition.
    """

    sweep: specs._TSW

    def __init__(self, sweep: specs._TSW, source: sources.SourceBase | None = None):
        self.set_sweep(sweep)
        self.source = source

    def open(self):
        pass

    def close(self):
        pass

    def setup(self) -> None:
        pass

    def arm(self, capture: specs._TC):
        """called while the capture is being armed in the radio.

        This then returns a dictionary of {field_name: value} pairs to update in `capture`.
        """
        return None

    def acquire(self, capture: specs._TC) -> dict[str, typing.Any]:
        """called while the capture is being acquired in the radio.

        This returns a dictionary of new {data_variable: value} pairs that specify that
        a data variable named `data_variable` should be added to the saved dataset. Value
        can be a scalar or an xarray DataArray.
        """
        return {}

    def set_sweep(self, sweep: specs._TSW):
        self.sweep = sweep

    def set_source(self, source: sources.SourceBase | None = None):
        self.source = source

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()


class NoPeripherals(PeripheralsBase):
    def __init__(self, sweep=None, source=None):
        pass
