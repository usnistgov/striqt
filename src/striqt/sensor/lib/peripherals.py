from __future__ import annotations
from . import specs


class PeripheralsBase:
    """base class defining the object protocol peripheral hardware support.

    This is implemented primarily through connection management and callback
    methods for arming and acquisition.
    """

    def __init__(self, sweep: specs.Sweep | None):
        self.set_sweep(sweep)
        super().__init__()

    def open(self):
        pass

    def close(self):
        pass

    def arm(self, capture: specs.RadioCapture) -> dict[str]:
        """called while the capture is being armed in the radio.

        This then returns a dictionary of {field_name: value} pairs to update in `capture`.
        """
        return {}

    def acquire(self, capture: specs.RadioCapture) -> dict[str]:
        """called while the capture is being acquired in the radio.

        This returns a dictionary of new {data_variable: value} pairs that specify that
        a data variable named `data_variable` should be added to the saved dataset. Value
        can be a scalar or an xarray DataArray.
        """
        return {}

    def set_sweep(self, sweep: specs.Sweep):
        self.sweep = sweep

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()


class NoPeripherals(PeripheralsBase):
    def __init__(self, sweep=None):
        pass
