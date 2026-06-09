from .lib.peripherals import PeripheralsBase, NoPeripherals


for obj in list(locals().values()):
    if getattr(obj, '__module__', '').startswith(__name__):
        obj.__module__ = __name__

del obj  # pyright: ignore
