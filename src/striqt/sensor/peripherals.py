from .lib.peripherals import NoPeripherals, PeripheralsBase

for obj in list(locals().values()):
    if getattr(obj, '__module__', '').startswith(__name__):
        obj.__module__ = __name__

del obj
