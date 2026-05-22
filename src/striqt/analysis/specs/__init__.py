from .structs import *
from . import types, helpers
from .types import Meta
from .helpers import frozendict

for obj in list(locals().values()):
    if getattr(obj, '__module__', '').startswith(__name__):
        obj.__module__ = __name__
del obj
