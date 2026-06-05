from .structs import *
from . import doc, types, helpers
from .types import Meta
from .helpers import frozendict

for name, obj in dict(locals()).items():
    if getattr(obj, '__module__', '').startswith(__name__):
        obj.__module__ = __name__
del obj  # pyright: ignore
