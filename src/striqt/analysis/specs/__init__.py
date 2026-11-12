# `structs` has no `__all__`, so the star import also binds its `helpers` and `types`
# imports. ty resolves the explicit module imports below only after the star import.
from .structs import *

# isort: split
from . import doc, helpers, types
from .helpers import frozendict
from .types import Meta

for obj in dict(locals()).values():
    if getattr(obj, '__module__', '').startswith(__name__):
        obj.__module__ = __name__
del obj
