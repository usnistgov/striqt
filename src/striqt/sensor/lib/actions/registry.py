"""registry of the action runner for each action spec type.

Bound classes subclass the action spec type they are built from. A runner
registered for the base type therefore also runs the bindings derived from it.
"""

from __future__ import annotations as __

from typing import TYPE_CHECKING, Any, Callable, cast

from ... import specs
from ..typing import TR

if TYPE_CHECKING:
    from .core import ActionRunner


class ActionRegistry(dict['type[specs.ActionSpec]', 'type[ActionRunner]']):
    """map from an action spec type to the `ActionRunner` subclass that runs it."""

    def runner(
        self, spec_type: type[specs.ActionSpec[Any, Any]]
    ) -> Callable[[type[TR]], type[TR]]:
        """register the decorated `ActionRunner` subclass as the runner of `spec_type`.

        The decorator returns the class unchanged.

        Raises:
            TypeError: if a runner is already registered for `spec_type`
        """

        def wrapper(cls: type[TR]) -> type[TR]:
            if spec_type in self:
                raise TypeError(
                    f'an action runner is already registered for {spec_type!r}'
                )
            self[spec_type] = cls
            return cls

        return wrapper

    def lookup(
        self, spec: specs.ActionSpec | type[specs.ActionSpec]
    ) -> type[ActionRunner[Any, Any, Any, Any, Any, Any]]:
        """return the runner registered for the spec type or the nearest of its bases.

        Args:
            spec: a sweep spec instance, or its type

        Raises:
            TypeError: if no class in the MRO of the spec type is registered
        """
        spec_type = spec if isinstance(spec, type) else type(spec)
        for cls in spec_type.__mro__:
            if cls in self:
                return self[cast('type[specs.ActionSpec]', cls)]
        raise TypeError(f'no action runner is registered for {spec_type!r}')


registry = ActionRegistry()
