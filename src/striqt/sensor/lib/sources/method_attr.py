"""implement method attributes that validate hardware gain and channel settings
as defined in structs.RadioCapture"""

from __future__ import annotations
import numbers
import typing

import labbench as lb

from .. import util



if typing.TYPE_CHECKING:
    ElementType = typing.TypeVar('ElementType')


@typing.overload
def _number_if_single(seq: ElementType) -> ElementType:
    pass


@typing.overload
def _number_if_single(seq: tuple[ElementType]) -> ElementType:
    pass


@typing.overload
def _number_if_single(seq: tuple[ElementType, ...]) -> tuple[ElementType]:
    pass


@util.lru_cache()
def _number_if_single(seq: ElementType | tuple[ElementType, ...]):
    if isinstance(seq, numbers.Number):
        return seq
    elif len(seq) == 1:
        return seq[0]
    else:
        return seq


@util.lru_cache()
def _validate_tuple_numbers(
    type_, values: numbers.Number | tuple, min, max, step, allow_duplicates=True
):
    """return a sorted unique tuple of 0-indexed channel ports, or raise a ValueError"""

    if isinstance(values, (bytes, str, bool, numbers.Number)):
        return type_(values)

    ret = []

    for value in values:
        if not isinstance(value, (bytes, str, bool, numbers.Number)):
            raise ValueError(
                f"a '{ChannelMaybeTupleMethod.__qualname__}' attribute supports only numerical, str, or bytes types"
            )

        if max is not None and value > max:
            raise ValueError(f'{value} is greater than the max limit {max}')

        if min is not None and value < min:
            raise ValueError(f'{value} is less than the min limit {min}')

        if step is not None:
            value = value - (value % step)

        ret.append(type_(value))

    if not allow_duplicates and len(frozenset(ret)) != len(ret):
        raise ValueError('duplicate values are not allowed')

    return _number_if_single(tuple(ret))


class BoundedNumberMaybeTupleMethod(
    lb.paramattr.method.Method, lb.paramattr._types.Tuple
):
    contained_type: ElementType = object
    sets: bool = True
    min: ElementType = None
    max: ElementType = None
    step: ElementType = None
    allow_duplicates: bool = True

    def validate(self, obj: ElementType | tuple[ElementType, ...], owner=None):
        if hasattr(obj, '__len__'):
            obj = tuple(obj)

        return _validate_tuple_numbers(
            self.contained_type,
            obj,
            self.min,
            self.max,
            self.step,
            self.allow_duplicates,
        )

    def to_pythonic(self, values: tuple[int, ...]):
        return self.validate(values)


class FloatMaybeTupleMethod(BoundedNumberMaybeTupleMethod[tuple[float, ...]]):
    contained_type: ElementType = float


class IntMaybeTupleMethod(BoundedNumberMaybeTupleMethod[tuple[int, ...]]):
    contained_type: ElementType = int


class ChannelMaybeTupleMethod(IntMaybeTupleMethod):
    min: int = 0
    allow_duplicates: bool = False

    def validate(self, obj: int | tuple[int, ...], owner=None):
        if self.max is None and owner is not None:
            max_ = owner.rx_channel_count - 1
        else:
            max_ = None

        if hasattr(obj, '__len__'):
            obj = tuple(obj)

        return _validate_tuple_numbers(
            self.contained_type, obj, self.min, max_, self.step, self.allow_duplicates
        )
