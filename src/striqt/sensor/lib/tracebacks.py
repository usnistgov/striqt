import typing

from IPython.core import ultratb

MATCH = ['site-packages/click', 'contextlib.py:', 'futures.py:']


def _without_library_frames(tbs: list[str]) -> list[str]:
    """return the traceback lines of `tbs` outside click, contextlib, and futures."""
    return [tb for tb in tbs if not any(m in tb for m in MATCH)]


class FormattedTB(ultratb.FormattedTB):
    def structured_traceback(
        self, *args: typing.Any, **kwargs: typing.Any
    ) -> list[str]:
        """return the traceback lines of the base class without the library frames."""
        return _without_library_frames(super().structured_traceback(*args, **kwargs))


class VerboseTB(ultratb.VerboseTB):
    def structured_traceback(
        self, *args: typing.Any, **kwargs: typing.Any
    ) -> list[str]:
        """return the traceback lines of the base class without the library frames."""
        return _without_library_frames(super().structured_traceback(*args, **kwargs))
