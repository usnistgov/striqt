from IPython.core import ultratb

MATCH = ['site-packages/click', 'contextlib.py:']


class FormattedTB(ultratb.FormattedTB):
    def structured_traceback(self, *args, **kws) -> list[str]:
        tbs = super().structured_traceback(*args, **kws)
        return [tb for tb in tbs if not any(m in tb for m in MATCH)]


class VerboseTB(ultratb.VerboseTB):
    def structured_traceback(self, *args, **kws) -> list[str]:
        tbs = super().structured_traceback(*args, **kws)
        return [tb for tb in tbs if not any(m in tb for m in MATCH)]
