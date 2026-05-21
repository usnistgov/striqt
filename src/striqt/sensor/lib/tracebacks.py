from IPython.core import ultratb
import typing


MATCH = ['site-packages/click', 'contextlib.py:', 'futures.py:']


class FormattedTB(ultratb.FormattedTB):
    def structured_traceback(
        self,
        etype: type,
        evalue: typing.Optional[BaseException],
        etb=None,
        tb_offset=None,
        context: int = 5,
    ) -> list[str]:
        tbs = super().structured_traceback(type(evalue), evalue, etb, tb_offset, context)
        return [tb for tb in tbs if not any(m in tb for m in MATCH)]


class VerboseTB(ultratb.VerboseTB):
    def structured_traceback(
        self,
        etype: type,
        evalue: typing.Optional[BaseException],
        etb=None,
        tb_offset=None,
        context: int = 5,
    ) -> list[str]:
        print('verbose traceback', etype, evalue)
        tbs = super().structured_traceback(type(evalue), evalue, etb, tb_offset, context)
        return [tb for tb in tbs if not any(m in tb for m in MATCH)]
