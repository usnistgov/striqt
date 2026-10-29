# A plotting environment for notebooks

# figure titles and captions are written into the image 'Title' metadata for export
# to pdf or html
from __future__ import annotations

import datetime
import functools

import matplotlib.dates as mpldates
import matplotlib.pyplot as plt
import matplotlib.units as mplunits
import numpy as np
from IPython.display import HTML, display
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib_inline import backend_inline

_captions = {}


@functools.wraps(FigureCanvasSVG.print_svg)
def print_svg(self, *a, **k):
    def guess_title(fig):
        if self.figure._suptitle is not None:
            return self.figure._suptitle.get_text()

        for ax in self.figure.get_axes()[::-1]:
            title_ = ax.get_title()
            if title_:
                return title_
        return 'untitled'

    def title_to_label(title_):
        """replace 1 or more non-alphanumeric characters with '-'"""
        import re

        pattern = re.compile(r'[\W_]+')
        return pattern.sub('-', title_).lower()

    k = dict(k)
    label = title_to_label(guess_title(self.figure))
    caption_text = _captions.get(id(self.figure), '')
    title_ = f'{label}##{caption_text}' if caption_text else label
    k.setdefault('metadata', {})['Title'] = title_

    return FigureCanvasSVG._ps(self, *a, **k)  # type: ignore


FigureCanvasSVG.print_svg, FigureCanvasSVG._ps = (  # type: ignore
    print_svg,
    FigureCanvasSVG.print_svg,
)


@functools.wraps(backend_inline.set_matplotlib_formats)
def set_matplotlib_formats(formats, *args, **kws):
    """set inline figure formats and display a label and caption below each figure.

    The arguments are those of `backend_inline.set_matplotlib_formats`.
    The label is the figure suptitle, or the title of the last axes without a
    suptitle, lowercased with runs of non-alphanumeric characters replaced by ``-``.
    The caption is the text from `set_caption`. The IPython `print_figure` function
    is replaced process-wide.
    """

    backend_inline.set_matplotlib_formats(formats, *args, **kws)

    # monkeypatch IPython's internal print_figure to include title metadata
    from importlib import reload

    from IPython.core import pylabtools as pltt

    pltt = reload(pltt)

    def guess_title(fig):
        if fig._suptitle is not None:
            return fig._suptitle.get_text()

        for ax in fig.get_axes()[::-1]:
            title_ = ax.get_title()
            if title_:
                return title_
        return 'untitled'

    def title_to_label(title_):
        """replace 1 or more non-alphanumeric characters with '-'"""
        import re

        pattern = re.compile(r'[\W_]+')
        return pattern.sub('-', title_).lower()

    @functools.wraps(pltt.print_figure)
    def wrapper(fig, fmt='png', *a, **k):
        k = dict(k)
        label = title_to_label(guess_title(fig))
        caption_text = _captions.get(id(fig), '')

        ret = pltt._print_figure(fig, *a, fmt=fmt, **k)

        markup = f'<tt>{label}.{fmt}:</tt>{"<br>" + caption_text if caption_text else " (no caption data)"}'
        display(HTML(markup))

        return ret

    pltt.print_figure, pltt._print_figure = wrapper, pltt.print_figure  # type: ignore


def set_caption(*args):
    """set the caption of a figure in a jupyter notebook.

    Call as ``set_caption(fig, text)``, or as ``set_caption(text)`` for the current
    figure.

    Raises:
        ValueError: if called with neither 1 nor 2 positional arguments
    """
    if len(args) == 1:
        fig, text = plt.gcf(), args[0]
    elif len(args) == 2:
        fig, text = args
    else:
        raise ValueError(f'expected 1 or 2 args, but got {len(args)}')

    _captions[id(fig)] = text


convert_datetime = mplunits.registry[np.datetime64]

# concise date formatting by default
converter = mpldates.ConciseDateConverter()
mplunits.registry[np.datetime64] = converter
mplunits.registry[datetime.date] = converter
mplunits.registry[datetime.datetime] = converter

set_matplotlib_formats('svg')
