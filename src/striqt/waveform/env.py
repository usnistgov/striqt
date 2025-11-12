# A plotting environment for notebooks

# Set up the plotting environment for notebooks that convert cleanly
# to pdf or html output.
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import IPython
import IPython.display
from IPython.display import display, HTML, set_matplotlib_formats

import datetime
import functools

_captions = {}

from matplotlib.backends import backend_svg


@functools.wraps(backend_svg.FigureCanvasSVG.print_svg)
def print_svg(self, *a, **k):
    def guess_title(fig):
        if self.figure._suptitle is not None:
            return self.figure._suptitle.get_text()

        for ax in self.figure.get_axes()[::-1]:
            title_ = ax.get_title()
            if title_:
                return title_
        else:
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

    return backend_svg.FigureCanvasSVG._print_svg(self, *a, **k)


backend_svg.FigureCanvasSVG.print_svg, backend_svg.FigureCanvasSVG._print_svg = (
    print_svg,
    backend_svg.FigureCanvasSVG.print_svg,
)


@functools.wraps(IPython.display.set_matplotlib_formats)
def set_matplotlib_formats(formats, *args, **kws):
    """apply wrappers to inject title (from figure or axis titles) and caption (from set_caption metadata),
    when available, into image 'Title' metadata
    """

    IPython.display.set_matplotlib_formats(formats, *args, **kws)

    # monkeypatch IPython's internal print_figure to include title metadata
    from IPython.core import pylabtools

    from importlib import reload

    pylabtools = reload(pylabtools)

    def guess_title(fig):
        if fig._suptitle is not None:
            return fig._suptitle.get_text()

        for ax in fig.get_axes()[::-1]:
            title_ = ax.get_title()
            if title_:
                return title_
        else:
            return 'untitled'

    def title_to_label(title_):
        """replace 1 or more non-alphanumeric characters with '-'"""
        import re

        pattern = re.compile(r'[\W_]+')
        return pattern.sub('-', title_).lower()

    @functools.wraps(pylabtools.print_figure)
    def wrapper(fig, fmt='png', *a, **k):
        k = dict(k)
        label = title_to_label(guess_title(fig))
        caption_text = _captions.get(id(fig), '')

        ret = pylabtools._print_figure(fig, fmt=fmt, *a, **k)

        markup = f'<tt>{label}.{fmt}:</tt>{"<br>" + caption_text if caption_text else " (no caption data)"}'
        display(HTML(markup))

        return ret

    pylabtools.print_figure, pylabtools._print_figure = wrapper, pylabtools.print_figure


convert_datetime = mpl.units.registry[np.datetime64]


def set_caption(*args):
    """sets the caption in a jupyter notebook for the

    Usage: either set_caption(fig, text) or set_caption(text) to use the current figure
    """
    global _captions

    if len(args) == 1:
        fig, text = plt.gcf(), args[0]
    elif len(args) == 2:
        fig, text = args
    else:
        raise ValueError(f'expected 1 or 2 args, but got {len(args)}')

    _captions[id(fig)] = text


# concise date formatting by default
converter = mpl.dates.ConciseDateConverter()
mpl.units.registry[np.datetime64] = converter
mpl.units.registry[datetime.date] = converter
mpl.units.registry[datetime.datetime] = converter

set_matplotlib_formats('svg')
