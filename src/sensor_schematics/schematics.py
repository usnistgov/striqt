from schemdraw import elements, segments


def copy_element(el: elements.Element) -> elements.Element:
    return type(el)(**el.params)


class Attenuator(elements.Element2Term):
    """Resistor (IEEE/U.S. style)"""

    def __init__(self, *d, **kwargs):
        super().__init__(*d, **kwargs)

        resheight = 0.25 * 0.7  # Resistor height
        reswidth = 1.0 / 6 * 0.7  # Full (inner) length of resistor is 1.0 data unit
        woffset = (1 - reswidth * 6) / 2

        self.segments.append(
            segments.Segment(
                [
                    (0, 0),
                    (woffset, 0),
                    (0.5 * reswidth + woffset, resheight),
                    (1.5 * reswidth + woffset, -resheight),
                    (2.5 * reswidth + woffset, resheight),
                    (3.5 * reswidth + woffset, -resheight),
                    (4.5 * reswidth + woffset, resheight),
                    (5.5 * reswidth + woffset, -resheight),
                    (6 * reswidth + woffset, 0),
                ]
            )
        )

        self.segments.append(
            segments.Segment([(0, 0), (0, 0.5), (1, 0.5), (1, -0.5), (0, -0.5), (0, 0)])
        )


class AttenuatorVarIEEE(Attenuator):
    """Variable resistor (U.S. style)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        resheight = 0.25 * 0.7  # Resistor height
        reswidth = 1.0 / 6 * 0.7  # Full (inner) length of resistor is 1.0 data unit
        # woffset = (1-reswidth*6)/2

        self.segments.append(
            segments.Segment(
                [(1.5 * reswidth, -resheight * 2), (7.5 * reswidth, reswidth * 3.5)],
                arrow='->',
                arrowwidth=0.16,
                arrowlength=0.2,
            )
        )
