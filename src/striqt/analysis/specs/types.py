from typing import Annotated, Literal, Union
from .helpers import Meta
import striqt.waveform as sw

AnalysisBandwidth = Annotated[float, Meta('Analysis bandwidth', 'Hz')]
AsXArray = Literal['delayed', True, False]
CellSSBSymbolIndexes = Annotated[
    sw.typing.CellSSBIndexes,
    Meta(
        'index locations in the SSB, or a cell search case from 3GPP TS 38.213 Sec. 4.1)'
    ),
]
Duration = Annotated[float, Meta('Duration of the analysis waveform', 's')]
MaxBeams = Annotated[
    int, Meta('limit the beam count in 5G sync evaluation', gt=0)
]
MaxLagSymbols = Annotated[
    int, Meta('limit the number of symbols of lag in the 5G sync correlator', ge=1, le=6)
]

PerPort = Annotated[
    bool, Meta('whether to evaluate signal synchronization separately on each port')
]
SampleRate = Annotated[float, Meta('Analysis sample rate', 'S/s')]
WindowFill = Annotated[
    float, Meta('Fraction of a symbol to fill with weighting function', gt=0, le=1)
]
WindowType = Union[str, tuple[str, float]]
