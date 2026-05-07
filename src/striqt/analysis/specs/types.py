from typing import Annotated, Literal, Union
from .helpers import Meta
import striqt.waveform as sw

DurationType = Annotated[float, Meta('Duration of the analysis waveform', 's')]
SampleRateType = Annotated[float, Meta('Analysis sample rate', 'S/s')]
AnalysisBandwidthType = Annotated[float, Meta('Analysis bandwidth', 'Hz')]
WindowType = Union[str, tuple[str, float]]
AsXArray = Literal['delayed', True, False]
CellSSBSymbolIndexes = Annotated[sw.typing.CellSSBIndexes, Meta('index locations in the SSB, or a cell search case from 3GPP TS 38.213 Sec. 4.1)')]