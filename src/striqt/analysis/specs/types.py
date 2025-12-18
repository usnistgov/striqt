from typing import Annotated, Literal, Union
from .helpers import Meta

DurationType = Annotated[float, Meta('Duration of the analysis waveform', 's')]
SampleRateType = Annotated[float, Meta('Analysis sample rate', 'S/s')]
AnalysisBandwidthType = Annotated[float, Meta('Analysis bandwidth', 'Hz')]
WindowType = Union[str, tuple[str, float]]
AsXArray = bool | Literal['delayed']
