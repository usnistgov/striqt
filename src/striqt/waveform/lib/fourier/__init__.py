"""Fourier-domain signal processing on numpy and cupy: FFTs with a roundoff model,
window design, the STFT and frequency-domain edits of an STFT, overlap-add filters,
and rational resamplers.

The package namespace contains only the submodules. A name is reached as
``fourier.stft.stft`` or ``fourier.windows.get_window``. `striqt.waveform.fourier`
re-exports the public names.
"""

from . import fft, filters, resamplers, spectrum, stft, windows
