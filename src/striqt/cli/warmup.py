"""run the first-use compilation of the GPU kernels for the current platform.

cupy compiles the `cupy.fuse` and `ReductionKernel` detector kernels on first call
and caches the compiled kernels on disk across runs. Without cupy, the command runs
the cyclic-prefix correlation on numpy and compiles nothing.
"""

import click
import numpy as np


@click.command()
def cli() -> None:
    """compile the cupy detector and dB kernels, when cupy is importable.

    The cyclic-prefix correlation runs on a small complex64 waveform on numpy. With
    cupy importable, the correlation, the ``'mean'`` and ``'max'`` binned power
    detectors, and the dB conversion run on the same waveform as a cupy array.
    Progress is echoed to stdout. A failure in the cupy step is echoed instead of
    raised.
    """
    import striqt.waveform as sw

    nfft, ncp = 64, 16
    inds = (
        np.arange(ncp, dtype=np.int64)[np.newaxis, :]
        + (nfft + ncp) * np.arange(4, dtype=np.int64)[:, np.newaxis]
    )
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4 * (nfft + ncp) + nfft, 2)).view(np.complex128)
    x = x[:, 0].astype(np.complex64)

    sw.comms_5gnr.correlate_cyclic_shift(inds, x, nfft, norm=True)
    sw.comms_5gnr.correlate_cyclic_shift(inds, x, nfft, norm=False)
    click.echo('  cyclic-prefix correlation (numpy, complex64) run')

    try:
        import cupy as cp  # ty: ignore[import-not-found]

        x_gpu = cp.asarray(x)
        inds_gpu = cp.asarray(inds)
        sw.comms_5gnr.correlate_cyclic_shift(inds_gpu, x_gpu, nfft, norm=True)
        sw.comms_5gnr.correlate_cyclic_shift(inds_gpu, x_gpu, nfft, norm=False)
        click.echo('  cyclic-prefix correlation (cupy, complex64) run')

        for kind in ('mean', 'max'):
            sw.power_analysis.iq_to_bin_power(x_gpu, Ts=1.0, Tbin=float(ncp), kind=kind)
            click.echo(f'  {kind!r} power detector kernel compiled')

        sw.powtodB(cp.abs(x_gpu) ** 2 + 1)
        click.echo('  dB conversion kernel compiled')

        cp.cuda.Device().synchronize()
    except Exception as exc:
        click.echo(f'  cupy step skipped ({exc})')
