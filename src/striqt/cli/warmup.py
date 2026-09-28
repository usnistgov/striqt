"""warm the numba JIT cache for the current platform.

Run once after install on a new machine. The cache persists in the user cache
directory, so a later run only reloads it; on a Jetson TX2i the first run takes about
2 minutes.
"""

import click
import numpy as np


@click.command()
def cli() -> None:
    """warm the numba JIT caches of the CPU and CUDA `corr_at_indices` kernels.

    The CPU kernel is compiled for complex64 input. The CUDA kernel is compiled when
    `cupy` imports; any failure there is echoed and skipped rather than raised.
    Progress is echoed to stdout.
    """
    import striqt.waveform.lib.jit  # noqa: F401  triggers NUMBA_CACHE_DIR setup + stamp patch

    from striqt.waveform.lib.jit.cpu import _corr_at_indices

    nfft, ncp = 64, 16
    inds = np.arange(4, dtype=np.int64)

    for dtype in (np.complex64,):
        x = np.zeros(nfft + ncp + 128, dtype=dtype)
        out = np.empty(nfft + ncp, dtype=dtype)
        _corr_at_indices(inds, x, nfft, ncp, True, out)
        _corr_at_indices(inds, x, nfft, ncp, False, out)
        click.echo(f'  CPU kernel ({np.dtype(dtype).name}) warmed')

    try:
        import cupy as cp  # ty: ignore[unresolved-import]
        from striqt.waveform.lib.jit.cuda import _corr_at_indices as _cuda_corr

        x_gpu = cp.zeros(nfft + ncp + 128, dtype=np.complex64)
        inds_gpu = cp.asarray(inds)
        out_gpu = cp.empty(nfft + ncp, dtype=np.complex64)
        tpb = 32
        bpg = max((x_gpu.size + tpb - 1) // tpb, 1)
        _cuda_corr[bpg, tpb](inds_gpu, x_gpu, nfft, ncp, True, out_gpu)
        cp.cuda.Device().synchronize()
        click.echo('  CUDA kernel warmed')
    except (ImportError, Exception) as exc:
        click.echo(f'  CUDA kernel skipped ({exc})')
