"""Tests for the JIT kernels in striqt.waveform.lib.jit.

The numba `_corr_at_indices` kernels (CPU and CUDA) are checked against a numpy
reference, and the `cupy.fuse` kernels in `jit.cuda` are checked against the numexpr
path of `striqt.waveform.lib.power_analysis` on the same data. The CUDA tests skip when
cupy is not available.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    dB_arrays,
    envelope_arrays,
    gaussian_iq,
    iq_waveforms,
    positive_power_arrays,
)
from hypothesis import given
from hypothesis import strategies as st
from numeric_checks import (
    FLOAT_DTYPES,
    assert_close,
    by_dtype,
    corr_atol,
    dtype_id,
    envelope_power_rtol,
    log_conversion_tol,
    pow_conversion_rtol,
)

from striqt.waveform import ofdm
from striqt.waveform.lib import power_analysis
from striqt.waveform.lib.arrays import float_dtype_like

# thread block sizing from ofdm.corr_at_indices
THREADS_PER_BLOCK = 32

# real and complex float32 envelopes within the range measured for the fused kernels
COMPLEX_ENVELOPES = envelope_arrays(
    min_magnitude=1e-4, max_magnitude=1e4, dtype=np.float32, min_dims=1, max_dims=1
)


def corr_reference(inds, x, nfft, ncp, norm):
    """float64 evaluation of the cyclic-prefix correlation computed by _corr_at_indices.

    Index pairs that run past the end of `x` are dropped, as the CPU kernel's zero-fill
    does for any index order.
    """
    xc = np.asarray(x).astype(np.complex128)
    inds = np.asarray(inds)
    out = np.empty(nfft + ncp, dtype=np.complex128)

    for j in range(nfft + ncp):
        ix = inds + j
        valid = ix + nfft < xc.shape[0]
        a = xc[ix[valid]]
        b = xc[ix[valid] + nfft]
        corr = np.sum(a * b.conj())
        if norm:
            corr /= np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2))
        else:
            corr /= inds.shape[0]
        out[j] = corr

    return out


def corr_cases(
    min_inds=1, max_inds=8, dtype=None, ncp_from_inds=False, sorted_inds=False
):
    """Strategy for (inds, x, nfft, ncp, norm) kernel arguments.

    The waveform length is drawn so that some examples run index pairs past the end of
    `x`, which exercises the kernels' bounds handling; the indices are drawn in
    arbitrary order unless `sorted_inds`, so an out-of-range pair may precede a valid
    one. With `ncp_from_inds`, ncp is the number of indices, as ofdm.corr_at_indices
    infers it from the last axis of `inds`.
    """

    @st.composite
    def _case(draw):
        nfft = draw(st.sampled_from([32, 64, 128]))
        n_inds = draw(st.integers(min_value=min_inds, max_value=max_inds))
        if ncp_from_inds:
            ncp = n_inds
        else:
            ncp = draw(st.integers(min_value=4, max_value=nfft // 4))
        positions = st.integers(min_value=0, max_value=2 * nfft)
        inds = draw(st.lists(positions, min_size=n_inds, max_size=n_inds, unique=True))
        inds = np.asarray(sorted(inds) if sorted_inds else inds, dtype=np.int64)
        # every lag keeps at least one valid pair (from the smallest index) so the
        # correlation is defined; larger indexes may still run past the end of x
        min_size = int(inds.min()) + 2 * nfft + ncp
        full_size = int(inds.max()) + 2 * nfft + ncp
        size = draw(st.integers(min_value=min_size, max_value=full_size))
        x = draw(iq_waveforms(min_size=size, max_size=size, dtype=dtype))
        norm = draw(st.booleans())
        return inds, x, nfft, ncp, norm

    return _case()


def assert_matches_reference(out, inds, x, nfft, ncp, norm):
    expected = corr_reference(inds, x, nfft, ncp, norm)
    assert_close(out, expected, atol=corr_atol(x, inds.size, norm))


class TestCorrAtIndicesKernels:
    """The numba kernels called directly, against the numpy reference."""

    @given(case=corr_cases())
    def test_cpu(self, case):
        from striqt.waveform.lib.jit.cpu import _corr_at_indices

        inds, x, nfft, ncp, norm = case
        out = np.empty(nfft + ncp, dtype=x.dtype)
        _corr_at_indices(inds, x, nfft, ncp, norm, out)
        assert_matches_reference(out, inds, x, nfft, ncp, norm)

    @staticmethod
    def _run_cuda(cp, inds, x, nfft, ncp, norm):
        from striqt.waveform.lib.jit.cuda import _corr_at_indices

        x_cp = cp.asarray(x)
        out = cp.empty(nfft + ncp, dtype=x.dtype)
        bpg = max((x_cp.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK, 1)
        _corr_at_indices[bpg, THREADS_PER_BLOCK](
            cp.asarray(inds), x_cp, nfft, ncp, norm, out
        )
        cp.cuda.Device().synchronize()
        return out

    # the CUDA kernel stops at the first out-of-range pair, which only matches the
    # reference for sorted indices; test_cuda_unsorted_indices covers the rest
    @given(case=corr_cases(sorted_inds=True))
    def test_cuda(self, cupy_available, case):
        inds, x, nfft, ncp, norm = case
        out = self._run_cuda(cupy_available, inds, x, nfft, ncp, norm)
        assert_matches_reference(out, inds, x, nfft, ncp, norm)

    @pytest.mark.xfail(
        strict=True,
        reason='the CUDA _corr_at_indices breaks out of the index loop at the first '
        'out-of-range pair (jit/cuda.py:25-26) instead of skipping it like the CPU '
        'kernel, so a valid index after an out-of-range one is dropped',
    )
    def test_cuda_unsorted_indices(self, cupy_available):
        nfft, ncp = 32, 4
        x = gaussian_iq(3 * nfft, seed=0)
        # the first index runs past the end of x at every lag; the second never does
        inds = np.array([2 * nfft, 0], dtype=np.int64)
        out = self._run_cuda(cupy_available, inds, x, nfft, ncp, False)
        assert_matches_reference(out, inds, x, nfft, ncp, False)


class TestCorrAtIndicesDispatcher:
    """ofdm.corr_at_indices selects the kernel for the array namespace."""

    @given(case=corr_cases(ncp_from_inds=True, sorted_inds=True))
    def test_sizes_out_from_inds(self, xp, case):
        inds, x, nfft, ncp, norm = case
        inds_2d = xp.asarray(inds[np.newaxis, :])
        x_xp = xp.asarray(x)

        out = ofdm.corr_at_indices(inds_2d, x_xp, nfft, norm=norm)
        assert isinstance(out, xp.ndarray)
        assert out.shape == (nfft + ncp,)
        assert out.dtype == x.dtype
        assert_matches_reference(out, inds, x, nfft, ncp, norm)

        buffer = xp.empty(nfft + ncp, dtype=x.dtype)
        ret = ofdm.corr_at_indices(inds_2d, x_xp, nfft, norm=norm, out=buffer)
        assert ret is buffer
        assert_matches_reference(ret, inds, x, nfft, ncp, norm)


# (kernel name, power_analysis function, its keyword arguments, kernel takes eps)
FUSED_LOG_KERNELS = [
    ('powtodB', power_analysis.powtodB, {'abs': True, 'eps': 0}, False),
    ('powtodB_noabs', power_analysis.powtodB, {'abs': False, 'eps': 0}, False),
    ('powtodB_eps', power_analysis.powtodB, {'abs': True, 'eps': 1e-6}, True),
    ('powtodB_eps_noabs', power_analysis.powtodB, {'abs': False, 'eps': 1e-6}, True),
    ('envtodB', power_analysis.envtodB, {'abs': True, 'eps': 0}, False),
    ('envtodB_noabs', power_analysis.envtodB, {'abs': False, 'eps': 0}, False),
    ('envtodB_eps', power_analysis.envtodB, {'abs': True, 'eps': 1e-6}, True),
    ('envtodB_eps_noabs', power_analysis.envtodB, {'abs': False, 'eps': 1e-6}, True),
]


class TestFusedKernelsCuda:
    """The cupy.fuse kernels in jit.cuda against the numexpr path on the same data.

    Tolerances are the two-implementation budgets of numeric_checks.
    """

    @staticmethod
    def _run(cp, name, x, *args):
        from striqt.waveform.lib.jit import cuda

        x_cp = cp.asarray(x)
        out = cp.empty(x.shape, dtype=float_dtype_like(x))
        ret = getattr(cuda, name)(x_cp, out, *args)
        cp.cuda.Device().synchronize()
        return ret

    @pytest.mark.parametrize(
        'name,func,kws,takes_eps',
        FUSED_LOG_KERNELS,
        ids=[k[0] for k in FUSED_LOG_KERNELS],
    )
    @pytest.mark.parametrize('dtype', FLOAT_DTYPES, ids=dtype_id)
    @given(data=st.data())
    def test_log_kernels(self, cupy_available, data, dtype, name, func, kws, takes_eps):
        scale = 10 if name.startswith('powtodB') else 20
        lim = by_dtype(dtype, float32=1e4, float64=1e6)
        powers = positive_power_arrays(
            min_value=1 / lim, max_value=lim, dtype=dtype, min_dims=1, max_dims=1
        )
        x = data.draw(powers)

        expected = func(x, **kws)
        args = (kws['eps'],) if takes_eps else ()
        result = self._run(cupy_available, name, x, *args)
        assert_close(result, expected, **log_conversion_tol(dtype, scale, n_impl=2))

    @given(env=COMPLEX_ENVELOPES)
    def test_envtodB_complex(self, cupy_available, env):
        expected = power_analysis.envtodB(env)
        result = self._run(cupy_available, 'envtodB', env)
        tol = log_conversion_tol(
            np.float32, 20, complex_input=np.iscomplexobj(env), n_impl=2
        )
        assert_close(result, np.real(expected), **tol)

    @given(env=COMPLEX_ENVELOPES)
    def test_envtopow(self, cupy_available, env):
        expected = power_analysis.envtopow(env)
        result = self._run(cupy_available, 'envtopow', env)
        rtol = envelope_power_rtol(
            np.float32, complex_input=np.iscomplexobj(env), n_impl=2
        )
        assert_close(result, expected, rtol=rtol)

    @pytest.mark.parametrize('dtype', FLOAT_DTYPES, ids=dtype_id)
    @given(data=st.data())
    def test_dBtopow(self, cupy_available, data, dtype):
        lim = by_dtype(dtype, float32=30, float64=100)
        levels = dB_arrays(
            min_value=-lim, max_value=lim, dtype=dtype, min_dims=1, max_dims=1
        )
        dB = data.draw(levels)

        expected = power_analysis.dBtopow(dB)
        result = self._run(cupy_available, 'dBtopow', dB)
        rtol = pow_conversion_rtol(dtype, np.abs(dB).max(), n_impl=2)
        assert_close(result, expected, rtol=rtol)
