"""Tests for the JIT kernels in striqt.waveform.lib.jit.

The numba `_corr_at_indices` kernels (CPU and CUDA) are checked against a numpy
reference and against each other, and the `cupy.fuse` kernels in `jit.cuda` are checked
against the numexpr path of `striqt.waveform.lib.power_analysis` on the same data. The
CUDA tests skip when cupy is not available.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import dB_arrays, envelope_arrays, iq_waveforms, positive_power_arrays
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from test_power_analysis import (
    ROUNDOFF_SAFETY,
    envelope_power_rtol,
    log_conversion_tol,
    pow_conversion_rtol,
    unit_roundoff,
)

from striqt.waveform.lib import power_analysis
from striqt.waveform.lib.arrays import float_dtype_like

PROPERTY = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
)

# thread block sizing from ofdm.corr_at_indices
THREADS_PER_BLOCK = 32


@pytest.fixture
def cupy_available():
    from conftest import _cupy

    if _cupy is None:
        pytest.skip('cupy is not available')
    return _cupy


def corr_reference(inds, x, nfft, ncp, norm):
    """float64 evaluation of the cyclic-prefix correlation computed by _corr_at_indices.

    Index pairs that run past the end of `x` are dropped, matching the kernels'
    zero-fill (CPU) and early exit (CUDA).
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


def corr_atol(x, n_inds, norm, n_impl=1):
    """absolute tolerance on _corr_at_indices against exact arithmetic.

    Each of the n_inds products a*conj(b) and the power terms are rounded at the input
    precision before the complex128 accumulation, and the result is rounded once more
    on output. With norm=True the Cauchy-Schwarz bound sum|a||b| <= sqrt(Pa*Pb) makes
    the error relative to a unit-scale output; with norm=False it is relative to the
    largest product magnitude.
    """
    u = unit_roundoff(float_dtype_like(x))
    if norm:
        scale = 1.0
    else:
        scale = float(np.abs(x).max() ** 2)
    return ROUNDOFF_SAFETY * n_impl * (n_inds + 3) * u * scale


def corr_cases(min_inds=1, max_inds=8, dtype=None, ncp_from_inds=False):
    """Strategy for (inds, x, nfft, ncp, norm) kernel arguments.

    The waveform length is drawn so that some examples run index pairs past the end of
    `x`, which exercises the kernels' bounds handling. With `ncp_from_inds`, ncp is the
    number of indices, as ofdm.corr_at_indices infers it from the last axis of `inds`.
    """

    @st.composite
    def _case(draw):
        nfft = draw(st.sampled_from([32, 64, 128]))
        n_inds = draw(st.integers(min_value=min_inds, max_value=max_inds))
        if ncp_from_inds:
            ncp = n_inds
        else:
            ncp = draw(st.integers(min_value=4, max_value=nfft // 4))
        inds = np.sort(
            np.asarray(
                draw(
                    st.lists(
                        st.integers(min_value=0, max_value=2 * nfft),
                        min_size=n_inds,
                        max_size=n_inds,
                        unique=True,
                    )
                ),
                dtype=np.int64,
            )
        )
        # every lag keeps at least one valid pair (from the smallest index) so the
        # correlation is defined; larger indexes may still run past the end of x
        min_size = int(inds.min()) + 2 * nfft + ncp
        full_size = int(inds.max()) + 2 * nfft + ncp
        size = draw(st.integers(min_value=min_size, max_value=full_size))
        x = draw(iq_waveforms(min_size=size, max_size=size, dtype=dtype))
        norm = draw(st.booleans())
        return inds, x, nfft, ncp, norm

    return _case()


class TestCorrAtIndicesCpu:
    """The numba CPU kernel against the numpy reference."""

    @PROPERTY
    @given(case=corr_cases())
    def test_matches_reference(self, case):
        from striqt.waveform.lib.jit.cpu import _corr_at_indices

        inds, x, nfft, ncp, norm = case
        out = np.empty(nfft + ncp, dtype=x.dtype)
        _corr_at_indices(inds, x, nfft, ncp, norm, out)

        expected = corr_reference(inds, x, nfft, ncp, norm)
        assert_allclose(out, expected, rtol=0, atol=corr_atol(x, inds.size, norm))

    @PROPERTY
    @given(case=corr_cases(ncp_from_inds=True))
    def test_dispatcher_numpy(self, case):
        """ofdm.corr_at_indices selects the CPU kernel and sizes `out` from `inds`."""
        from striqt.waveform import ofdm

        inds, x, nfft, ncp, norm = case
        inds_2d = inds[np.newaxis, :]

        out = ofdm.corr_at_indices(inds_2d, x, nfft, norm=norm)
        assert out.shape == (nfft + ncp,)
        assert out.dtype == x.dtype

        expected = corr_reference(inds, x, nfft, ncp, norm)
        assert_allclose(out, expected, rtol=0, atol=corr_atol(x, inds.size, norm))

    def test_dispatcher_reuses_out(self):
        from striqt.waveform import ofdm

        rng = np.random.default_rng(0)
        x = (rng.normal(size=512) + 1j * rng.normal(size=512)).astype(np.complex64)
        inds = np.arange(4)[np.newaxis, :]
        out = np.empty(64 + 4, dtype=np.complex64)

        ret = ofdm.corr_at_indices(inds, x, 64, out=out)
        assert ret is out


class TestCorrAtIndicesCuda:
    """The numba CUDA kernel against the CPU kernel and the numpy reference."""

    @staticmethod
    def _run_cuda(cp, inds, x, nfft, ncp, norm):
        from striqt.waveform.lib.jit.cuda import _corr_at_indices

        x_cp = cp.asarray(x)
        inds_cp = cp.asarray(inds)
        out = cp.empty(nfft + ncp, dtype=x.dtype)
        bpg = max((x_cp.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK, 1)
        _corr_at_indices[bpg, THREADS_PER_BLOCK](inds_cp, x_cp, nfft, ncp, norm, out)
        cp.cuda.Device().synchronize()
        return out.get()

    @PROPERTY
    @given(case=corr_cases())
    def test_matches_cpu_kernel(self, cupy_available, case):
        from striqt.waveform.lib.jit.cpu import _corr_at_indices

        inds, x, nfft, ncp, norm = case
        out_cpu = np.empty(nfft + ncp, dtype=x.dtype)
        _corr_at_indices(inds, x, nfft, ncp, norm, out_cpu)

        out_cuda = self._run_cuda(cupy_available, inds, x, nfft, ncp, norm)

        assert_allclose(
            out_cuda, out_cpu, rtol=0, atol=corr_atol(x, inds.size, norm, n_impl=2)
        )

    @PROPERTY
    @given(case=corr_cases())
    def test_matches_reference(self, cupy_available, case):
        inds, x, nfft, ncp, norm = case
        out = self._run_cuda(cupy_available, inds, x, nfft, ncp, norm)
        expected = corr_reference(inds, x, nfft, ncp, norm)
        assert_allclose(out, expected, rtol=0, atol=corr_atol(x, inds.size, norm))

    @PROPERTY
    @given(case=corr_cases(ncp_from_inds=True))
    def test_dispatcher_cupy(self, cupy_available, case):
        """ofdm.corr_at_indices on cupy input selects the CUDA kernel."""
        from striqt.waveform import ofdm

        cp = cupy_available
        inds, x, nfft, ncp, norm = case
        inds_2d = inds[np.newaxis, :]

        out = ofdm.corr_at_indices(cp.asarray(inds_2d), cp.asarray(x), nfft, norm=norm)
        cp.cuda.Device().synchronize()
        assert isinstance(out, cp.ndarray)
        assert out.shape == (nfft + ncp,)

        expected = corr_reference(inds, x, nfft, ncp, norm)
        assert_allclose(out.get(), expected, rtol=0, atol=corr_atol(x, inds.size, norm))

    def test_dispatcher_reuses_out(self, cupy_available):
        from striqt.waveform import ofdm

        cp = cupy_available
        rng = np.random.default_rng(0)
        x = cp.asarray(
            (rng.normal(size=512) + 1j * rng.normal(size=512)).astype(np.complex64)
        )
        inds = cp.arange(4)[cp.newaxis, :]
        out = cp.empty(64 + 4, dtype=np.complex64)

        ret = ofdm.corr_at_indices(inds, x, 64, out=out)
        assert ret is out


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

    Tolerances are the two-implementation budgets from test_power_analysis.
    """

    @staticmethod
    def _run(cp, name, x, *args):
        from striqt.waveform.lib.jit import cuda

        x_cp = cp.asarray(x)
        out = cp.empty(x.shape, dtype=float_dtype_like(x))
        ret = getattr(cuda, name)(x_cp, out, *args)
        cp.cuda.Device().synchronize()
        return ret.get()

    @PROPERTY
    @given(data=st.data(), dtype=st.sampled_from([np.float32, np.float64]))
    @pytest.mark.parametrize(
        'name,func,kws,takes_eps',
        FUSED_LOG_KERNELS,
        ids=[k[0] for k in FUSED_LOG_KERNELS],
    )
    def test_log_kernels(self, cupy_available, data, dtype, name, func, kws, takes_eps):
        scale = 10 if name.startswith('powtodB') else 20
        lim = {np.float64: 1e6, np.float32: 1e4}[dtype]
        x = data.draw(
            positive_power_arrays(
                min_value=1 / lim, max_value=lim, dtype=dtype, min_dims=1, max_dims=1
            )
        )

        expected = func(x, **kws)
        args = (kws['eps'],) if takes_eps else ()
        result = self._run(cupy_available, name, x, *args)

        rtol, atol = log_conversion_tol(dtype, scale, n_impl=2)
        assert_allclose(result, expected, rtol=rtol, atol=atol)

    @PROPERTY
    @given(
        env=envelope_arrays(
            include_complex=True,
            min_magnitude=1e-4,
            max_magnitude=1e4,
            dtype=np.float32,
            min_dims=1,
            max_dims=1,
        )
    )
    def test_envtodB_complex(self, cupy_available, env):
        expected = power_analysis.envtodB(env)
        result = self._run(cupy_available, 'envtodB', env)
        rtol, atol = log_conversion_tol(
            np.float32, 20, complex_input=np.iscomplexobj(env), n_impl=2
        )
        assert_allclose(result, np.real(expected), rtol=rtol, atol=atol)

    @PROPERTY
    @given(
        env=envelope_arrays(
            include_complex=True,
            min_magnitude=1e-4,
            max_magnitude=1e4,
            dtype=np.float32,
            min_dims=1,
            max_dims=1,
        )
    )
    def test_envtopow(self, cupy_available, env):
        expected = power_analysis.envtopow(env)
        result = self._run(cupy_available, 'envtopow', env)
        rtol = envelope_power_rtol(
            np.float32, complex_input=np.iscomplexobj(env), n_impl=2
        )
        assert_allclose(result, expected, rtol=rtol)

    @PROPERTY
    @given(data=st.data(), dtype=st.sampled_from([np.float32, np.float64]))
    def test_dBtopow(self, cupy_available, data, dtype):
        lim = {np.float64: 100, np.float32: 30}[dtype]
        dB = data.draw(
            dB_arrays(
                min_value=-lim, max_value=lim, dtype=dtype, min_dims=1, max_dims=1
            )
        )

        expected = power_analysis.dBtopow(dB)
        result = self._run(cupy_available, 'dBtopow', dB)

        rtol = pow_conversion_rtol(dtype, np.abs(dB).max(), n_impl=2)
        assert_allclose(result, expected, rtol=rtol)
