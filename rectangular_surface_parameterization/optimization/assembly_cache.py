"""
Assembly caches for the RSP Newton solver.

The Newton iteration in ``optimize_RSP`` re-evaluates the integrability
oracle and the objective many times. The original port rebuilt several
sparse operators from scratch on every evaluation, even though most of
their structure is constant for a given mesh:

* the zeroed dual gradient ``d0d`` (only depends on ``dec`` and the
  hard/boundary edge sets),
* the left block of ``O`` (``-star1p @ d0p_tri``), and
* the (row, col) index patterns of ``Dv_tri``, ``dO`` and ``D_vth``.

This module provides:

1. :func:`get_assembly_cache` / :func:`get_zeroed_d0d` -- cache the
   topology-dependent operators once per mesh (improvement: eliminate
   repeated assembly by caching topology and sparse patterns).
2. :class:`FixedPatternCSR` -- build a sparse pattern once and only
   scatter new values into it on subsequent iterations (improvement:
   separate pattern and values, building indices once and updating only
   values).
3. An optional Numba-accelerated scatter-add used by
   :class:`FixedPatternCSR`, with an identical NumPy fallback
   (improvement: replace the assembly inner loop with Numba when it is
   available, without making it a hard dependency).

All transformations implemented here are mathematically identical to the
original ``coo_matrix(...).tocsr()`` based assembly; only the amount of
repeated work changes.
"""

import numpy as np
import scipy.sparse as sp

try:  # Optional acceleration. Pure-NumPy fallback is used when absent.
    from numba import njit

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - numba is an optional dependency
    _HAS_NUMBA = False


if _HAS_NUMBA:

    @njit(cache=True)
    def _scatter_add_numba(inverse, values, nnz):  # pragma: no cover - jit
        out = np.zeros(nnz, dtype=np.float64)
        for k in range(inverse.shape[0]):
            out[inverse[k]] += values[k]
        return out


def _scatter_add(inverse, values, nnz):
    """Scatter-add ``values`` into ``nnz`` bins indexed by ``inverse``.

    Equivalent to ``np.bincount(inverse, weights=values, minlength=nnz)``
    and to the duplicate-summation performed by ``coo_matrix.tocsr()``.
    """
    values = np.ascontiguousarray(values, dtype=np.float64)
    if _HAS_NUMBA:
        return _scatter_add_numba(inverse, values, nnz)
    return np.bincount(inverse, weights=values, minlength=nnz).astype(
        np.float64, copy=False
    )


class FixedPatternCSR:
    """Assemble CSR matrices that all share a fixed sparsity pattern.

    The ``(row, col)`` structure is analyzed once. Subsequent ``build``
    calls only scatter-add new values into the precomputed canonical
    pattern, avoiding the COO -> CSR sort/deduplicate that would otherwise
    run on every solver iteration.

    The result of ``build(values)`` is identical to
    ``scipy.sparse.coo_matrix((values, (rows, cols)), shape=shape).tocsr()``.
    """

    __slots__ = ("shape", "indptr", "indices", "nnz", "inverse")

    def __init__(self, rows, cols, shape):
        rows = np.asarray(rows, dtype=np.int64).ravel()
        cols = np.asarray(cols, dtype=np.int64).ravel()
        if rows.shape != cols.shape:
            raise ValueError("rows and cols must have the same length")

        self.shape = (int(shape[0]), int(shape[1]))

        # Canonical CSR pattern (duplicates merged, indices sorted).
        pattern = sp.csr_matrix(
            (np.ones(rows.shape[0], dtype=np.float64), (rows, cols)),
            shape=self.shape,
        )
        pattern.sum_duplicates()
        pattern.sort_indices()

        self.indptr = pattern.indptr
        self.indices = pattern.indices
        self.nnz = int(pattern.nnz)

        # Map every input entry to its slot in the canonical pattern.
        # With sorted indices the canonical linear keys (row * ncols + col)
        # are strictly increasing, so searchsorted yields the exact slot.
        ncols = np.int64(self.shape[1])
        if self.nnz > 0:
            row_of_slot = np.repeat(
                np.arange(self.shape[0], dtype=np.int64), np.diff(self.indptr)
            )
            canon_keys = row_of_slot * ncols + self.indices.astype(np.int64)
            entry_keys = rows * ncols + cols
            self.inverse = np.searchsorted(canon_keys, entry_keys).astype(np.intp)
        else:
            self.inverse = np.zeros(0, dtype=np.intp)

    def build(self, values):
        """Return a CSR matrix with the fixed pattern and the given values."""
        data = _scatter_add(
            self.inverse, np.asarray(values, dtype=np.float64).ravel(), self.nnz
        )
        return sp.csr_matrix(
            (data, self.indices, self.indptr), shape=self.shape, copy=False
        )


class _AssemblyCache:
    """Container for the per-mesh cached operators (attached to ``dec``)."""

    __slots__ = (
        "ne",
        "nf",
        "t2e_id",
        "dv_signs",
        "dv_assembler",
        "dO_assembler",
        "O_left",
        "dvth_assembler",
    )


def get_assembly_cache(dec, mesh):
    """Return (lazily building) the assembly cache attached to ``dec``.

    The cache is keyed on the mesh sizes and the identity of ``mesh.T2E``
    so it is automatically rebuilt if ``dec`` is ever paired with a
    different mesh.
    """
    ne = int(mesh.num_edges)
    nf = int(mesh.num_faces)
    t2e_id = id(mesh.T2E)

    cache = getattr(dec, "_rsp_assembly_cache", None)
    if (
        cache is not None
        and cache.ne == ne
        and cache.nf == nf
        and cache.t2e_id == t2e_id
    ):
        return cache

    cache = _AssemblyCache()
    cache.ne = ne
    cache.nf = nf
    cache.t2e_id = t2e_id

    edge_idx = mesh.T2E.indices  # (nf, 3) 0-based edge indices
    edge_sign = mesh.T2E.signs  # (nf, 3) signs
    corner_indices = np.arange(3 * nf).reshape((nf, 3), order="F")  # (nf, 3)

    # --- omega_from_scale: Dv_tri pattern (ne x 3nf), 9 entries per face ---
    I_arr = np.column_stack(
        [
            edge_idx[:, 0], edge_idx[:, 0], edge_idx[:, 0],
            edge_idx[:, 1], edge_idx[:, 1], edge_idx[:, 1],
            edge_idx[:, 2], edge_idx[:, 2], edge_idx[:, 2],
        ]
    ).ravel()
    J_arr = np.column_stack(
        [
            corner_indices[:, 0], corner_indices[:, 1], corner_indices[:, 2],
            corner_indices[:, 0], corner_indices[:, 1], corner_indices[:, 2],
            corner_indices[:, 0], corner_indices[:, 1], corner_indices[:, 2],
        ]
    ).ravel()
    cache.dv_signs = np.column_stack(
        [
            edge_sign[:, 0], edge_sign[:, 0], edge_sign[:, 0],
            edge_sign[:, 1], edge_sign[:, 1], edge_sign[:, 1],
            edge_sign[:, 2], edge_sign[:, 2], edge_sign[:, 2],
        ]
    ).ravel()
    cache.dv_assembler = FixedPatternCSR(I_arr, J_arr, (ne, 3 * nf))

    # --- omega_from_scale derivative: dO pattern (ne x nf) ---
    dO_I = np.concatenate([edge_idx[:, 0], edge_idx[:, 1], edge_idx[:, 2]])
    dO_J = np.tile(np.arange(nf), 3)
    cache.dO_assembler = FixedPatternCSR(dO_I, dO_J, (ne, nf))

    # --- invariant left block of O: -star1p @ d0p_tri (ne x 3nf) ---
    cache.O_left = (-(dec.star1p @ dec.d0p_tri)).tocsr()

    # --- integrability: D_vth pattern (nf x 3nf) ---
    I_rows = np.tile(np.arange(nf).reshape(-1, 1), (1, 3))
    J_cols = np.arange(3 * nf).reshape((nf, 3), order="F")
    cache.dvth_assembler = FixedPatternCSR(
        I_rows.ravel("F"), J_cols.ravel("F"), (nf, 3 * nf)
    )

    dec._rsp_assembly_cache = cache
    return cache


def _zero_csr_rows(mat, rows):
    """Return a copy of ``mat`` (CSR) with the given rows set to zero."""
    mat = mat.tocsr()
    keep = np.ones(mat.shape[0], dtype=np.float64)
    keep[rows] = 0.0
    return (sp.diags(keep) @ mat).tocsr()


def get_zeroed_d0d(dec, param):
    """Return ``dec.d0d`` with hard and boundary edge rows zeroed.

    The result only depends on ``dec.d0d`` and on the hard/boundary edge
    sets, so it is cached on ``dec`` and reused across solver iterations.
    """
    d0d = dec.d0d
    ide_hard = getattr(param, "ide_hard", None)
    ide_bound = getattr(param, "ide_bound", None)

    cache = getattr(dec, "_rsp_d0d_cache", None)
    if (
        cache is not None
        and cache["d0d_id"] == id(d0d)
        and cache["hard_id"] == id(ide_hard)
        and cache["bound_id"] == id(ide_bound)
    ):
        return cache["d0d_zeroed"]

    rows = []
    if ide_hard is not None and len(ide_hard) > 0:
        rows.append(np.asarray(ide_hard, dtype=np.int64).ravel())
    if ide_bound is not None and len(ide_bound) > 0:
        rows.append(np.asarray(ide_bound, dtype=np.int64).ravel())

    if rows:
        d0d_zeroed = _zero_csr_rows(d0d, np.concatenate(rows))
    else:
        d0d_zeroed = d0d.tocsr()

    dec._rsp_d0d_cache = {
        "d0d_id": id(d0d),
        "hard_id": id(ide_hard),
        "bound_id": id(ide_bound),
        "d0d_zeroed": d0d_zeroed,
    }
    return d0d_zeroed
