"""Tests for intentional-opening preservation in the libQEx wrapper.

These cover the pure-Python helpers added to preserve real mesh openings
(eyes, nostrils, open borders) while still filling small extraction-artifact
holes. They do NOT require the compiled ``pyqex`` module.
"""

import numpy as np
import pytest

# Importing the wrapper is safe even without the compiled pyqex extension
# (the import is guarded inside the module).
from rectangular_surface_parameterization.utils.libqex_wrapper import (
    _compute_boundary_vertex_set,
    _fill_holes_with_triangles,
)


def _open_grid(n=2):
    """An (n x n) quad grid of triangles lying on the z=0 plane (has a boundary)."""
    verts = []
    for j in range(n + 1):
        for i in range(n + 1):
            verts.append([float(i), float(j), 0.0])
    verts = np.array(verts, dtype=np.float64)

    tris = []
    stride = n + 1
    for j in range(n):
        for i in range(n):
            v00 = j * stride + i
            v10 = j * stride + i + 1
            v01 = (j + 1) * stride + i
            v11 = (j + 1) * stride + i + 1
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])
    return verts, np.array(tris, dtype=np.int32)


def _ring_quads():
    """A 3x3 quad grid (4x4 vertices) with the center quad removed.

    Produces exactly two boundary loops:
      - outer loop: 12 edges
      - inner hole: 4 edges (vertices 5, 6, 10, 9 on the 4x4 grid)
    """
    verts = []
    for j in range(4):
        for i in range(4):
            verts.append([float(i), float(j), 0.0])
    verts = np.array(verts, dtype=np.float64)

    def idx(i, j):
        return j * 4 + i

    quads = []
    for j in range(3):
        for i in range(3):
            if (i, j) == (1, 1):
                continue  # remove center -> inner hole
            quads.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])
    inner_loop = {idx(1, 1), idx(2, 1), idx(2, 2), idx(1, 2)}
    return verts, np.array(quads, dtype=np.int32), inner_loop


# =========================================================================
# _compute_boundary_vertex_set
# =========================================================================

def test_boundary_set_open_mesh_detects_perimeter():
    V, T = _open_grid(n=2)  # 3x3 vertices, perimeter = 8, center = 1 interior
    # quad vertices coincide with input vertices
    matched = _compute_boundary_vertex_set(V, T, V, verbose=False)
    # The single interior vertex (index 4, position (1,1)) must NOT be matched.
    assert 4 not in matched
    assert len(matched) == 8


def test_boundary_set_closed_mesh_is_empty():
    # Tetrahedron: closed surface, no boundary edges.
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    T = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    matched = _compute_boundary_vertex_set(V, T, V, verbose=False)
    assert matched == set()


# =========================================================================
# _fill_holes_with_triangles: preservation behavior
# =========================================================================

def test_ring_fills_both_loops_by_default():
    V, quads, _ = _ring_quads()
    tris, new_verts = _fill_holes_with_triangles(V, quads, verbose=False)
    # Both the outer boundary and the inner hole get a centroid -> 2 new verts.
    assert len(new_verts) == 2
    assert len(tris) > 0


def test_max_hole_size_skips_large_loop():
    V, quads, _ = _ring_quads()
    # Outer loop has 12 edges (> 6) -> skipped; inner hole has 4 edges -> filled.
    tris, new_verts = _fill_holes_with_triangles(
        V, quads, verbose=False, max_hole_size=6)
    assert len(new_verts) == 1


def test_original_boundary_loop_is_preserved():
    V, quads, inner_loop = _ring_quads()
    # Mark the inner hole vertices as original mesh boundary -> must stay open.
    tris, new_verts = _fill_holes_with_triangles(
        V, quads, verbose=False, original_boundary_verts=inner_loop)
    # Inner loop skipped (boundary), outer loop still filled -> 1 new vertex.
    assert len(new_verts) == 1


def test_boundary_and_size_skip_everything():
    V, quads, inner_loop = _ring_quads()
    tris, new_verts = _fill_holes_with_triangles(
        V, quads, verbose=False, max_hole_size=6,
        original_boundary_verts=inner_loop)
    # Outer skipped by size, inner skipped by boundary membership -> nothing filled.
    assert len(new_verts) == 0
    assert tris == []
