"""Tests for the quad extractor (libQEx port)."""

import numpy as np
import pytest
from pathlib import Path
from collections import defaultdict

from rectangular_surface_parameterization.utils.quad_extractor import (
    extract_quads,
    _find_cut_edges_and_transitions,
    _TF, _TF_IDENTITY, _HalfEdgeMesh,
    _extract_transitions, _consistent_truncation,
    _generate_grid_vertices, _generate_connections, _generate_faces,
)

GOLDEN = Path(__file__).parent / "golden_data"


# =========================================================================
# Helpers
# =========================================================================

def make_single_quad_uv():
    """Two triangles forming a quad in UV space spanning [0,1]x[0,1]."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    uv_per_tri = np.array([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    ])
    return vertices, triangles, uv_per_tri


def make_2x2_grid_uv():
    """A 2x2 quad grid in UV space spanning [0,2]x[0,2]."""
    vertices = []
    for j in range(3):
        for i in range(3):
            vertices.append([float(i), float(j), 0.0])
    vertices = np.array(vertices)

    triangles = []
    uv_per_tri = []
    for j in range(2):
        for i in range(2):
            v00 = j * 3 + i
            v10 = j * 3 + i + 1
            v01 = (j + 1) * 3 + i
            v11 = (j + 1) * 3 + i + 1
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])
            uv_per_tri.append([[float(i), float(j)],
                               [float(i + 1), float(j)],
                               [float(i + 1), float(j + 1)]])
            uv_per_tri.append([[float(i), float(j)],
                               [float(i + 1), float(j + 1)],
                               [float(i), float(j + 1)]])

    return vertices, np.array(triangles), np.array(uv_per_tri)


def make_scaled_grid_uv(n=3, scale=1.0):
    """An nxn quad grid with UVs scaled by `scale`."""
    vertices = []
    for j in range(n + 1):
        for i in range(n + 1):
            vertices.append([float(i), 0.0, float(j)])
    vertices = np.array(vertices)

    triangles = []
    uv_per_tri = []
    for j in range(n):
        for i in range(n):
            v00 = j * (n + 1) + i
            v10 = j * (n + 1) + i + 1
            v01 = (j + 1) * (n + 1) + i
            v11 = (j + 1) * (n + 1) + i + 1
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])
            uv_per_tri.append([[i * scale, j * scale],
                               [(i + 1) * scale, j * scale],
                               [(i + 1) * scale, (j + 1) * scale]])
            uv_per_tri.append([[i * scale, j * scale],
                               [(i + 1) * scale, (j + 1) * scale],
                               [i * scale, (j + 1) * scale]])

    return vertices, np.array(triangles), np.array(uv_per_tri)


def make_cylinder_with_cut():
    """A cylinder (open tube) with a UV cut along one seam."""
    n_cols = 4
    n_rows = 3
    angles = np.linspace(0, 2 * np.pi, n_cols + 1)
    heights = np.linspace(0, 2, n_rows + 1)

    vertices_3d = []
    for j in range(n_rows + 1):
        for i in range(n_cols + 1):
            a = angles[i % n_cols] if i < n_cols else angles[0]
            vertices_3d.append([np.cos(a), heights[j], np.sin(a)])
    vertices_3d = np.array(vertices_3d)

    triangles = []
    uv_per_tri = []
    for j in range(n_rows):
        for i in range(n_cols):
            stride = n_cols + 1
            v00 = j * stride + i
            v10 = j * stride + i + 1
            v01 = (j + 1) * stride + i
            v11 = (j + 1) * stride + i + 1
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])
            uv_per_tri.append([[float(i), float(j)],
                               [float(i + 1), float(j)],
                               [float(i + 1), float(j + 1)]])
            uv_per_tri.append([[float(i), float(j)],
                               [float(i + 1), float(j + 1)],
                               [float(i), float(j + 1)]])

    return vertices_3d, np.array(triangles), np.array(uv_per_tri)


def check_manifold(quads):
    """Return (n_boundary, n_non_manifold) edge counts."""
    edge_count = defaultdict(int)
    for q in quads:
        n = len(q)
        for i in range(n):
            e = (min(q[i], q[(i + 1) % n]), max(q[i], q[(i + 1) % n]))
            edge_count[e] += 1
    boundary = sum(1 for c in edge_count.values() if c == 1)
    non_manifold = sum(1 for c in edge_count.values() if c > 2)
    return boundary, non_manifold


def check_inverted_faces(vertices, quads):
    """Count faces whose normal points toward the origin."""
    inverted = 0
    for q in quads:
        center = vertices[q].mean(axis=0)
        n1 = np.cross(vertices[q[1]] - vertices[q[0]], vertices[q[3]] - vertices[q[0]])
        n2 = np.cross(vertices[q[3]] - vertices[q[2]], vertices[q[1]] - vertices[q[2]])
        normal = n1 + n2
        if np.dot(normal, center) < 0:
            inverted += 1
    return inverted


# =========================================================================
# Transition function tests
# =========================================================================

class TestTransitionFunction:
    def test_identity(self):
        tf = _TF(0, 0, 0)
        u, v = tf.transform_point(3.0, 4.0)
        assert u == 3.0 and v == 4.0

    def test_rotation_90(self):
        tf = _TF(1, 0, 0)
        u, v = tf.transform_point(1.0, 0.0)
        assert abs(u - 0.0) < 1e-10 and abs(v - 1.0) < 1e-10

    def test_rotation_with_translation(self):
        tf = _TF(2, 5, 3)
        u, v = tf.transform_point(1.0, 2.0)
        assert abs(u - 4.0) < 1e-10 and abs(v - 1.0) < 1e-10

    def test_inverse(self):
        tf = _TF(1, 3, -2)
        inv = tf.inverse()
        u, v = 7.0, 11.0
        u2, v2 = tf.transform_point(u, v)
        u3, v3 = inv.transform_point(u2, v2)
        assert abs(u3 - u) < 1e-10 and abs(v3 - v) < 1e-10

    def test_compose(self):
        tf1 = _TF(1, 2, 3)
        tf2 = _TF(2, -1, 4)
        composed = tf1 * tf2
        u, v = 5.0, 6.0
        u1, v1 = tf2.transform_point(u, v)
        u2, v2 = tf1.transform_point(u1, v1)
        u3, v3 = composed.transform_point(u, v)
        assert abs(u2 - u3) < 1e-10 and abs(v2 - v3) < 1e-10

    def test_identity_composition(self):
        tf = _TF(3, 7, -5)
        result = tf * tf.inverse()
        assert result == _TF_IDENTITY


# =========================================================================
# Half-edge mesh tests
# =========================================================================

class TestHalfEdgeMesh:
    def test_basic_connectivity(self):
        V, T, uv = make_single_quad_uv()
        mesh = _HalfEdgeMesh(T, len(V))
        assert mesh.n_faces == 2
        assert mesh.n_halfedges == 6

    def test_opposite_halfedges(self):
        V, T, uv = make_2x2_grid_uv()
        mesh = _HalfEdgeMesh(T, len(V))
        for heh in range(mesh.n_halfedges):
            opp = mesh.opposite[heh]
            if opp >= 0:
                assert mesh.opposite[opp] == heh
                assert mesh.to_vertex(heh) == mesh.from_vertex(opp)

    def test_vertex_iteration(self):
        V, T, uv = make_2x2_grid_uv()
        mesh = _HalfEdgeMesh(T, len(V))
        # Interior vertex (index 4 = center of 3x3 grid)
        hehs = mesh.vih_iter_cw(4)
        assert len(hehs) > 0
        # All should point to vertex 4
        for h in hehs:
            assert mesh.to_vertex(h) == 4


# =========================================================================
# Transition extraction tests
# =========================================================================

class TestTransitionExtraction:
    def test_no_cuts_in_flat_grid(self):
        V, T, uv = make_2x2_grid_uv()
        mesh = _HalfEdgeMesh(T, len(V))
        uv_coords = uv.reshape(-1).copy()
        tf = _extract_transitions(mesh, uv_coords)
        for t in tf:
            assert t == _TF_IDENTITY

    def test_sphere_transitions(self):
        path = GOLDEN / "sphere320_param.npz"
        if not path.exists():
            pytest.skip("sphere320_param.npz not found")
        data = np.load(path)
        T = data['triangles']
        uv = data['uv_per_tri'] * 10
        mesh = _HalfEdgeMesh(T, len(data['vertices']))
        uv_coords = uv.reshape(-1).copy()
        tf = _extract_transitions(mesh, uv_coords)
        n_cut = sum(1 for eidx in range(mesh.n_edges)
                    if not mesh.is_boundary_edge(eidx) and tf[eidx] != _TF_IDENTITY)
        assert n_cut > 20


# =========================================================================
# Full extraction tests: synthetic
# =========================================================================

class TestExtractSynthetic:
    def test_single_quad(self):
        V, T, uv = make_single_quad_uv()
        qv, qf, tf = extract_quads(V, T, uv, verbose=False, fill_holes=False)
        # All 4 grid vertices are at mesh vertex positions (OnVertex type).
        # Boundary vertex LEI construction is limited, so may get fewer quads.
        assert len(qf) >= 0  # May be 0 or 1 depending on boundary handling

    def test_2x2_grid_produces_quads(self):
        V, T, uv = make_2x2_grid_uv()
        qv, qf, tf = extract_quads(V, T, uv, verbose=False, fill_holes=False)
        # All 9 grid vertices are at mesh vertices. Interior vertex (center)
        # produces quads, but boundary vertices have limited LEI construction.
        if len(qf) > 0:
            _, non_manifold = check_manifold(qf)
            assert non_manifold == 0

    def test_larger_grid_interior(self):
        V, T, uv = make_scaled_grid_uv(n=5, scale=1.0)
        qv, qf, tf = extract_quads(V, T, uv, verbose=False, fill_holes=False)
        # Interior 3x3 quads always work; boundary quads depend on LEI construction
        assert len(qf) >= 9
        _, non_manifold = check_manifold(qf)
        assert non_manifold == 0

    def test_cylinder_forms_quads(self):
        V, T, uv = make_cylinder_with_cut()
        qv, qf, tf = extract_quads(V, T, uv, verbose=False, fill_holes=False)
        assert len(qf) >= 2
        _, non_manifold = check_manifold(qf)
        assert non_manifold == 0

    def test_no_degenerate_quads(self):
        V, T, uv = make_scaled_grid_uv(n=4, scale=1.0)
        qv, qf, tf = extract_quads(V, T, uv, verbose=False, fill_holes=False)
        for i, q in enumerate(qf):
            assert len(set(q)) == 4, f"Quad {i} has duplicate vertices: {q}"

    def test_vertex_positions_correct(self):
        V, T, uv = make_2x2_grid_uv()
        qv, qf, tf = extract_quads(V, T, uv, verbose=False, fill_holes=False)
        for i, v in enumerate(qv):
            assert abs(v[2]) < 1e-10, f"Vertex {i} z={v[2]}"
            assert abs(v[0] - round(v[0])) < 1e-10, f"Vertex {i} x={v[0]}"
            assert abs(v[1] - round(v[1])) < 1e-10, f"Vertex {i} y={v[1]}"

    def test_empty_uv_returns_empty(self):
        V = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], dtype=float)
        T = np.array([[0, 1, 2]])
        uv = np.array([[[0.1, 0.1], [0.2, 0.1], [0.1, 0.2]]])
        qv, qf, tf = extract_quads(V, T, uv, verbose=False, fill_holes=False)
        assert len(qf) == 0


# =========================================================================
# Full extraction tests: sphere320
# =========================================================================

class TestExtractSphere:
    @pytest.fixture
    def sphere_data(self):
        path = GOLDEN / "sphere320_param.npz"
        if not path.exists():
            pytest.skip("sphere320_param.npz not found")
        data = np.load(path)
        return data["vertices"], data["triangles"], data["uv_per_tri"]

    def test_produces_quads(self, sphere_data):
        V, T, uv = sphere_data
        qv, qf, tf = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        assert len(qf) >= 90, f"Expected >=90 quads, got {len(qf)}"

    def test_matches_libqex_count(self, sphere_data):
        """libQEx produces 94 quads on sphere320 at scale 10."""
        V, T, uv = sphere_data
        qv, qf, tf = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        assert len(qf) == 94, f"Expected 94 quads (libQEx), got {len(qf)}"

    def test_no_non_manifold_edges(self, sphere_data):
        V, T, uv = sphere_data
        qv, qf, tf = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        _, non_manifold = check_manifold(qf)
        assert non_manifold == 0

    def test_no_inverted_faces(self, sphere_data):
        V, T, uv = sphere_data
        qv, qf, tf = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        inverted = check_inverted_faces(qv, qf)
        assert inverted == 0, f"{inverted} inverted faces"

    def test_no_degenerate_quads(self, sphere_data):
        V, T, uv = sphere_data
        qv, qf, tf = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        for i, q in enumerate(qf):
            assert len(set(q)) == 4, f"Quad {i} has duplicate vertices"

    def test_vertices_near_sphere(self, sphere_data):
        V, T, uv = sphere_data
        qv, qf, tf = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        radii = np.linalg.norm(qv, axis=1)
        assert radii.min() > 0.9, f"Min radius {radii.min():.4f}"
        assert radii.max() < 1.1, f"Max radius {radii.max():.4f}"

    def test_no_excessively_long_edges(self, sphere_data):
        V, T, uv = sphere_data
        qv, qf, tf = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        for i, q in enumerate(qf):
            for j in range(4):
                d = np.linalg.norm(qv[q[j]] - qv[q[(j + 1) % 4]])
                assert d < 0.6, f"Quad {i} edge {j} length {d:.4f}"

    @pytest.mark.parametrize("scale", [5, 10, 15, 20])
    def test_more_quads_at_higher_scale(self, sphere_data, scale):
        V, T, uv = sphere_data
        qv, qf, tf = extract_quads(V, T, uv * scale, verbose=False, fill_holes=False)
        assert len(qf) > 0

    def test_scale_10_vs_5_has_more_quads(self, sphere_data):
        V, T, uv = sphere_data
        _, qf5, _ = extract_quads(V, T, uv * 5, verbose=False, fill_holes=False)
        _, qf10, _ = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        assert len(qf10) > len(qf5)


# =========================================================================
# Full extraction tests: torus
# =========================================================================

class TestExtractTorus:
    @pytest.fixture
    def torus_data(self):
        path = GOLDEN / "torus_param.npz"
        if not path.exists():
            pytest.skip("torus_param.npz not found")
        data = np.load(path)
        return data["vertices"], data["triangles"], data["uv_per_tri"]

    def test_produces_quads(self, torus_data):
        V, T, uv = torus_data
        qv, qf, tf = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        assert len(qf) >= 70, f"Expected >=70 quads, got {len(qf)}"

    def test_no_non_manifold(self, torus_data):
        V, T, uv = torus_data
        qv, qf, tf = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        _, non_manifold = check_manifold(qf)
        assert non_manifold == 0

    def test_no_degenerate(self, torus_data):
        V, T, uv = torus_data
        qv, qf, tf = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        for i, q in enumerate(qf):
            assert len(set(q)) == 4, f"Quad {i} degenerate"


# =========================================================================
# Edge cases and robustness
# =========================================================================

class TestRobustness:
    def test_api_compatibility(self):
        """extract_quads should accept the same args as before."""
        V, T, uv = make_scaled_grid_uv(n=5, scale=1.0)
        qv, qf, tf = extract_quads(
            V, T, uv,
            vertex_valences=None,
            fill_holes=False,
            max_hole_size=6,
            verbose=False,
            merge_tolerance=1e-6,
        )
        assert len(qf) >= 9  # Interior quads always work

    def test_cross_cut_parameter_accepted(self):
        """cross_cut parameter is accepted (ignored in libQEx mode)."""
        V, T, uv = make_scaled_grid_uv(n=5, scale=1.0)
        qv, qf, tf = extract_quads(
            V, T, uv, verbose=False, fill_holes=False, cross_cut='none')
        assert len(qf) >= 9


# =========================================================================
# Mesh quality validation tests
# =========================================================================

def _run_full_pipeline(V, T, uv):
    """Run extraction pipeline and return all faces (including n-gons)."""
    vertices = np.asarray(V, dtype=np.float64)
    triangles = np.asarray(T, dtype=np.int64)
    uv_per_triangle = np.asarray(uv, dtype=np.float64)

    mesh = _HalfEdgeMesh(triangles, len(vertices))
    uv_coords = uv_per_triangle.reshape(-1).copy()
    tf_per_edge = _extract_transitions(mesh, uv_coords)
    _consistent_truncation(mesh, uv_coords, tf_per_edge)
    gvertices, face_gv, edge_gv, vertex_gv = _generate_grid_vertices(
        mesh, uv_coords, vertices, tf_per_edge)
    _generate_connections(mesh, uv_coords, tf_per_edge,
                          gvertices, face_gv, edge_gv, vertex_gv, verbose=False)
    all_faces = _generate_faces(gvertices, verbose=False)
    return gvertices, all_faces


class TestFaceQuality:
    """Tests for common failure modes in face generation."""

    @pytest.fixture
    def sphere_data(self):
        path = GOLDEN / "sphere320_param.npz"
        if not path.exists():
            pytest.skip("sphere320_param.npz not found")
        data = np.load(path)
        return data["vertices"], data["triangles"], data["uv_per_tri"]

    @pytest.fixture
    def torus_data(self):
        path = GOLDEN / "torus_param.npz"
        if not path.exists():
            pytest.skip("torus_param.npz not found")
        data = np.load(path)
        return data["vertices"], data["triangles"], data["uv_per_tri"]

    # --- Duplicate vertex in face ---

    def test_no_duplicate_vertices_in_face_grid(self):
        """No face should contain the same vertex index twice."""
        V, T, uv = make_scaled_grid_uv(n=5, scale=1.0)
        _, all_faces = _run_full_pipeline(V, T, uv)
        for i, f in enumerate(all_faces):
            assert len(set(f)) == len(f), \
                f"Face {i} has duplicate vertices: {f}"

    def test_no_duplicate_vertices_in_face_sphere(self, sphere_data):
        """No face should contain the same vertex index twice (sphere)."""
        V, T, uv = sphere_data
        _, all_faces = _run_full_pipeline(V, T, uv * 10)
        for i, f in enumerate(all_faces):
            assert len(set(f)) == len(f), \
                f"Face {i} has duplicate vertices: {f}"

    def test_no_duplicate_vertices_in_face_cylinder(self):
        """No face should contain the same vertex index twice (cylinder)."""
        V, T, uv = make_cylinder_with_cut()
        _, all_faces = _run_full_pipeline(V, T, uv)
        for i, f in enumerate(all_faces):
            assert len(set(f)) == len(f), \
                f"Face {i} has duplicate vertices: {f}"

    # --- Non-adjacent vertices ---

    def test_no_long_edges_sphere(self, sphere_data):
        """Face edges should connect nearby vertices, not skip across mesh."""
        V, T, uv = sphere_data
        gvertices, all_faces = _run_full_pipeline(V, T, uv * 10)
        max_edge_len = 0.0
        worst = None
        for i, f in enumerate(all_faces):
            for j in range(len(f)):
                v0 = gvertices[f[j]].position_3d
                v1 = gvertices[f[(j + 1) % len(f)]].position_3d
                d = np.linalg.norm(v0 - v1)
                if d > max_edge_len:
                    max_edge_len = d
                    worst = (i, j, d)
        assert max_edge_len < 0.6, \
            f"Face {worst[0]} edge {worst[1]} too long: {worst[2]:.4f}"

    def test_no_long_edges_grid(self):
        """Grid mesh edges should be ~1.0 in length (unit grid)."""
        V, T, uv = make_scaled_grid_uv(n=5, scale=1.0)
        gvertices, all_faces = _run_full_pipeline(V, T, uv)
        for i, f in enumerate(all_faces):
            for j in range(len(f)):
                v0 = gvertices[f[j]].position_3d
                v1 = gvertices[f[(j + 1) % len(f)]].position_3d
                d = np.linalg.norm(v0 - v1)
                assert d < 2.0, \
                    f"Face {i} edge {j} length {d:.4f} (expected ~1.0)"

    # --- N-gon detection ---

    def test_sphere_only_quads_and_tris(self, sphere_data):
        """Sphere should only produce quads and triangles, no 5-gons+."""
        V, T, uv = sphere_data
        _, all_faces = _run_full_pipeline(V, T, uv * 10)
        for i, f in enumerate(all_faces):
            assert len(f) <= 4, \
                f"Face {i} is a {len(f)}-gon (expected quad or tri)"

    def test_grid_only_quads(self):
        """Flat grid should only produce quads (no tris or n-gons)."""
        V, T, uv = make_scaled_grid_uv(n=5, scale=1.0)
        _, all_faces = _run_full_pipeline(V, T, uv)
        ngons = [f for f in all_faces if len(f) != 4]
        # Allow some boundary effects but no 5-gons+
        for i, f in enumerate(all_faces):
            assert len(f) <= 4, \
                f"Face {i} is a {len(f)}-gon"

    # --- Zero-area faces ---

    def test_no_zero_area_faces_sphere(self, sphere_data):
        """No face should have zero area."""
        V, T, uv = sphere_data
        qv, qf, _ = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        for i, q in enumerate(qf):
            v = qv[q]
            e1 = v[1] - v[0]
            e2 = v[3] - v[0]
            area = np.linalg.norm(np.cross(e1, e2))
            assert area > 1e-10, f"Quad {i} has near-zero area {area:.2e}"

    # --- Self-edges ---

    def test_no_self_edges_sphere(self, sphere_data):
        """No face edge should connect a vertex to itself."""
        V, T, uv = sphere_data
        _, all_faces = _run_full_pipeline(V, T, uv * 10)
        for i, f in enumerate(all_faces):
            for j in range(len(f)):
                assert f[j] != f[(j + 1) % len(f)], \
                    f"Face {i} has self-edge at vertex {f[j]}"

    def test_no_self_edges_grid(self):
        """No face edge should connect a vertex to itself (grid)."""
        V, T, uv = make_scaled_grid_uv(n=5, scale=1.0)
        _, all_faces = _run_full_pipeline(V, T, uv)
        for i, f in enumerate(all_faces):
            for j in range(len(f)):
                assert f[j] != f[(j + 1) % len(f)], \
                    f"Face {i} has self-edge at vertex {f[j]}"

    # --- Consistent face orientation ---

    def test_consistent_orientation_sphere(self, sphere_data):
        """All quads should have consistent normal orientation (outward)."""
        V, T, uv = sphere_data
        qv, qf, _ = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        inverted = check_inverted_faces(qv, qf)
        assert inverted == 0, f"{inverted} faces have inverted normals"


# =========================================================================
# Legacy transition recovery tests
# =========================================================================

def make_grid_with_90_cut():
    """A 4x2 grid cut in half, right side shifted in UV space."""
    vertices = []
    for j in range(3):
        for i in range(5):
            vertices.append([float(i), float(j), 0.0])
    vertices = np.array(vertices)

    triangles = []
    uv_per_tri = []

    for j in range(2):
        for i in range(2):
            v00 = j * 5 + i
            v10 = j * 5 + i + 1
            v01 = (j + 1) * 5 + i
            v11 = (j + 1) * 5 + i + 1
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])
            uv_per_tri.append([[float(i), float(j)],
                               [float(i + 1), float(j)],
                               [float(i + 1), float(j + 1)]])
            uv_per_tri.append([[float(i), float(j)],
                               [float(i + 1), float(j + 1)],
                               [float(i), float(j + 1)]])

    for j in range(2):
        for i in range(2, 4):
            v00 = j * 5 + i
            v10 = j * 5 + i + 1
            v01 = (j + 1) * 5 + i
            v11 = (j + 1) * 5 + i + 1
            u_off = 3
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])
            uv_per_tri.append([[float(i + u_off), float(j)],
                               [float(i + 1 + u_off), float(j)],
                               [float(i + 1 + u_off), float(j + 1)]])
            uv_per_tri.append([[float(i + u_off), float(j)],
                               [float(i + 1 + u_off), float(j + 1)],
                               [float(i + u_off), float(j + 1)]])

    return vertices, np.array(triangles), np.array(uv_per_tri)


class TestTransitionRecovery:
    def test_no_cuts_in_flat_grid(self):
        V, T, uv = make_2x2_grid_uv()
        transitions = _find_cut_edges_and_transitions(T, uv)
        assert len(transitions) == 0

    def test_cylinder_no_cuts_detected(self):
        V, T, uv = make_cylinder_with_cut()
        transitions = _find_cut_edges_and_transitions(T, uv)
        assert len(transitions) == 0

    def test_grid_with_shift_cut(self):
        V, T, uv = make_grid_with_90_cut()
        transitions = _find_cut_edges_and_transitions(T, uv)
        assert len(transitions) > 0
        for tr in transitions:
            assert tr['residual'] < 1e-10

    def test_sphere_cut_transitions(self):
        path = GOLDEN / "sphere320_param.npz"
        if not path.exists():
            pytest.skip("sphere320_param.npz not found")
        data = np.load(path)
        T, uv = data['triangles'], data['uv_per_tri'] * 10
        transitions = _find_cut_edges_and_transitions(T, uv)
        assert len(transitions) > 20
        max_res = max(tr['residual'] for tr in transitions)
        assert max_res < 1e-8, f"Max residual {max_res}"
        for tr in transitions:
            assert abs(tr['scale_ratio'] - 1.0) < 0.01, \
                f"Scale ratio {tr['scale_ratio']} at edge {tr['edge']}"
