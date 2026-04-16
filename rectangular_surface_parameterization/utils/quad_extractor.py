"""
libQEx-faithful quad mesh extraction from parameterized triangle meshes.

Port of MeshExtractorT.cc from libQEx (Bommes & Ebke, RWTH Aachen, 2013).

Algorithm:
1. Build half-edge data structure from triangle mesh
2. Extract per-edge transition functions from UV coordinates
3. Consistent truncation (make transitions numerically exact)
4. Generate grid vertices at integer UV coordinates (on faces, edges, vertices)
5. For each grid vertex, determine outgoing iso-line directions (local edges)
6. Trace connections along iso-lines through the mesh, crossing cuts via transitions
7. Walk connection graph to construct quad/triangle faces

Same API as before — drop-in replacement.
"""

import numpy as np
import cmath
from math import ceil, floor, pi, acos, sqrt
from collections import defaultdict


# =========================================================================
# Constants
# =========================================================================

_LECI_CONNECTED_THRESH = 0
_LECI_NO_CONNECTION = -1
_LECI_BOUNDARY = -2
_LECI_DEGENERACY = -3
_LECI_ERROR = -4

_GV_ON_VERTEX = 0
_GV_ON_EDGE = 1
_GV_ON_FACE = 2

# Cardinal directions: +u, +v, -u, -v
_DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def _round_qex(x):
    """Round-half-away-from-zero, matching libQEx's ROUND_QME macro."""
    return int(x - 0.5) if x < 0 else int(x + 0.5)


# =========================================================================
# Transition Function (port of TransitionFunctionT<int>)
# =========================================================================

class _TF:
    """Integer transition function: TF(u,v) = R^r * (u,v) + (tu,tv)."""
    __slots__ = ('r', 'tu', 'tv')

    def __init__(self, r=0, tu=0, tv=0):
        self.r = r % 4
        self.tu = tu
        self.tv = tv

    def transform_point(self, u, v):
        r = self.r
        if r == 1:
            u, v = -v, u
        elif r == 2:
            u, v = -u, -v
        elif r == 3:
            u, v = v, -u
        return u + self.tu, v + self.tv

    def transform_vector(self, u, v):
        r = self.r
        if r == 1:
            return -v, u
        elif r == 2:
            return -u, -v
        elif r == 3:
            return v, -u
        return u, v

    def inverse(self):
        r_new = (4 - self.r) % 4
        tu, tv = -self.tu, -self.tv
        if r_new == 1:
            tu, tv = -tv, tu
        elif r_new == 2:
            tu, tv = -tu, -tv
        elif r_new == 3:
            tu, tv = tv, -tu
        return _TF(r_new, _round_qex(tu), _round_qex(tv))

    def __mul__(self, other):
        tu2, tv2 = self.transform_point(other.tu, other.tv)
        return _TF((self.r + other.r) % 4, _round_qex(tu2), _round_qex(tv2))

    def __eq__(self, other):
        return isinstance(other, _TF) and self.r == other.r and self.tu == other.tu and self.tv == other.tv

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f"TF(r={self.r}, t=({self.tu},{self.tv}))"

_TF_IDENTITY = _TF(0, 0, 0)


# =========================================================================
# 2D Geometry Helpers
# =========================================================================

def _orient2d_pts(a, b, c):
    """Orientation of triangle (a,b,c): 1=CCW, -1=CW, 0=collinear."""
    det = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if det > 0:
        return 1
    elif det < 0:
        return -1
    return 0


def _orient2d_vecs(a, b):
    """Orientation of vectors a,b from origin: sign of cross product."""
    det = a[0] * b[1] - a[1] * b[0]
    if det > 0:
        return 1
    elif det < 0:
        return -1
    return 0


def _has_on_collinear(a, b, c):
    """Assuming a,b,c collinear, is c on segment [a,b]?"""
    return (min(a[0], b[0]) <= c[0] + 1e-15 and max(a[0], b[0]) >= c[0] - 1e-15 and
            min(a[1], b[1]) <= c[1] + 1e-15 and max(a[1], b[1]) >= c[1] - 1e-15)


def _seg_has_on(a, b, c):
    """Is point c on segment [a,b]?"""
    if _orient2d_pts(a, b, c) != 0:
        return False
    return _has_on_collinear(a, b, c)


def _seg_intersects(a1, a2, b1, b2):
    """Do segments [a1,a2] and [b1,b2] intersect?"""
    o1a = _orient2d_pts(a1, a2, b1)
    if o1a == 0 and _has_on_collinear(a1, a2, b1):
        return True
    o1b = _orient2d_pts(a1, a2, b2)
    if o1b == 0 and _has_on_collinear(a1, a2, b2):
        return True
    if o1a == o1b:
        return False
    o2a = _orient2d_pts(b1, b2, a1)
    if o2a == 0 and _has_on_collinear(b1, b2, a1):
        return True
    o2b = _orient2d_pts(b1, b2, a2)
    if o2b == 0 and _has_on_collinear(b1, b2, a2):
        return True
    if o2a == o2b:
        return False
    return True


def _tri_boundedness(t0, t1, t2, pt):
    """Is pt on bounded side (1), boundary (0), or unbounded (-1) of triangle?"""
    o_a = _orient2d_pts(t1, t2, pt)
    o_b = _orient2d_pts(t2, t0, pt)
    o_c = _orient2d_pts(t0, t1, pt)
    if o_a == 0 and o_b == 0 and o_c == 0:
        if _has_on_collinear(t0, t1, pt) or _has_on_collinear(t1, t2, pt):
            return 0
        return -1
    if o_a == o_b == o_c:
        return 1
    if o_a == 0 or o_b == 0 or o_c == 0:
        nz = [x for x in (o_a, o_b, o_c) if x != 0]
        if len(set(nz)) == 1:
            return 0
    return -1


def _tri_orientation(t0, t1, t2):
    return _orient2d_pts(t0, t1, t2)


def _ori_to_idx(dx, dy):
    """Map cardinal direction to index 0..3."""
    if dx == 1:
        return 0
    elif dy == 1:
        return 1
    elif dx == -1:
        return 2
    else:
        return 3


def _ori_to_idx_inverse(dx, dy):
    return 3 - _ori_to_idx(dx, dy)


# =========================================================================
# Half-Edge Mesh
# =========================================================================

class _HalfEdgeMesh:
    """Lightweight half-edge data structure for triangle meshes.

    Halfedge 3*fi+k:
      to_vertex   = triangles[fi, k]
      from_vertex  = triangles[fi, (k+2)%3]
      face         = fi
      uv at heh    = uv_per_triangle[fi, k, :]
    """

    def __init__(self, triangles, n_vertices):
        n_faces = len(triangles)
        n_he = 3 * n_faces
        self.n_vertices = n_vertices
        self.n_faces = n_faces
        self.n_halfedges = n_he
        self.triangles = triangles

        # Build directed edge → halfedge map
        directed = {}
        for fi in range(n_faces):
            for k in range(3):
                v_from = int(triangles[fi, (k + 2) % 3])
                v_to = int(triangles[fi, k])
                directed[(v_from, v_to)] = 3 * fi + k

        # Opposite halfedge
        self.opposite = np.full(n_he, -1, dtype=np.int32)
        for (va, vb), heh in directed.items():
            opp = directed.get((vb, va), -1)
            self.opposite[heh] = opp

        # Edge list: (heh0, heh1) where heh1 may be -1 for boundary
        self._edge_of_heh = np.full(n_he, -1, dtype=np.int32)
        edges = []
        for heh in range(n_he):
            if self._edge_of_heh[heh] >= 0:
                continue
            opp = self.opposite[heh]
            eidx = len(edges)
            if opp >= 0 and self._edge_of_heh[opp] >= 0:
                continue
            edges.append((heh, opp))
            self._edge_of_heh[heh] = eidx
            if opp >= 0:
                self._edge_of_heh[opp] = eidx
        self.edges = edges
        self.n_edges = len(edges)

        # Per-vertex: one incoming halfedge (to_vertex = v)
        self._vertex_heh = np.full(n_vertices, -1, dtype=np.int32)
        for fi in range(n_faces):
            for k in range(3):
                v = int(triangles[fi, k])
                self._vertex_heh[v] = 3 * fi + k

        # Edge validity (non-degenerate in UV space)
        self.edge_valid = [True] * self.n_edges

    def to_vertex(self, heh):
        fi, k = divmod(heh, 3)
        return int(self.triangles[fi, k])

    def from_vertex(self, heh):
        fi, k = divmod(heh, 3)
        return int(self.triangles[fi, (k + 2) % 3])

    def face(self, heh):
        return heh // 3

    def next_heh(self, heh):
        fi, k = divmod(heh, 3)
        return 3 * fi + (k + 1) % 3

    def prev_heh(self, heh):
        fi, k = divmod(heh, 3)
        return 3 * fi + (k + 2) % 3

    def edge(self, heh):
        return self._edge_of_heh[heh]

    def is_boundary_heh(self, heh):
        return self.opposite[heh] < 0

    def is_boundary_edge(self, eidx):
        return self.edges[eidx][1] < 0

    def is_boundary_vertex(self, v):
        start = self._vertex_heh[v]
        if start < 0:
            return True
        out = self.opposite[start]
        if out < 0:
            return True
        heh = out
        while True:
            prev_h = self.prev_heh(heh)
            nxt = self.opposite[prev_h]
            if nxt < 0:
                return True
            heh = nxt
            if heh == out:
                break
        return False

    def uv_at(self, heh, uv_coords):
        """UV coordinates stored at this halfedge (at to_vertex)."""
        return uv_coords[2 * heh], uv_coords[2 * heh + 1]

    def vih_iter_cw(self, v):
        """Incoming halfedges in CW order (matches libQEx -- iteration).

        For interior vertices: rotation-based walk (exact libQEx behavior).
        For boundary vertices: face-adjacency ordering.
        """
        if not self.is_boundary_vertex(v):
            # Interior vertex: rotation-based CW walk
            start_in = self._vertex_heh[v]
            if start_in < 0:
                return []
            start_out = self.opposite[start_in]
            if start_out < 0:
                return [start_in]

            result = []
            out = start_out
            for _ in range(self.n_faces + 2):
                in_heh = self.opposite[out]
                if in_heh >= 0:
                    result.append(in_heh)
                prev_h = self.prev_heh(out)
                nxt_out = self.opposite[prev_h]
                if nxt_out < 0:
                    break
                out = nxt_out
                if out == start_out:
                    break
            return result

        # Boundary vertex: use face adjacency ordering
        all_in = []
        for fi in range(self.n_faces):
            for k in range(3):
                if int(self.triangles[fi, k]) == v:
                    all_in.append(3 * fi + k)
        if len(all_in) <= 1:
            return all_in

        from_v_to_heh = {}
        for h in all_in:
            fv = self.from_vertex(h)
            from_v_to_heh[fv] = h

        cw_next = {}
        for h in all_in:
            nxt = self.next_heh(h)
            w = self.to_vertex(nxt)
            if w in from_v_to_heh:
                cw_next[h] = from_v_to_heh[w]

        has_pred = set(cw_next.values())
        starts = [h for h in all_in if h not in has_pred]
        start = starts[0] if starts else all_in[0]

        result = [start]
        visited = {start}
        h = start
        while h in cw_next:
            nxt = cw_next[h]
            if nxt in visited:
                break
            result.append(nxt)
            visited.add(nxt)
            h = nxt
        return result

    def vih_iter_ccw(self, v):
        """Incoming halfedges in CCW order (matches libQEx ++ iteration)."""
        cw = self.vih_iter_cw(v)
        return list(reversed(cw))


# =========================================================================
# Grid Vertex and Local Edge Info
# =========================================================================

class _LocalEdgeInfo:
    __slots__ = ('fh_from', 'uv_from', 'uv_intended_to', 'uv_to',
                 'connected_to_idx', 'orientation_idx',
                 'face_constructed', 'accumulated_tf')

    def __init__(self, fh_from, uv_from, uv_to):
        self.fh_from = fh_from
        self.uv_from = uv_from
        self.uv_intended_to = uv_to
        self.uv_to = None
        self.connected_to_idx = _LECI_NO_CONNECTION
        self.orientation_idx = -1
        self.face_constructed = False
        self.accumulated_tf = _TF_IDENTITY

    def is_connected(self):
        return self.connected_to_idx >= _LECI_CONNECTED_THRESH

    def is_unconnected(self):
        return self.connected_to_idx == _LECI_NO_CONNECTION

    def complete(self, conn_idx, ori_idx, uv_to, tf):
        self.connected_to_idx = conn_idx
        self.orientation_idx = ori_idx
        self.uv_to = uv_to
        self.accumulated_tf = tf


class _GridVertex:
    __slots__ = ('type', 'heh', 'position_uv', 'position_3d',
                 'is_boundary', 'missing_leis', 'local_edges')

    def __init__(self, gv_type, heh, uv, pos_3d, is_boundary=False):
        self.type = gv_type
        self.heh = heh
        self.position_uv = uv
        self.position_3d = pos_3d
        self.is_boundary = is_boundary
        self.missing_leis = 0
        self.local_edges = []

    def n_edges(self):
        return len(self.local_edges)

    def local_edge(self, i):
        n = self.n_edges()
        if n == 0:
            raise IndexError("No local edges")
        i = ((i % n) + n) % n
        return self.local_edges[i]


# =========================================================================
# Transition Extraction (port of extract_transition_functions)
# =========================================================================

def _get_transition(mesh, tf_per_edge, heh):
    """Get transition for crossing via this halfedge: face(heh) → face(opposite(heh))."""
    eidx = mesh.edge(heh)
    if eidx < 0 or mesh.is_boundary_edge(eidx):
        return _TF(0, 0, 0)
    if mesh.edges[eidx][0] == heh:
        return tf_per_edge[eidx]
    else:
        return tf_per_edge[eidx].inverse()


def _extract_transitions(mesh, uv_coords):
    """Extract per-edge transition functions from UV coordinates."""
    tf = [_TF(0, 0, 0)] * mesh.n_edges

    for eidx in range(mesh.n_edges):
        heh0, heh1 = mesh.edges[eidx]
        if heh1 < 0:
            tf[eidx] = _TF(0, 0, 0)
            continue

        heh0p = mesh.prev_heh(heh0)
        heh1p = mesh.prev_heh(heh1)

        l0 = complex(uv_coords[2 * heh0], uv_coords[2 * heh0 + 1])
        l1 = complex(uv_coords[2 * heh0p], uv_coords[2 * heh0p + 1])
        r0 = complex(uv_coords[2 * heh1p], uv_coords[2 * heh1p + 1])
        r1 = complex(uv_coords[2 * heh1], uv_coords[2 * heh1 + 1])

        denom = l0 - l1
        if abs(denom) < 1e-30:
            tf[eidx] = _TF(0, 0, 0)
            continue

        r = _round_qex(2.0 * cmath.log((r0 - r1) / denom).imag / pi)
        r = ((r % 4) + 4) % 4
        t = r0 - (1j ** r) * l0
        tf[eidx] = _TF(r, _round_qex(t.real), _round_qex(t.imag))

    return tf


# =========================================================================
# Consistent Truncation (port of consistent_truncation)
# =========================================================================

def _transition_around_vertex(mesh, tf_per_edge, v):
    """Compute accumulated transition around vertex v."""
    if mesh.is_boundary_vertex(v):
        return _TF(0, 0, 0)

    hehs = mesh.vih_iter_ccw(v)
    if not hehs:
        return _TF(0, 0, 0)

    tf_first = _get_transition(mesh, tf_per_edge, mesh.opposite[hehs[0]])
    result = _TF(0, 0, 0)
    for i in range(1, len(hehs)):
        t = _get_transition(mesh, tf_per_edge, mesh.opposite[hehs[i]])
        result = t * result
    return tf_first * result


def _consistent_truncation(mesh, uv_coords, tf_per_edge):
    """Make UV coordinates exactly consistent with transition functions."""
    for v in range(mesh.n_vertices):
        hehs = mesh.vih_iter_ccw(v)
        if not hehs:
            continue

        # Find max UV magnitude and max transition magnitude
        max_u_abs = 0.0
        max_trans_abs = 0.0
        for heh in hehs:
            if mesh.is_boundary_heh(heh):
                continue
            max_u_abs = max(max_u_abs, abs(uv_coords[2 * heh]))
            max_u_abs = max(max_u_abs, abs(uv_coords[2 * heh + 1]))
            opp = mesh.opposite[heh]
            if opp >= 0 and not mesh.is_boundary_heh(opp):
                eidx = mesh.edge(heh)
                if eidx >= 0:
                    max_trans_abs = max(max_trans_abs, abs(tf_per_edge[eidx].tu))
                    max_trans_abs = max(max_trans_abs, abs(tf_per_edge[eidx].tv))

        # Bit-clearing trick for numerical precision
        max_v = max_u_abs + max_trans_abs + 1
        import math
        max_v = 2.0 ** (math.ceil(math.log2(max_v)) + 1) if max_v > 0 else 2.0

        first_heh = hehs[0]
        if mesh.is_boundary_heh(first_heh):
            # Find first non-boundary incoming halfedge
            for heh in hehs:
                if not mesh.is_boundary_heh(heh):
                    first_heh = heh
                    break

        if mesh.is_boundary_heh(first_heh):
            continue

        # Clear low-order bits
        uv_coords[2 * first_heh] = (uv_coords[2 * first_heh] + max_v) - max_v
        uv_coords[2 * first_heh + 1] = (uv_coords[2 * first_heh + 1] + max_v) - max_v

        # Fix singular vertex UVs
        vtrans = _transition_around_vertex(mesh, tf_per_edge, v)
        if not mesh.is_boundary_vertex(v) and vtrans != _TF_IDENTITY:
            r = vtrans.r
            tu, tv = vtrans.tu, vtrans.tv
            if r == 1:
                uv_coords[2 * first_heh] = (tu - tv) / 2.0
                uv_coords[2 * first_heh + 1] = (tu + tv) / 2.0
            elif r == 2:
                uv_coords[2 * first_heh] = tu / 2.0
                uv_coords[2 * first_heh + 1] = tv / 2.0
            elif r == 3:
                uv_coords[2 * first_heh] = (tu + tv) / 2.0
                uv_coords[2 * first_heh + 1] = (tv - tu) / 2.0

        # Propagate around vertex using transitions
        u_cur = uv_coords[2 * first_heh]
        v_cur = uv_coords[2 * first_heh + 1]

        for i in range(1, len(hehs)):
            heh_cur = hehs[i]
            if mesh.is_boundary_heh(heh_cur):
                continue
            # Apply transition from previous face to current face
            opp = mesh.opposite[heh_cur]
            t = _get_transition(mesh, tf_per_edge, opp)
            u_cur, v_cur = t.transform_point(u_cur, v_cur)
            uv_coords[2 * heh_cur] = u_cur
            uv_coords[2 * heh_cur + 1] = v_cur


# =========================================================================
# Grid Vertex Generation (port of generate_vertices)
# =========================================================================

def _barycentric_interp_3d(mesh, vertices_3d, heh, uv_coords, x, y):
    """Compute 3D position for UV point (x,y) inside the triangle of heh."""
    fi = mesh.face(heh)
    heh0 = 3 * fi
    heh1 = 3 * fi + 1
    heh2 = 3 * fi + 2

    p0 = (uv_coords[2 * heh0], uv_coords[2 * heh0 + 1])
    p1 = (uv_coords[2 * heh1], uv_coords[2 * heh1 + 1])
    p2 = (uv_coords[2 * heh2], uv_coords[2 * heh2 + 1])

    # Barycentric coordinates
    v0 = (p1[0] - p0[0], p1[1] - p0[1])
    v1 = (p2[0] - p0[0], p2[1] - p0[1])
    v2 = (x - p0[0], y - p0[1])

    d00 = v0[0] * v0[0] + v0[1] * v0[1]
    d01 = v0[0] * v1[0] + v0[1] * v1[1]
    d02 = v0[0] * v2[0] + v0[1] * v2[1]
    d11 = v1[0] * v1[0] + v1[1] * v1[1]
    d12 = v1[0] * v2[0] + v1[1] * v2[1]

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-30:
        return vertices_3d[mesh.to_vertex(heh0)]

    inv = 1.0 / denom
    u = (d11 * d02 - d01 * d12) * inv
    v = (d00 * d12 - d01 * d02) * inv

    pp0 = vertices_3d[mesh.to_vertex(heh0)]
    pp1 = vertices_3d[mesh.to_vertex(heh1)]
    pp2 = vertices_3d[mesh.to_vertex(heh2)]

    return (1 - u - v) * pp0 + u * pp1 + v * pp2


def _generate_grid_vertices(mesh, uv_coords, vertices_3d, tf_per_edge):
    """Generate all grid vertices: on faces, edges, and mesh vertices."""
    gvertices = []
    face_gv = [[] for _ in range(mesh.n_faces)]
    edge_gv = [[] for _ in range(mesh.n_edges)]
    vertex_gv = [[] for _ in range(mesh.n_vertices)]

    # ----- Face grid vertices -----
    for fi in range(mesh.n_faces):
        heh0 = 3 * fi
        heh1 = 3 * fi + 1
        heh2 = 3 * fi + 2

        p0 = (uv_coords[2 * heh0], uv_coords[2 * heh0 + 1])
        p1 = (uv_coords[2 * heh1], uv_coords[2 * heh1 + 1])
        p2 = (uv_coords[2 * heh2], uv_coords[2 * heh2 + 1])

        tri_ori = _tri_orientation(p0, p1, p2)
        if tri_ori == 0:
            continue

        xs = [p0[0], p1[0], p2[0]]
        ys = [p0[1], p1[1], p2[1]]
        x_min = int(ceil(min(xs)))
        x_max = int(floor(max(xs)))
        y_min = int(ceil(min(ys)))
        y_max = int(floor(max(ys)))

        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                pt = (float(x), float(y))
                bs = _tri_boundedness(p0, p1, p2, pt)
                if bs == 1:  # strictly inside
                    pos_3d = _barycentric_interp_3d(
                        mesh, vertices_3d, heh0, uv_coords, x, y)
                    gv = _GridVertex(_GV_ON_FACE, heh0, pt, pos_3d, False)
                    _construct_lei_face(gv, fi, uv_coords, tri_ori)
                    face_gv[fi].append(len(gvertices))
                    gvertices.append(gv)

    # ----- Edge grid vertices -----
    for eidx in range(mesh.n_edges):
        heh0, heh1 = mesh.edges[eidx]

        # Use the non-boundary halfedge
        ref_heh = heh0
        if mesh.is_boundary_heh(heh0):
            if heh1 >= 0 and not mesh.is_boundary_heh(heh1):
                ref_heh = heh1
            else:
                mesh.edge_valid[eidx] = False
                continue

        # Edge UV endpoints: the two vertices of the edge
        # ref_heh points to to_vertex, prev(ref_heh) points to from_vertex
        prev_h = mesh.prev_heh(ref_heh)
        p0 = (uv_coords[2 * ref_heh], uv_coords[2 * ref_heh + 1])
        p1 = (uv_coords[2 * prev_h], uv_coords[2 * prev_h + 1])

        if abs(p0[0] - p1[0]) < 1e-15 and abs(p0[1] - p1[1]) < 1e-15:
            mesh.edge_valid[eidx] = False
            continue

        pp0 = vertices_3d[mesh.to_vertex(ref_heh)]
        pp1 = vertices_3d[mesh.to_vertex(prev_h)]

        bb_xmin = min(p0[0], p1[0])
        bb_xmax = max(p0[0], p1[0])
        bb_ymin = min(p0[1], p1[1])
        bb_ymax = max(p0[1], p1[1])

        if bb_xmax - bb_xmin >= bb_ymax - bb_ymin:
            x_min = int(ceil(bb_xmin))
            x_max = int(floor(bb_xmax))
            if float(x_min) == bb_xmin:
                x_min += 1
            if float(x_max) == bb_xmax:
                x_max -= 1
            for x in range(x_min, x_max + 1):
                if abs(p1[0] - p0[0]) < 1e-15:
                    continue
                alpha = (x - p0[0]) / (p1[0] - p0[0])
                y = _round_qex(p0[1] + alpha * (p1[1] - p0[1]))
                if y >= int(ceil(bb_ymin)) and y <= int(floor(bb_ymax)):
                    pt = (float(x), float(y))
                    if _seg_has_on(p0, p1, pt):
                        pos_3d = (1 - alpha) * pp0 + alpha * pp1
                        gv = _GridVertex(_GV_ON_EDGE, ref_heh, pt, pos_3d, False)
                        if mesh.is_boundary_edge(eidx):
                            gv.is_boundary = True
                        _construct_lei_edge(gv, mesh, uv_coords, tf_per_edge)
                        edge_gv[eidx].append(len(gvertices))
                        gvertices.append(gv)
        else:
            y_min = int(ceil(bb_ymin))
            y_max = int(floor(bb_ymax))
            if float(y_min) == bb_ymin:
                y_min += 1
            if float(y_max) == bb_ymax:
                y_max -= 1
            for y in range(y_min, y_max + 1):
                if abs(p1[1] - p0[1]) < 1e-15:
                    continue
                alpha = (y - p0[1]) / (p1[1] - p0[1])
                x = _round_qex(p0[0] + alpha * (p1[0] - p0[0]))
                if x >= int(ceil(bb_xmin)) and x <= int(floor(bb_xmax)):
                    pt = (float(x), float(y))
                    if _seg_has_on(p0, p1, pt):
                        pos_3d = (1 - alpha) * pp0 + alpha * pp1
                        gv = _GridVertex(_GV_ON_EDGE, ref_heh, pt, pos_3d, False)
                        if mesh.is_boundary_edge(eidx):
                            gv.is_boundary = True
                        _construct_lei_edge(gv, mesh, uv_coords, tf_per_edge)
                        edge_gv[eidx].append(len(gvertices))
                        gvertices.append(gv)

    # ----- Vertex grid vertices -----
    for v in range(mesh.n_vertices):
        # Find any incoming halfedge to this vertex (prefer non-boundary)
        heh = mesh._vertex_heh[v]
        if heh < 0:
            continue
        # Try to find a non-boundary halfedge (better for LEI construction)
        if mesh.is_boundary_heh(heh):
            for fi in range(mesh.n_faces):
                for k in range(3):
                    if int(mesh.triangles[fi, k]) == v:
                        candidate = 3 * fi + k
                        if not mesh.is_boundary_heh(candidate):
                            heh = candidate
                            break
                else:
                    continue
                break

        u, vv = uv_coords[2 * heh], uv_coords[2 * heh + 1]
        if abs(u - _round_qex(u)) < 1e-10 and abs(vv - _round_qex(vv)) < 1e-10:
            pt = (float(_round_qex(u)), float(_round_qex(vv)))
            pos_3d = vertices_3d[v]
            gv = _GridVertex(_GV_ON_VERTEX, heh, pt, pos_3d, False)
            if mesh.is_boundary_vertex(v):
                gv.is_boundary = True
            _construct_lei_vertex(gv, v, mesh, uv_coords, tf_per_edge)
            vertex_gv[v].append(len(gvertices))
            gvertices.append(gv)

    return gvertices, face_gv, edge_gv, vertex_gv


# =========================================================================
# Local Edge Construction
# =========================================================================

def _construct_lei_face(gv, fi, uv_coords, tri_ori):
    """Construct local edge info for a grid vertex strictly inside a face."""
    gv.local_edges = []
    uv = gv.position_uv
    for dx, dy in _DIRECTIONS:
        to_uv = (uv[0] + dx, uv[1] + dy)
        gv.local_edges.append(_LocalEdgeInfo(fi, uv, to_uv))
    if tri_ori == -1:
        gv.local_edges.reverse()


def _construct_lei_edge(gv, mesh, uv_coords, tf_per_edge):
    """Construct local edge info for a grid vertex on an edge."""
    gv.local_edges = []
    heh = gv.heh
    heh_opp = mesh.opposite[heh]

    if mesh.is_boundary_heh(heh):
        return
    if mesh.is_boundary_edge(mesh.edge(heh)):
        gv.is_boundary = True

    fi = mesh.face(heh)
    ori = _face_uv_orientation(fi, uv_coords)

    fi_opp = -1
    ori_opp = 0
    if heh_opp >= 0 and not mesh.is_boundary_heh(heh_opp):
        fi_opp = mesh.face(heh_opp)
        ori_opp = _face_uv_orientation(fi_opp, uv_coords)

    uv = gv.position_uv

    # Get edge direction in face(heh)
    prev_h = mesh.prev_heh(heh)
    p1 = (uv_coords[2 * heh], uv_coords[2 * heh + 1])
    p0 = (uv_coords[2 * prev_h], uv_coords[2 * prev_h + 1])

    # Compute transformed UV in opposite face
    tf = _get_transition(mesh, tf_per_edge, heh)
    uv_opp_u, uv_opp_v = tf.transform_point(uv[0], uv[1])
    uv_opp = (uv_opp_u, uv_opp_v)

    # Add directions in face one
    leis_face1 = []
    middle_el = 0
    for dx, dy in _DIRECTIONS:
        to_uv = (uv[0] + dx, uv[1] + dy)
        path_ori = _orient2d_pts(p0, p1, to_uv)
        if path_ori == ori:
            leis_face1.append(_LocalEdgeInfo(fi, uv, to_uv))
        elif path_ori == 0:
            edge_dir = (p1[0] - p0[0], p1[1] - p0[1])
            dir_vec = (float(dx), float(dy))
            dot = edge_dir[0] * dir_vec[0] + edge_dir[1] * dir_vec[1]
            if dot > 0 or fi_opp < 0:
                leis_face1.append(_LocalEdgeInfo(fi, uv, to_uv))
            else:
                middle_el = len(leis_face1)
        else:
            middle_el = len(leis_face1)

    if middle_el > 0 and middle_el < len(leis_face1):
        leis_face1 = leis_face1[middle_el:] + leis_face1[:middle_el]
    if ori == -1:
        leis_face1.reverse()
    gv.local_edges.extend(leis_face1)

    # Add directions in face two (opposite face)
    if fi_opp >= 0:
        prev_opp = mesh.prev_heh(heh_opp)
        p1_opp = (uv_coords[2 * heh_opp], uv_coords[2 * heh_opp + 1])
        p0_opp = (uv_coords[2 * prev_opp], uv_coords[2 * prev_opp + 1])

        le_ofs = len(gv.local_edges)
        leis_face2 = []
        middle_el = 0
        for dx, dy in _DIRECTIONS:
            to_uv_local = (uv[0] + dx, uv[1] + dy)
            to_u, to_v = tf.transform_point(to_uv_local[0], to_uv_local[1])
            to_uv_opp = (to_u, to_v)
            trans_dir = (to_u - uv_opp[0], to_v - uv_opp[1])

            path_ori = _orient2d_pts(p0_opp, p1_opp, to_uv_opp)
            if path_ori == ori_opp:
                leis_face2.append(_LocalEdgeInfo(fi_opp, uv_opp, to_uv_opp))
            elif path_ori == 0:
                edge_dir = (p1_opp[0] - p0_opp[0], p1_opp[1] - p0_opp[1])
                dot = edge_dir[0] * trans_dir[0] + edge_dir[1] * trans_dir[1]
                if dot > 0:
                    leis_face2.append(_LocalEdgeInfo(fi_opp, uv_opp, to_uv_opp))
                else:
                    middle_el = len(leis_face2)
            else:
                middle_el = len(leis_face2)

        if middle_el > 0 and middle_el < len(leis_face2):
            leis_face2 = leis_face2[middle_el:] + leis_face2[:middle_el]
        if ori_opp == -1:
            leis_face2.reverse()
        gv.local_edges.extend(leis_face2)


def _construct_lei_vertex(gv, v, mesh, uv_coords, tf_per_edge):
    """Construct local edge info for a grid vertex at a mesh vertex."""
    gv.local_edges = []

    if gv.heh < 0:
        return

    if mesh.is_boundary_vertex(v):
        gv.is_boundary = True

    # Iterate faces around vertex (CW, matching libQEx --)
    hehs = mesh.vih_iter_cw(v)

    pos_angle_sum = 0.0
    neg_angle_sum = 0.0
    initial_neg_angle_sum = 0.0

    for heh in hehs:
        if mesh.is_boundary_heh(heh):
            continue

        heh1 = mesh.next_heh(heh)
        heh2 = mesh.next_heh(heh1)
        uv0 = (uv_coords[2 * heh], uv_coords[2 * heh + 1])
        uv1 = (uv_coords[2 * heh1], uv_coords[2 * heh1 + 1])
        uv2 = (uv_coords[2 * heh2], uv_coords[2 * heh2 + 1])

        sector_left = (uv2[0] - uv0[0], uv2[1] - uv0[1])
        sector_right = (uv1[0] - uv0[0], uv1[1] - uv0[1])
        orientation = _tri_orientation(uv0, uv1, uv2)

        # Accumulate angle sum for valence computation
        sl_norm = sqrt(sector_left[0] ** 2 + sector_left[1] ** 2)
        sr_norm = sqrt(sector_right[0] ** 2 + sector_right[1] ** 2)
        if sl_norm > 1e-15 and sr_norm > 1e-15:
            dot = (sector_left[0] * sector_right[0] + sector_left[1] * sector_right[1])
            cos_val = max(-1.0, min(1.0, dot / (sl_norm * sr_norm)))
            angle = acos(cos_val)
        else:
            angle = 0.0

        if orientation == 1:  # CCW (positive)
            if neg_angle_sum > 0:
                pos_angle_sum += 2 * pi - neg_angle_sum
                neg_angle_sum = 0
            pos_angle_sum += angle
        elif orientation == -1:  # CW (negative)
            if pos_angle_sum == 0:
                initial_neg_angle_sum += angle
            else:
                neg_angle_sum += angle

        # Check which cardinal directions fall inside this face sector
        fi = mesh.face(heh)
        is_left_boundary = mesh.is_boundary_heh(mesh.opposite[heh]) if mesh.opposite[heh] >= 0 else True

        leis_per_face = []
        middle_el = 0
        for i, (dx, dy) in enumerate(_DIRECTIONS):
            d = (float(dx), float(dy))
            ori1 = _orient2d_vecs(sector_right, d)
            ori2 = _orient2d_vecs(d, sector_left)

            accepted = False
            if is_left_boundary and ori2 == 0:
                dot_left = sector_left[0] * d[0] + sector_left[1] * d[1]
                if dot_left > 0:
                    accepted = True
            if not accepted and ori1 == 0:
                dot_right = sector_right[0] * d[0] + sector_right[1] * d[1]
                if dot_right > 0:
                    accepted = True
            if not accepted and ori1 == orientation and ori2 == orientation:
                accepted = True

            if accepted:
                leis_per_face.append(
                    _LocalEdgeInfo(fi, uv0, (uv0[0] + dx, uv0[1] + dy)))
            else:
                middle_el = len(leis_per_face)

        if middle_el > 0 and middle_el < len(leis_per_face):
            leis_per_face = leis_per_face[middle_el:] + leis_per_face[:middle_el]
        if orientation == -1:
            leis_per_face.reverse()
        gv.local_edges.extend(leis_per_face)

    # Handle negative angle runs
    if initial_neg_angle_sum > 0 or neg_angle_sum > 0:
        neg_angle_sum += initial_neg_angle_sum
        pos_angle_sum += 2 * pi - neg_angle_sum

    ninety_jump = pos_angle_sum / (pi / 2) if pos_angle_sum > 0 else 4
    expected = _round_qex(ninety_jump)
    gv.missing_leis = expected - len(gv.local_edges)
    if gv.is_boundary:
        gv.missing_leis = 0


def _face_uv_orientation(fi, uv_coords):
    heh0 = 3 * fi
    heh1 = 3 * fi + 1
    heh2 = 3 * fi + 2
    p0 = (uv_coords[2 * heh0], uv_coords[2 * heh0 + 1])
    p1 = (uv_coords[2 * heh1], uv_coords[2 * heh1 + 1])
    p2 = (uv_coords[2 * heh2], uv_coords[2 * heh2 + 1])
    return _tri_orientation(p0, p1, p2)


# =========================================================================
# Intra-GridVertex Transition (for vertices shared by multiple faces)
# =========================================================================

def _intra_gv_transition(mesh, tf_per_edge, from_fh, to_fh, gv, return_identity_if_same):
    """Compute transition between two faces sharing a grid vertex."""
    if return_identity_if_same and from_fh == to_fh:
        return _TF_IDENTITY

    if gv.type == _GV_ON_FACE:
        return _TF_IDENTITY

    if gv.type == _GV_ON_EDGE:
        heh = gv.heh
        opp = mesh.opposite[heh]
        fh_heh = mesh.face(heh)
        if fh_heh == from_fh:
            t1 = _get_transition(mesh, tf_per_edge, heh)
            if from_fh == to_fh:
                t2 = _TF_IDENTITY
                if opp >= 0:
                    t2 = _get_transition(mesh, tf_per_edge, opp)
                return t2 * t1
            return t1
        else:
            if opp >= 0:
                t1 = _get_transition(mesh, tf_per_edge, opp)
                if from_fh == to_fh:
                    t2 = _get_transition(mesh, tf_per_edge, heh)
                    return t2 * t1
                return t1
        return _TF_IDENTITY

    if gv.type == _GV_ON_VERTEX:
        v = mesh.to_vertex(gv.heh)
        result = _TF_IDENTITY

        # Walk around vertex from from_fh to to_fh
        hehs = mesh.vih_iter_ccw(v)
        # Find starting position
        start_idx = -1
        for i, h in enumerate(hehs):
            if not mesh.is_boundary_heh(h) and mesh.face(h) == from_fh:
                start_idx = i
                break
        if start_idx < 0:
            return _TF_IDENTITY

        idx = start_idx
        while True:
            h = hehs[idx]
            fi = mesh.face(h)
            if fi == to_fh and idx != start_idx:
                break
            # Transition across next edge
            nxt = mesh.next_heh(h)
            t = _get_transition(mesh, tf_per_edge, nxt)
            result = t * result
            idx = (idx + 1) % len(hehs)
            if idx == start_idx:
                break

        return result

    return _TF_IDENTITY


# =========================================================================
# Path Tracing (port of find_path + find_local_connection)
# =========================================================================

def _find_local_connection(uv_from, uv_to, fi, mesh, uv_coords, tf_per_edge,
                           gvertices, face_gv, edge_gv, vertex_gv,
                           accumulated_tf, heh0, heh1, heh2, bs):
    """Find a grid vertex connection within the current triangle."""
    p0 = (uv_coords[2 * heh0], uv_coords[2 * heh0 + 1])
    p1 = (uv_coords[2 * heh1], uv_coords[2 * heh1 + 1])
    p2 = (uv_coords[2 * heh2], uv_coords[2 * heh2 + 1])

    tri_ori = _tri_orientation(p0, p1, p2)
    if tri_ori == 0:
        return _LECI_DEGENERACY, 0, accumulated_tf

    # Strictly inside face
    if bs == 1:
        face_ori = tri_ori
        dx = uv_from[0] - uv_to[0]
        dy = uv_from[1] - uv_to[1]
        # Determine reverse direction index
        rdx, rdy = _round_qex(dx), _round_qex(dy)
        if face_ori == -1:
            ori_idx = _ori_to_idx_inverse(rdx, rdy)
        else:
            ori_idx = _ori_to_idx(rdx, rdy)

        for gvidx in face_gv[fi]:
            gv = gvertices[gvidx]
            if ori_idx < len(gv.local_edges):
                lei = gv.local_edges[ori_idx]
                uv_from_f = (float(_round_qex(uv_from[0])), float(_round_qex(uv_from[1])))
                uv_to_f = (float(_round_qex(uv_to[0])), float(_round_qex(uv_to[1])))
                lei_to = lei.uv_intended_to
                lei_from = lei.uv_from
                if (abs(lei_to[0] - uv_from_f[0]) < 0.5 and abs(lei_to[1] - uv_from_f[1]) < 0.5 and
                        abs(lei_from[0] - uv_to_f[0]) < 0.5 and abs(lei_from[1] - uv_to_f[1]) < 0.5):
                    return gvidx, ori_idx, accumulated_tf
        return _LECI_ERROR, 0, accumulated_tf

    # On boundary: check vertices then edges
    for i, (th, tri_pt) in enumerate([(heh0, p0), (heh1, p1), (heh2, p2)]):
        if abs(uv_to[0] - tri_pt[0]) < 1e-10 and abs(uv_to[1] - tri_pt[1]) < 1e-10:
            return _find_local_at_vertex(
                uv_from, uv_to, th, mesh, uv_coords, tf_per_edge,
                gvertices, vertex_gv, accumulated_tf, p0, p1, p2)

    segs = [(p2, p0, heh0), (p0, p1, heh1), (p1, p2, heh2)]
    for s0, s1, sh in segs:
        if _seg_has_on(s0, s1, uv_to):
            return _find_local_at_edge(
                uv_from, uv_to, sh, mesh, uv_coords, tf_per_edge,
                gvertices, edge_gv, accumulated_tf)

    return _LECI_ERROR, 0, accumulated_tf


def _find_local_at_edge(uv_from, uv_to, heh, mesh, uv_coords, tf_per_edge,
                        gvertices, edge_gv, accumulated_tf):
    """Find connection at an edge grid vertex."""
    eidx = mesh.edge(heh)
    fi = mesh.face(heh)
    heh_opp = mesh.opposite[heh]
    fi_opp = mesh.face(heh_opp) if heh_opp >= 0 and not mesh.is_boundary_heh(heh_opp) else -1

    cross_tf = _get_transition(mesh, tf_per_edge, heh)
    uv_from_opp_u, uv_from_opp_v = cross_tf.transform_point(uv_from[0], uv_from[1])
    uv_to_opp_u, uv_to_opp_v = cross_tf.transform_point(uv_to[0], uv_to[1])
    uv_from_opp = (uv_from_opp_u, uv_from_opp_v)
    uv_to_opp = (uv_to_opp_u, uv_to_opp_v)

    for gvidx in edge_gv[eidx]:
        gv = gvertices[gvidx]
        for j, lei in enumerate(gv.local_edges):
            if ((lei.fh_from == fi and
                 abs(lei.uv_from[0] - uv_to[0]) < 0.5 and abs(lei.uv_from[1] - uv_to[1]) < 0.5 and
                 abs(lei.uv_intended_to[0] - uv_from[0]) < 0.5 and abs(lei.uv_intended_to[1] - uv_from[1]) < 0.5) or
                (lei.fh_from == fi_opp and fi_opp >= 0 and
                 abs(lei.uv_from[0] - uv_to_opp[0]) < 0.5 and abs(lei.uv_from[1] - uv_to_opp[1]) < 0.5 and
                 abs(lei.uv_intended_to[0] - uv_from_opp[0]) < 0.5 and abs(lei.uv_intended_to[1] - uv_from_opp[1]) < 0.5)):

                gv_face = mesh.face(gv.heh)
                if gv_face == fi:
                    return gvidx, j, accumulated_tf
                elif gv_face == fi_opp:
                    return gvidx, j, cross_tf * accumulated_tf
                else:
                    return gvidx, j, accumulated_tf

    return _LECI_ERROR, 0, accumulated_tf


def _find_local_at_vertex(uv_from, uv_to, heh, mesh, uv_coords, tf_per_edge,
                          gvertices, vertex_gv, accumulated_tf, p0, p1, p2):
    """Find connection at a mesh vertex grid vertex."""
    vh = mesh.to_vertex(heh)

    fi = mesh.face(heh)
    candidates = [(fi, uv_from, uv_to, _TF_IDENTITY)]

    # Check neighboring faces if path is collinear with triangle edges
    if _orient2d_pts(uv_from, uv_to, p2) == 0:
        opp = mesh.opposite[heh]
        if opp >= 0 and not mesh.is_boundary_heh(opp):
            tf = _get_transition(mesh, tf_per_edge, heh)
            uf, vf = tf.transform_point(uv_from[0], uv_from[1])
            ut, vt = tf.transform_point(uv_to[0], uv_to[1])
            candidates.append((mesh.face(opp), (uf, vf), (ut, vt), tf))

    if _orient2d_pts(uv_from, uv_to, p1) == 0:
        nheh = mesh.next_heh(heh)
        opp_nheh = mesh.opposite[nheh]
        if opp_nheh >= 0 and not mesh.is_boundary_heh(opp_nheh):
            tf = _get_transition(mesh, tf_per_edge, nheh)
            uf, vf = tf.transform_point(uv_from[0], uv_from[1])
            ut, vt = tf.transform_point(uv_to[0], uv_to[1])
            candidates.append((mesh.face(opp_nheh), (uf, vf), (ut, vt), tf))

    for gvidx in vertex_gv[vh]:
        gv = gvertices[gvidx]
        for j, lei in enumerate(gv.local_edges):
            for c_fh, c_from, c_to, c_tf in candidates:
                if (lei.fh_from == c_fh and
                        abs(lei.uv_from[0] - c_to[0]) < 0.5 and abs(lei.uv_from[1] - c_to[1]) < 0.5 and
                        abs(lei.uv_intended_to[0] - c_from[0]) < 0.5 and abs(lei.uv_intended_to[1] - c_from[1]) < 0.5):
                    intra_tf = _intra_gv_transition(
                        mesh, tf_per_edge, c_fh, mesh.face(gv.heh), gv, True)
                    return gvidx, j, intra_tf * c_tf * accumulated_tf

    return _LECI_ERROR, 0, accumulated_tf


def _find_path(gv, lei, mesh, uv_coords, tf_per_edge,
               gvertices, face_gv, edge_gv, vertex_gv):
    """Trace an iso-line from a grid vertex to find its connection.

    Returns (connected_to_idx, orientation_idx, accumulated_tf).
    """
    cur_fh = lei.fh_from
    uv_from = list(lei.uv_from)
    uv_to = list(lei.uv_intended_to)

    heh0 = 3 * cur_fh
    heh1 = 3 * cur_fh + 1
    heh2 = 3 * cur_fh + 2

    p0 = (uv_coords[2 * heh0], uv_coords[2 * heh0 + 1])
    p1 = (uv_coords[2 * heh1], uv_coords[2 * heh1 + 1])
    p2 = (uv_coords[2 * heh2], uv_coords[2 * heh2 + 1])

    tri_ori = _tri_orientation(p0, p1, p2)
    inverted = (tri_ori == -1)

    accumulated_tf = _TF_IDENTITY

    # Check if target is in the starting triangle
    bs = _tri_boundedness(p0, p1, p2, tuple(uv_to))
    if bs >= 0:
        return _find_local_connection(
            tuple(uv_from), tuple(uv_to), cur_fh, mesh, uv_coords, tf_per_edge,
            gvertices, face_gv, edge_gv, vertex_gv,
            accumulated_tf, heh0, heh1, heh2, bs)

    # Find which edge the path exits through
    cur_heh = -1
    path_from = tuple(uv_from)
    path_to = tuple(uv_to)

    if gv.type == _GV_ON_FACE:
        if _seg_intersects(path_from, path_to, p2, p0):
            cur_heh = heh0
        elif _seg_intersects(path_from, path_to, p0, p1):
            cur_heh = heh1
        elif _seg_intersects(path_from, path_to, p1, p2):
            cur_heh = heh2
        else:
            # Fallback: use orientation to find exit edge
            if tri_ori != 0:
                oe0 = _orient2d_pts(p2, p0, path_to)
                oe1 = _orient2d_pts(p0, p1, path_to)
                oe2 = _orient2d_pts(p1, p2, path_to)
                if oe0 != 0 and oe0 != tri_ori:
                    cur_heh = heh0
                elif oe1 != 0 and oe1 != tri_ori:
                    cur_heh = heh1
                elif oe2 != 0 and oe2 != tri_ori:
                    cur_heh = heh2
                else:
                    return _LECI_ERROR, 0, _TF_IDENTITY
            else:
                return _LECI_ERROR, 0, _TF_IDENTITY

    elif gv.type == _GV_ON_EDGE:
        ref_heh = gv.heh
        if mesh.is_boundary_heh(ref_heh) or mesh.face(ref_heh) != cur_fh:
            ref_heh = mesh.opposite[ref_heh]

        prev_h = mesh.prev_heh(ref_heh)
        next_h = mesh.next_heh(ref_heh)

        p_cur = (uv_coords[2 * ref_heh], uv_coords[2 * ref_heh + 1])
        p_nxt = (uv_coords[2 * next_h], uv_coords[2 * next_h + 1])

        if _seg_intersects(path_from, path_to, p_cur, p_nxt):
            cur_heh = next_h
        else:
            cur_heh = prev_h

    elif gv.type == _GV_ON_VERTEX:
        vh = mesh.to_vertex(gv.heh)
        if mesh.to_vertex(heh0) == vh:
            cur_heh = heh2
        elif mesh.to_vertex(heh1) == vh:
            cur_heh = heh0
        elif mesh.to_vertex(heh2) == vh:
            cur_heh = heh1
        else:
            return _LECI_ERROR, 0, _TF_IDENTITY

    if cur_heh < 0:
        return _LECI_ERROR, 0, _TF_IDENTITY

    # Check edge validity
    eidx = mesh.edge(cur_heh)
    if eidx >= 0 and not mesh.edge_valid[eidx]:
        return _LECI_DEGENERACY, 0, _TF_IDENTITY

    # Cross the first edge
    tf = _get_transition(mesh, tf_per_edge, cur_heh)
    uv_from[0], uv_from[1] = tf.transform_point(uv_from[0], uv_from[1])
    uv_to[0], uv_to[1] = tf.transform_point(uv_to[0], uv_to[1])
    accumulated_tf = tf * accumulated_tf
    cur_heh = mesh.opposite[cur_heh]

    # Main walking loop
    visited_faces = set()
    for _ in range(100000):
        if cur_heh < 0 or mesh.is_boundary_heh(cur_heh):
            return _LECI_BOUNDARY, 0, accumulated_tf

        cur_fh = mesh.face(cur_heh)

        # Cycle detection: if we revisit a face, the path is looping
        if cur_fh in visited_faces:
            return _LECI_ERROR, 0, accumulated_tf
        visited_faces.add(cur_fh)

        heh0 = cur_heh
        heh1 = mesh.next_heh(heh0)
        heh2 = mesh.next_heh(heh1)

        p0 = (uv_coords[2 * heh0], uv_coords[2 * heh0 + 1])
        p1 = (uv_coords[2 * heh1], uv_coords[2 * heh1 + 1])
        p2 = (uv_coords[2 * heh2], uv_coords[2 * heh2 + 1])

        tri_ori = _tri_orientation(p0, p1, p2)

        if tri_ori == 0:
            if p0 == p1 or p1 == p2 or p2 == p0:
                return _LECI_DEGENERACY, 0, accumulated_tf

        # Handle orientation change (inverted triangles)
        currently_inverted = (tri_ori == -1)
        if currently_inverted != inverted:
            inverted = currently_inverted
            uv_from, uv_to = uv_to, uv_from

        # Check if target is inside
        target = tuple(uv_to)
        bs = _tri_boundedness(p0, p1, p2, target)
        if bs >= 0:
            return _find_local_connection(
                tuple(uv_from), target, cur_fh, mesh, uv_coords, tf_per_edge,
                gvertices, face_gv, edge_gv, vertex_gv,
                accumulated_tf, heh0, heh1, heh2, bs)

        # Find exit edge (not the entry edge heh0)
        path_from = tuple(uv_from)
        path_to = target
        s1 = (p0, p1)
        s2 = (p2, p1)

        is1 = _seg_intersects(path_from, path_to, s1[0], s1[1])
        is2 = _seg_intersects(path_from, path_to, s2[0], s2[1])

        heh_upd = -1
        if is1 and not is2:
            heh_upd = heh1
        elif not is1 and is2:
            heh_upd = heh2
        elif is1 and is2:
            vis0 = _seg_has_on(path_from, path_to, p0)
            vis1 = _seg_has_on(path_from, path_to, p1)
            vis2 = _seg_has_on(path_from, path_to, p2)
            if not vis0 and not vis1 and vis2:
                heh_upd = heh1
            elif vis0 and vis2:
                if _orient2d_pts(path_from, path_to, p1) == tri_ori:
                    heh_upd = heh1
                else:
                    heh_upd = heh2
            else:
                heh_upd = heh2
        else:
            # Fallback: segment intersection failed for both edges.
            # This happens when the path passes near vertex p1 or when
            # transition functions place the path outside the triangle.
            # Use orientation of target relative to each edge to decide.
            if tri_ori != 0:
                ori_e1 = _orient2d_pts(p0, p1, path_to)
                ori_e2 = _orient2d_pts(p1, p2, path_to)
                outside_e1 = (ori_e1 != 0 and ori_e1 != tri_ori)
                outside_e2 = (ori_e2 != 0 and ori_e2 != tri_ori)

                if outside_e1 and not outside_e2:
                    heh_upd = heh1
                elif outside_e2 and not outside_e1:
                    heh_upd = heh2
                elif outside_e1 and outside_e2:
                    # Target beyond vertex p1 — pick edge more perpendicular
                    # to path direction (larger cross product = more crossing)
                    e1 = (p1[0] - p0[0], p1[1] - p0[1])
                    e2 = (p2[0] - p1[0], p2[1] - p1[1])
                    pd = (path_to[0] - path_from[0], path_to[1] - path_from[1])
                    cross1 = abs(pd[0] * e1[1] - pd[1] * e1[0])
                    cross2 = abs(pd[0] * e2[1] - pd[1] * e2[0])
                    heh_upd = heh1 if cross1 >= cross2 else heh2
                else:
                    return _LECI_ERROR, 0, accumulated_tf
            else:
                return _LECI_ERROR, 0, accumulated_tf

        if heh_upd < 0:
            return _LECI_ERROR, 0, accumulated_tf

        eidx = mesh.edge(heh_upd)
        if eidx >= 0 and not mesh.edge_valid[eidx]:
            return _LECI_DEGENERACY, 0, accumulated_tf

        tf = _get_transition(mesh, tf_per_edge, heh_upd)
        uv_from[0], uv_from[1] = tf.transform_point(uv_from[0], uv_from[1])
        uv_to[0], uv_to[1] = tf.transform_point(uv_to[0], uv_to[1])
        accumulated_tf = tf * accumulated_tf
        cur_heh = mesh.opposite[heh_upd]

    return _LECI_ERROR, 0, accumulated_tf


# =========================================================================
# Connection Generation (port of generate_connections)
# =========================================================================

def _generate_connections(mesh, uv_coords, tf_per_edge,
                          gvertices, face_gv, edge_gv, vertex_gv, verbose=True):
    """Trace connections between all grid vertices."""
    n_connected = 0
    n_boundary = 0
    n_error = 0
    n_no_conn = 0

    for i, gv in enumerate(gvertices):
        for j, lei in enumerate(gv.local_edges):
            if lei.is_unconnected() and lei.fh_from >= 0:
                conn_idx, ori_idx, acc_tf = _find_path(
                    gv, lei, mesh, uv_coords, tf_per_edge,
                    gvertices, face_gv, edge_gv, vertex_gv)

                lei.connected_to_idx = conn_idx
                lei.orientation_idx = ori_idx
                lei.accumulated_tf = acc_tf

                if conn_idx == _LECI_BOUNDARY:
                    gv.is_boundary = True
                    n_boundary += 1
                elif conn_idx >= _LECI_CONNECTED_THRESH:
                    n_connected += 1
                    # Store reverse connection
                    target = gvertices[conn_idx]
                    if ori_idx < len(target.local_edges):
                        rev_lei = target.local_edges[ori_idx]
                        if rev_lei.connected_to_idx < _LECI_CONNECTED_THRESH:
                            # Compute reverse transition
                            rev_tf = _intra_gv_transition(
                                mesh, tf_per_edge, rev_lei.fh_from,
                                mesh.face(target.heh), target, True).inverse()
                            rev_tf = rev_tf * acc_tf
                            rev_tf = rev_tf * _intra_gv_transition(
                                mesh, tf_per_edge, lei.fh_from,
                                mesh.face(gv.heh), gv, True).inverse()

                            opp_to_u, opp_to_v = rev_tf.inverse().transform_point(
                                gv.position_uv[0], gv.position_uv[1])
                            rev_tf_inv = rev_tf.inverse()

                            rev_lei.complete(i, j, (opp_to_u, opp_to_v), rev_tf_inv)
                        else:
                            # Already connected - conflict
                            lei.connected_to_idx = _LECI_NO_CONNECTION
                            n_connected -= 1
                            n_error += 1
                elif conn_idx == _LECI_NO_CONNECTION:
                    n_no_conn += 1
                else:
                    n_error += 1

    if verbose:
        total_lei = sum(gv.n_edges() for gv in gvertices)
        print(f"  Connections: {n_connected} connected, {n_boundary} boundary, "
              f"{n_error} errors, {n_no_conn} unconnected (of {total_lei} LEIs)")


# =========================================================================
# Face Generation (port of generate_faces_and_store_quadmesh)
# =========================================================================

def _generate_faces(gvertices, verbose=True):
    """Walk the connection graph to construct faces."""
    faces = []

    for i, gv in enumerate(gvertices):
        for j in range(gv.n_edges()):
            lei = gv.local_edges[j]
            if lei.face_constructed:
                continue
            if not lei.is_connected():
                continue

            face_verts = []
            cur_gv_idx = i
            cur_ori_idx = j

            for _ in range(100):
                if cur_gv_idx < _LECI_CONNECTED_THRESH:
                    break

                if cur_gv_idx == i and len(face_verts) > 0:
                    if len(face_verts) >= 3:
                        faces.append(face_verts)
                    break

                cur_gv = gvertices[cur_gv_idx]
                cur_lei = cur_gv.local_edge(cur_ori_idx)

                if cur_lei.face_constructed:
                    break

                face_verts.append(cur_gv_idx)
                cur_lei.face_constructed = True

                next_gv_idx = cur_lei.connected_to_idx
                next_ori_idx = cur_lei.orientation_idx

                cur_gv_idx = next_gv_idx
                cur_ori_idx = next_ori_idx - 1  # Turn right

    if verbose:
        by_size = defaultdict(int)
        for f in faces:
            by_size[len(f)] += 1
        parts = [f"{c} {n}-gons" for n, c in sorted(by_size.items())]
        print(f"  Faces: {len(faces)} total ({', '.join(parts)})")

    return faces


# =========================================================================
# Mesh validation (kept from original)
# =========================================================================

def _validate_mesh(vertices, quads, verbose=True):
    """Check output mesh for basic validity."""
    edge_count = defaultdict(int)
    for q in quads:
        n = len(q)
        for i in range(n):
            e = (min(q[i], q[(i + 1) % n]), max(q[i], q[(i + 1) % n]))
            edge_count[e] += 1

    non_manifold = sum(1 for c in edge_count.values() if c > 2)
    boundary = sum(1 for c in edge_count.values() if c == 1)

    if verbose:
        print(f"  Mesh validation: {len(vertices)} verts, {len(quads)} quads, "
              f"{len(edge_count)} edges, {boundary} boundary, {non_manifold} non-manifold")

    return {
        'n_verts': len(vertices),
        'n_quads': len(quads),
        'n_edges': len(edge_count),
        'boundary_edges': boundary,
        'non_manifold_edges': non_manifold,
    }


# =========================================================================
# Legacy transition recovery (kept for test compatibility)
# =========================================================================

_ROTATIONS = [
    np.array([[1, 0], [0, 1]], dtype=float),
    np.array([[0, -1], [1, 0]], dtype=float),
    np.array([[-1, 0], [0, -1]], dtype=float),
    np.array([[0, 1], [-1, 0]], dtype=float),
]


def _build_half_edge_map(triangles):
    """Map directed half-edge (v_from, v_to) -> (tri_index, local_edge_index)."""
    he_map = {}
    for ti in range(len(triangles)):
        for ei in range(3):
            v_from = int(triangles[ti, ei])
            v_to = int(triangles[ti, (ei + 1) % 3])
            he_map[(v_from, v_to)] = (ti, ei)
    return he_map


def _find_cut_edges_and_transitions(triangles, uv_per_triangle, tolerance=1e-6):
    """Find all cut edges and recover their transition functions."""
    he_map = _build_half_edge_map(triangles)
    transitions = []
    processed = set()

    for (v_from, v_to), (ti, ei) in he_map.items():
        opp = (v_to, v_from)
        if opp not in he_map:
            continue

        edge_key = (min(v_from, v_to), max(v_from, v_to))
        if edge_key in processed:
            continue
        processed.add(edge_key)

        tj, ej = he_map[opp]

        uv_from_i = uv_per_triangle[ti, ei]
        uv_to_i = uv_per_triangle[ti, (ei + 1) % 3]
        uv_to_j = uv_per_triangle[tj, ej]
        uv_from_j = uv_per_triangle[tj, (ej + 1) % 3]

        if (np.linalg.norm(uv_from_i - uv_from_j) < tolerance and
                np.linalg.norm(uv_to_i - uv_to_j) < tolerance):
            continue

        best_R_idx = -1
        best_t = None
        best_err = float('inf')

        for ri, R in enumerate(_ROTATIONS):
            t1 = uv_from_i - R @ uv_from_j
            t2 = uv_to_i - R @ uv_to_j
            err = np.linalg.norm(t1 - t2)
            if err < best_err:
                best_err = err
                best_R_idx = ri
                best_t = (t1 + t2) / 2

        edge_len_i = np.linalg.norm(uv_to_i - uv_from_i)
        edge_len_j = np.linalg.norm(uv_to_j - uv_from_j)
        scale_ratio = edge_len_i / edge_len_j if edge_len_j > 1e-12 else 1.0

        if best_err < tolerance * 10:
            transitions.append({
                'tri_i': ti, 'tri_j': tj,
                'edge': (v_from, v_to),
                'R_idx': best_R_idx,
                'R': _ROTATIONS[best_R_idx].copy(),
                't': best_t,
                'residual': best_err,
                'scale_ratio': scale_ratio,
            })

    return transitions


# =========================================================================
# Main extraction entry point
# =========================================================================

def extract_quads(vertices, triangles, uv_per_triangle,
                  vertex_valences=None, fill_holes=True,
                  max_hole_size=None, verbose=True, merge_tolerance=1e-6,
                  cross_cut='none'):
    """Extract a quad mesh from a parameterized triangle mesh.

    Drop-in replacement using libQEx iso-line tracing algorithm.

    Parameters
    ----------
    vertices : ndarray, shape (n_verts, 3)
    triangles : ndarray, shape (n_tris, 3)
    uv_per_triangle : ndarray, shape (n_tris, 3, 2)
    vertex_valences : ignored (API compatibility)
    fill_holes : bool
    max_hole_size : int or None
    verbose : bool
    merge_tolerance : float (unused, kept for API compat)
    cross_cut : str (unused, libQEx always traces across cuts)

    Returns
    -------
    quad_vertices : ndarray, shape (n_quad_verts, 3)
    quad_faces : ndarray, shape (n_quads, 4)
    tri_faces : ndarray or None
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    uv_per_triangle = np.asarray(uv_per_triangle, dtype=np.float64)

    if verbose:
        uv_flat = uv_per_triangle.reshape(-1, 2)
        uv_range = (uv_flat.min(axis=0), uv_flat.max(axis=0))
        print(f"  UV range: [{uv_range[0][0]:.2f}, {uv_range[1][0]:.2f}] x "
              f"[{uv_range[0][1]:.2f}, {uv_range[1][1]:.2f}]")

    # Step 1: Build half-edge mesh
    mesh = _HalfEdgeMesh(triangles, len(vertices))

    # Step 2: Convert UV to per-halfedge format
    uv_coords = uv_per_triangle.reshape(-1).copy()

    # Step 3: Extract transition functions
    tf_per_edge = _extract_transitions(mesh, uv_coords)

    if verbose:
        n_cut = sum(1 for t in tf_per_edge if t != _TF_IDENTITY and not mesh.is_boundary_edge(tf_per_edge.index(t) if t in tf_per_edge else -1))
        n_cut = sum(1 for eidx in range(mesh.n_edges)
                    if not mesh.is_boundary_edge(eidx) and tf_per_edge[eidx] != _TF_IDENTITY)
        print(f"  Extracted {mesh.n_edges} edge transitions ({n_cut} non-trivial cuts)")

    # Step 4: Consistent truncation
    _consistent_truncation(mesh, uv_coords, tf_per_edge)

    # Step 5: Generate grid vertices
    gvertices, face_gv, edge_gv, vertex_gv = _generate_grid_vertices(
        mesh, uv_coords, vertices, tf_per_edge)

    if verbose:
        n_face = sum(1 for gv in gvertices if gv.type == _GV_ON_FACE)
        n_edge = sum(1 for gv in gvertices if gv.type == _GV_ON_EDGE)
        n_vert = sum(1 for gv in gvertices if gv.type == _GV_ON_VERTEX)
        print(f"  Grid vertices: {len(gvertices)} total "
              f"({n_face} face, {n_edge} edge, {n_vert} vertex)")

    if len(gvertices) == 0:
        if verbose:
            print("  Warning: No grid vertices found. UV scale may be too small.")
        return np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int32), None

    # Step 6: Generate connections via iso-line tracing
    _generate_connections(mesh, uv_coords, tf_per_edge,
                          gvertices, face_gv, edge_gv, vertex_gv, verbose)

    # Step 7: Generate faces
    all_faces = _generate_faces(gvertices, verbose)

    # Separate quads and triangles
    quad_faces_list = [f for f in all_faces if len(f) == 4]
    tri_faces_list = [f for f in all_faces if len(f) == 3]

    # Build output vertex array
    out_vertices_list = [np.array(gv.position_3d) for gv in gvertices]

    # Remove isolated vertices (not part of any face)
    used = set()
    for f in all_faces:
        used.update(f)

    if len(used) < len(gvertices):
        old_to_new = {}
        final_vertices = []
        for old_idx in sorted(used):
            old_to_new[old_idx] = len(final_vertices)
            final_vertices.append(out_vertices_list[old_idx])
        quad_faces_list = [[old_to_new[v] for v in f] for f in quad_faces_list]
        tri_faces_list = [[old_to_new[v] for v in f] for f in tri_faces_list]
        out_vertices = np.array(final_vertices) if final_vertices else np.zeros((0, 3))
    else:
        out_vertices = np.array(out_vertices_list) if out_vertices_list else np.zeros((0, 3))

    quad_faces = np.array(quad_faces_list, dtype=np.int32) if quad_faces_list else np.zeros((0, 4), dtype=np.int32)

    # Fill holes if requested
    tri_faces = None
    if fill_holes and len(quad_faces) > 0:
        try:
            from rectangular_surface_parameterization.utils.libqex_wrapper import (
                _fill_holes_with_triangles)
            hole_tris, new_verts = _fill_holes_with_triangles(
                out_vertices, quad_faces, verbose=verbose,
                max_hole_size=max_hole_size)
            if hole_tris:
                tri_faces = np.array(hole_tris, dtype=np.int32)
                if len(new_verts) > 0:
                    out_vertices = np.vstack([out_vertices, new_verts])
        except ImportError:
            pass

    if tri_faces is None and tri_faces_list:
        tri_faces = np.array(tri_faces_list, dtype=np.int32)

    if verbose:
        _validate_mesh(out_vertices, quad_faces, verbose=True)

    return out_vertices, quad_faces, tri_faces
