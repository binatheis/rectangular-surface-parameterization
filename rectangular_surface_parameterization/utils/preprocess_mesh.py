"""
Mesh preprocessing utilities to prepare meshes for the RSP pipeline.

Uses PyMeshLab to clean and remesh input meshes to meet pipeline requirements:
- Manifold geometry (each edge shared by exactly 2 triangles)
- Closed surface (no boundary edges)
- Well-shaped triangles (avoid very obtuse/skinny triangles)
- Consistent orientation

Large meshes use staged decimation and manual OBJ export to avoid PyMeshLab
MemoryError on save_current_mesh when filter history grows.

Usage:
    from rectangular_surface_parameterization.utils.preprocess_mesh import preprocess_mesh

    clean_path = preprocess_mesh("bunny.obj", "bunny_clean.obj")
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Meshes above this face count always use staged decimation + manual export.
STAGED_DECIMATION_THRESHOLD = 80_000

# Refuse to continue if preprocess could not get below this multiple of target_faces.
MAX_FACE_OVERSHOOT_RATIO = 2.5

# Hard cap when target_faces is unset (auto decimation path).
DEFAULT_MAX_RSP_FACES = 120_000


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message)


def _count_mesh_faces(path: Path) -> int:
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(path))
    return ms.current_mesh().face_number()


def _write_obj_manual(ms, output_path: Path) -> None:
    """
    Export current mesh without pymeshlab.save_current_mesh.

    PyMeshLab's save path can raise MemoryError on large MeshSet histories even when
    the current mesh is small; writing vertices/faces directly avoids that.
    """
    mesh = ms.current_mesh()
    verts = mesh.vertex_matrix()
    faces = mesh.face_matrix()

    try:
        import trimesh

        geometry = trimesh.Trimesh(
            vertices=np.asarray(verts, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
        geometry.export(str(output_path))
        return
    except ImportError:
        pass

    with open(output_path, "w", encoding="utf-8") as handle:
        for x, y, z in verts:
            handle.write(f"v {x} {y} {z}\n")
        for a, b, c in faces:
            handle.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")


def _compact_reload_ms(ms, scratch_path: Path):
    """Drop PyMeshLab filter history by round-tripping through a manual OBJ export."""
    import pymeshlab

    _write_obj_manual(ms, scratch_path)
    del ms
    gc.collect()
    fresh = pymeshlab.MeshSet()
    fresh.load_new_mesh(str(scratch_path))
    return fresh


def _decimate_staged(
    ms,
    target_faces: int,
    log: Callable[[str], None],
    scratch_path: Path,
):
    """Reduce face count in steps; return (meshset, final face count)."""
    current = ms.current_mesh().face_number()
    if current <= target_faces:
        return ms, current

    log(f"  Large mesh ({current:,} faces): using staged decimation")

    last = current
    while current > int(target_faces * 1.1):
        step_target = max(target_faces, int(current * 0.45))
        log(f"    Decimating: {current:,} -> ~{step_target:,} faces")
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=step_target)
        current = ms.current_mesh().face_number()
        if current >= last:
            log("    Decimation stalled; stopping further reduction")
            break
        last = current
        if current > target_faces:
            log("    Compacting mesh to free PyMeshLab history…")
            ms = _compact_reload_ms(ms, scratch_path)
            current = ms.current_mesh().face_number()
    return ms, current


def _try_close_holes(ms, log: Callable[[str], None]) -> None:
    try:
        ms.meshing_close_holes(maxholesize=1000)
    except Exception as exc:
        log(f"    Hole closing after decimation skipped: {exc}")


def _basic_cleanup(ms, log: Callable[[str], None], verbose: bool) -> None:
    if verbose:
        log("  Removing duplicates and repairing geometry…")
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_null_faces()
    for fn_name in (
        "meshing_repair_non_manifold_edges",
        "meshing_repair_non_manifold_vertices",
    ):
        try:
            getattr(ms, fn_name)()
        except Exception:
            pass
    try:
        for max_size in (100, 1000, 10000, 100000):
            ms.meshing_close_holes(maxholesize=max_size)
            info = ms.get_topological_measures()
            if info.get("boundary_edges", 1) == 0:
                break
    except Exception as exc:
        log(f"    Hole closing skipped: {exc}")
    try:
        ms.meshing_re_orient_faces_coherentely()
    except Exception:
        pass


def _close_small_triangular_hole(ms, log: Callable[[str], None]):
    """Close remaining 3-edge boundary loops via trimesh (legacy helper)."""
    import pymeshlab

    try:
        info = ms.get_topological_measures()
        if info["boundary_edges"] <= 0 or info["boundary_edges"] > 10:
            return ms
        log(
            f"  Closing small remaining hole ({info['boundary_edges']} boundary edges)…"
        )
        import tempfile

        import trimesh
        from collections import Counter

        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as handle:
            temp_path = Path(handle.name)
        _write_obj_manual(ms, temp_path)

        tmesh = trimesh.load(temp_path)
        edge_counts = Counter(tuple(sorted(edge)) for edge in tmesh.edges)
        boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]

        if len(boundary_edges) == 3:
            boundary_verts = list({v for edge in boundary_edges for v in edge})
            new_faces = np.vstack([tmesh.faces, boundary_verts])
            tmesh = trimesh.Trimesh(
                vertices=tmesh.vertices, faces=new_faces, process=True
            )
            tmesh.fix_normals()
            tmesh.export(str(temp_path))
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(temp_path))
            log("    Closed triangular hole")

        temp_path.unlink(missing_ok=True)
    except ImportError:
        log("    trimesh not available for small hole fix")
    except Exception:
        pass
    return ms


def _validate_for_rsp(
    output_path: Path,
    target_faces: Optional[int],
    log: Callable[[str], None],
) -> None:
    n_faces = _count_mesh_faces(output_path)

    if target_faces and target_faces > 0:
        limit = int(target_faces * MAX_FACE_OVERSHOOT_RATIO)
    else:
        limit = DEFAULT_MAX_RSP_FACES

    if n_faces > limit:
        raise RuntimeError(
            f"Preprocess left {n_faces:,} triangles (limit {limit:,}). "
            "The mesh is likely non-manifold or could not be decimated."
        )

    try:
        import pymeshlab

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(output_path))
        ms.compute_selection_by_non_manifold_edges_per_face()
        n_bad = ms.current_mesh().selected_face_number()
        if n_bad > 0:
            log(f"  Warning: {n_bad} faces touch non-manifold edges after preprocess")
    except Exception:
        pass


def _save_mesh(ms, output_path: Path, use_manual: bool) -> None:
    if use_manual:
        _write_obj_manual(ms, output_path)
    else:
        ms.save_current_mesh(str(output_path))


def preprocess_mesh(
    input_path: str,
    output_path: Optional[str] = None,
    target_edge_length: Optional[float] = None,
    target_faces: Optional[int] = None,
    remesh_iterations: int = 5,
    verbose: bool = True,
) -> str:
    """
    Preprocess a mesh for the RSP pipeline.

    Parameters
    ----------
    input_path : str
        Path to input mesh file (OBJ, PLY, STL, etc.)
    output_path : str, optional
        Path for output mesh. If None, appends '_clean' to input name.
    target_edge_length : float, optional
        Target edge length for remeshing. If None, uses average edge length.
    target_faces : int, optional
        Target number of faces for decimation. Recommended for meshes >10K faces.
        If specified, decimation is used instead of remeshing.
    remesh_iterations : int
        Number of remeshing iterations (default: 5)
    verbose : bool
        Print progress information

    Returns
    -------
    str
        Path to the cleaned mesh file.
    """
    try:
        import pymeshlab
    except ImportError:
        raise ImportError("PyMeshLab required. Install with: pip install pymeshlab")

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_clean{input_path.suffix}"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log = lambda message: _log(verbose, message)
    log(f"Preprocessing mesh: {input_path}")

    try:
        n_initial = _count_mesh_faces(input_path)
    except Exception:
        n_initial = 0

    use_robust_export = n_initial > STAGED_DECIMATION_THRESHOLD
    scratch_path = output_path.with_suffix(".preprocess_scratch.obj")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(input_path))
    n_verts_initial = ms.current_mesh().vertex_number()
    n_faces_initial = ms.current_mesh().face_number()
    log(f"  Initial: {n_verts_initial:,} vertices, {n_faces_initial:,} faces")

    if use_robust_export:
        log(
            f"  Mesh has {n_initial:,} faces; using low-memory staged preprocess."
        )

    _basic_cleanup(ms, log, verbose)
    ms = _close_small_triangular_hole(ms, log)

    current_faces = ms.current_mesh().face_number()

    if target_faces is not None and target_faces > 0 and current_faces > target_faces:
        if current_faces > STAGED_DECIMATION_THRESHOLD // 2 or use_robust_export:
            try:
                ms, final = _decimate_staged(ms, target_faces, log, scratch_path)
                log(f"  After decimation: {final:,} faces")
                _try_close_holes(ms, log)
                use_robust_export = True
            except Exception as exc:
                log(f"    Staged decimation failed: {exc}")
        else:
            log(f"  Decimating from {current_faces} to ~{target_faces} faces...")
            try:
                ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
                _try_close_holes(ms, log)
            except Exception as exc:
                log(f"    Decimation failed: {exc}")
    elif target_edge_length is not None or target_faces is None:
        log("  Remeshing for better triangle quality...")
        mesh = ms.current_mesh()
        if target_edge_length is None:
            bbox = mesh.bounding_box()
            diagonal = np.sqrt(
                (bbox.max()[0] - bbox.min()[0]) ** 2
                + (bbox.max()[1] - bbox.min()[1]) ** 2
                + (bbox.max()[2] - bbox.min()[2]) ** 2
            )
            target_edge_length = diagonal / 50
        log(f"    Target edge length: {target_edge_length:.6f}")
        try:
            ms.meshing_isotropic_explicit_remeshing(
                targetlen=pymeshlab.PercentageValue(1.0),
                iterations=remesh_iterations,
                adaptive=True,
            )
        except Exception as exc:
            log(f"    Remeshing failed: {exc}")
            log("    Skipping remeshing step...")

    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()

    if use_robust_export or ms.current_mesh().face_number() > STAGED_DECIMATION_THRESHOLD // 4:
        log("  Final compact before export…")
        ms = _compact_reload_ms(ms, scratch_path)
        use_robust_export = True

    n_verts_final = ms.current_mesh().vertex_number()
    n_faces_final = ms.current_mesh().face_number()
    log(f"  Final: {n_verts_final:,} vertices, {n_faces_final:,} faces")

    try:
        _save_mesh(ms, output_path, use_manual=use_robust_export)
    except MemoryError as exc:
        log("  save_current_mesh failed; retrying with manual OBJ export…")
        _write_obj_manual(ms, output_path)
    finally:
        del ms
        gc.collect()
        scratch_path.unlink(missing_ok=True)

    log(f"  Saved to: {output_path}")
    _validate_for_rsp(output_path, target_faces, log)
    return str(output_path)


def check_mesh_quality(mesh_path: str, verbose: bool = True) -> dict:
    """
    Check mesh quality and return diagnostic information.

    Parameters
    ----------
    mesh_path : str
        Path to mesh file.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    dict
        Dictionary with quality metrics and issues found.
    """
    try:
        import pymeshlab
    except ImportError:
        raise ImportError("PyMeshLab required. Install with: pip install pymeshlab")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_path))
    mesh = ms.current_mesh()

    result = {
        "vertices": mesh.vertex_number(),
        "faces": mesh.face_number(),
        "issues": [],
        "is_manifold": True,
        "is_closed": True,
        "has_consistent_orientation": True,
    }

    # Check for non-manifold edges
    try:
        ms.compute_selection_by_non_manifold_edges_per_face()
        n_non_manifold = mesh.selected_face_number()
        if n_non_manifold > 0:
            result["issues"].append(
                f"Non-manifold edges: {n_non_manifold} faces affected"
            )
            result["is_manifold"] = False
        ms.set_selection_none()
    except Exception:
        pass

    # Check for boundary edges (holes)
    try:
        ms.compute_selection_by_border()
        n_border = mesh.selected_vertex_number()
        if n_border > 0:
            result["issues"].append(
                f"Boundary vertices: {n_border} (mesh has holes)"
            )
            result["is_closed"] = False
        ms.set_selection_none()
    except Exception:
        pass

    # Compute quality metrics
    try:
        quality = ms.get_topological_measures()
        result["euler_characteristic"] = quality.get("euler_characteristic", None)
        result["genus"] = quality.get("genus", None)
        result["connected_components"] = quality.get(
            "connected_components_number", None
        )
    except Exception:
        pass

    if verbose:
        print(f"Mesh: {mesh_path}")
        print(f"  Vertices: {result['vertices']}, Faces: {result['faces']}")
        print(f"  Manifold: {result['is_manifold']}, Closed: {result['is_closed']}")
        if result.get("genus") is not None:
            print(
                f"  Genus: {result['genus']}, "
                f"Euler char: {result['euler_characteristic']}"
            )
        if result["issues"]:
            print("  Issues:")
            for issue in result["issues"]:
                print(f"    - {issue}")

    return result


def make_delaunay(mesh_path: str, output_path: Optional[str] = None, verbose: bool = True) -> str:
    """
    Apply intrinsic Delaunay triangulation to improve triangle quality.

    This flips edges to achieve a Delaunay triangulation, which helps
    avoid negative Voronoi areas.

    Parameters
    ----------
    mesh_path : str
        Path to input mesh.
    output_path : str, optional
        Path for output. If None, appends '_delaunay'.
    verbose : bool
        Print progress.

    Returns
    -------
    str
        Path to output mesh.
    """
    try:
        import pymeshlab
    except ImportError:
        raise ImportError("PyMeshLab required. Install with: pip install pymeshlab")

    mesh_path = Path(mesh_path)
    if output_path is None:
        output_path = mesh_path.parent / f"{mesh_path.stem}_delaunay{mesh_path.suffix}"
    output_path = Path(output_path)

    if verbose:
        print(f"Applying Delaunay triangulation to: {mesh_path}")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_path))

    # Apply Delaunay edge flipping
    try:
        ms.meshing_surface_delaunay_triangulation()
        if verbose:
            print("  Applied surface Delaunay triangulation")
    except Exception as e:
        if verbose:
            print(f"  Delaunay failed: {e}")
        _write_obj_manual(ms, output_path)
        return str(output_path)

    try:
        ms.save_current_mesh(str(output_path))
    except MemoryError:
        if verbose:
            print("  save_current_mesh failed; using manual OBJ export")
        _write_obj_manual(ms, output_path)

    if verbose:
        print(f"  Saved to: {output_path}")

    return str(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess mesh for RSP pipeline",
        epilog="""
Examples:
  python preprocess_mesh.py bunny.obj bunny_clean.obj
  python preprocess_mesh.py bunny.obj bunny_clean.obj --target-faces 10000
        """,
    )
    parser.add_argument("input", help="Input mesh file")
    parser.add_argument("output", nargs="?", help="Output mesh file (default: input_clean.obj)")
    parser.add_argument("--target-faces", type=int, help="Target face count for decimation")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    verbose = not args.quiet

    # Check quality first
    if verbose:
        print("=" * 50)
        print("QUALITY CHECK (before)")
        print("=" * 50)
        check_mesh_quality(input_path)

    # Preprocess
    if verbose:
        print("\n" + "=" * 50)
        print("PREPROCESSING")
        print("=" * 50)
    result_path = preprocess_mesh(
        input_path, output_path, target_faces=args.target_faces, verbose=verbose
    )

    # Check quality after
    print("\n" + "=" * 50)
    print("QUALITY CHECK (after)")
    print("=" * 50)
    check_mesh_quality(result_path)
