"""
Mesh preprocessing utilities to prepare meshes for the RSP pipeline.

Uses PyMeshLab to clean and remesh input meshes to meet pipeline requirements:
- Manifold geometry (each edge shared by at most 2 triangles)
- A single connected surface (RSP parameterizes one connected component;
  tiny floating debris is removed, but legitimate separate parts are kept
  and reported as an error instead of being silently deleted)
- Well-shaped triangles (avoid very obtuse/skinny triangles)
- Consistent orientation

Note: boundaries/holes are preserved by default. RSP supports surfaces with
boundary, so intentional openings (e.g. a model's eye sockets) are NOT
filled unless ``close_holes=True`` is explicitly requested.

Usage:
    from rectangular_surface_parameterization.utils.preprocess_mesh import preprocess_mesh

    clean_path = preprocess_mesh("bunny.obj", "bunny_clean.obj")
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings


def _count_boundary_edges(ms) -> Optional[int]:
    """Return the number of boundary (border) edges, or None if unavailable."""
    try:
        info = ms.get_topological_measures()
        return int(info.get('boundary_edges', 0))
    except Exception:
        return None


def _close_all_holes(ms, verbose: bool = False) -> None:
    """Close all holes iteratively so the surface becomes watertight.

    Uses increasing maximum hole sizes and stops as soon as no boundary edge
    remains. This only adds faces to fill existing holes, so it does not alter
    the appearance of the genuine surface.
    """
    try:
        for max_size in [100, 1000, 10000, 100000, 1000000]:
            ms.meshing_close_holes(maxholesize=max_size)
            n_border = _count_boundary_edges(ms)
            if n_border == 0:
                break
    except Exception as e:  # pragma: no cover - depends on pymeshlab version
        if verbose:
            print(f"    Hole closing skipped: {e}")


def _connected_components(ms):
    """Return the connected components of the current mesh as trimesh meshes.

    Returns None if trimesh is unavailable or the mesh cannot be analyzed.
    """
    try:
        import trimesh
    except ImportError:
        return None

    import tempfile

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            tmp_path = f.name
        ms.save_current_mesh(tmp_path)

        mesh = trimesh.load(tmp_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                return []
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

        return list(mesh.split(only_watertight=False))
    except Exception:
        return None
    finally:
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass


def _remove_small_debris(ms, debris_fraction: float = 0.02,
                         verbose: bool = False) -> Optional[int]:
    """Remove only tiny floating components ("debris"), keeping every
    significant part of the mesh.

    A component counts as debris when it has fewer faces than
    ``debris_fraction`` times the largest component. This clears stray
    islands left by repair/decimation WITHOUT ever deleting legitimate
    separate parts (e.g. a model's eyes or ears), so the visible shape is
    preserved.

    Returns
    -------
    int or None
        Number of *significant* connected components remaining, or None if
        the component structure could not be determined (trimesh missing).
    """
    # Fast path: a single component needs no analysis.
    try:
        n_cc = ms.get_topological_measures().get('connected_components_number', None)
        if n_cc is not None and int(n_cc) <= 1:
            return 1
    except Exception:
        pass

    components = _connected_components(ms)
    if components is None:
        if verbose:
            print("    trimesh not available; cannot analyze connected components")
        return None
    if len(components) <= 1:
        return 1

    largest_faces = max(len(c.faces) for c in components)
    threshold = max(1.0, debris_fraction * largest_faces)
    significant = [c for c in components if len(c.faces) >= threshold]
    debris = [c for c in components if len(c.faces) < threshold]

    if debris:
        try:
            import trimesh
            import tempfile

            kept = (significant[0] if len(significant) == 1
                    else trimesh.util.concatenate(significant))
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
                    tmp_path = f.name
                kept.export(tmp_path)
                ms.load_new_mesh(tmp_path)
            finally:
                if tmp_path is not None:
                    try:
                        Path(tmp_path).unlink()
                    except Exception:
                        pass
            if verbose:
                dropped_faces = sum(len(c.faces) for c in debris)
                print(f"    Removed {len(debris)} tiny debris component(s) "
                      f"({dropped_faces} face(s)); kept {len(significant)} "
                      f"significant component(s)")
        except Exception as e:  # pragma: no cover - depends on optional deps
            if verbose:
                print(f"    Debris removal skipped: {e}")

    return len(significant)


def _raise_if_multiple_components(n_significant: Optional[int]) -> None:
    """Raise a clear error when the mesh has more than one significant part.

    RSP parameterizes a single connected surface; it cannot treat several
    disconnected parts as one. We refuse to silently delete legitimate parts
    (which would mutilate models such as Blender's Suzanne), and instead stop
    with an actionable message.
    """
    if n_significant is not None and n_significant > 1:
        raise ValueError(
            f"Mesh has {n_significant} significant connected components. "
            f"RSP parameterizes a single connected surface and cannot process "
            f"multiple disconnected parts as one. Separate the mesh and run the "
            f"pipeline on one component at a time, or join the parts into a "
            f"single connected surface before running it. (Tiny debris is "
            f"removed automatically; these are real parts, so they are kept.)"
        )


def preprocess_mesh(
    input_path: str,
    output_path: Optional[str] = None,
    target_edge_length: Optional[float] = None,
    target_faces: Optional[int] = None,
    remesh_iterations: int = 5,
    verbose: bool = True,
    close_holes: bool = False,
    debris_fraction: float = 0.02
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
    close_holes : bool
        Fill boundary holes to make the surface watertight. Disabled by
        default: RSP tolerates boundaries, and closing holes would destroy
        intentional openings (e.g. a model's eye sockets). Enable only when
        a mesh has genuine defects you explicitly want filled.
    debris_fraction : float
        A connected component is treated as removable "debris" when it has
        fewer faces than this fraction of the largest component (default
        0.02 = 2%). Significant separate parts are always kept.

    Returns
    -------
    str
        Path to the cleaned mesh file.

    Raises
    ------
    ValueError
        If the mesh still has more than one significant connected component
        after debris removal. RSP can only parameterize a single connected
        surface, so we stop with an actionable message instead of silently
        deleting legitimate parts.
    """
    try:
        import pymeshlab
    except ImportError:
        raise ImportError("PyMeshLab required. Install with: pip install pymeshlab")

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_clean{input_path.suffix}"
    output_path = Path(output_path)

    if verbose:
        print(f"Preprocessing mesh: {input_path}")

    # Create MeshSet and load mesh
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(input_path))

    # Get initial stats
    mesh = ms.current_mesh()
    n_verts_initial = mesh.vertex_number()
    n_faces_initial = mesh.face_number()

    if verbose:
        print(f"  Initial: {n_verts_initial} vertices, {n_faces_initial} faces")

    # Step 1: Remove duplicate vertices and faces
    if verbose:
        print("  Removing duplicates...")
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()

    # Step 2: Remove unreferenced vertices
    ms.meshing_remove_unreferenced_vertices()

    # Step 3: Remove zero-area faces
    ms.meshing_remove_null_faces()

    # Step 4: Repair non-manifold edges/vertices
    if verbose:
        print("  Repairing non-manifold geometry...")
    try:
        ms.meshing_repair_non_manifold_edges()
    except Exception:
        pass  # May not be needed
    try:
        ms.meshing_repair_non_manifold_vertices()
    except Exception:
        pass

    # Step 4b: Remove tiny floating debris (non-destructive).
    # RSP requires a single connected surface, but we must NOT delete
    # legitimate separate parts (eyes, ears, ...). We only strip negligible
    # debris and then fail loudly if several significant parts remain, so
    # the user can decide how to handle a genuinely multi-part model.
    if verbose:
        print("  Removing tiny debris components...")
    n_significant = _remove_small_debris(ms, debris_fraction=debris_fraction,
                                         verbose=verbose)
    _raise_if_multiple_components(n_significant)

    # Step 5: Close holes (OFF by default).
    # RSP tolerates boundaries, and closing holes destroys intentional
    # openings, so this only runs when explicitly requested.
    if close_holes:
        if verbose:
            print("  Closing holes...")
        _close_all_holes(ms, verbose=verbose)

    # Step 7: Re-orient faces consistently
    if verbose:
        print("  Re-orienting faces...")
    try:
        ms.meshing_re_orient_faces_coherentely()
    except Exception:
        pass

    # Step 8: Decimation or remeshing
    current_faces = ms.current_mesh().face_number()

    if target_faces is not None and target_faces < current_faces:
        # Use decimation for large meshes
        if verbose:
            print(f"  Decimating from {current_faces} to ~{target_faces} faces...")
        try:
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
            # Decimation can leave tiny disconnected slivers. Repair
            # non-manifold geometry and strip only that debris (legitimate
            # parts are preserved).
            try:
                ms.meshing_repair_non_manifold_edges()
            except Exception:
                pass
            n_significant = _remove_small_debris(
                ms, debris_fraction=debris_fraction, verbose=verbose)
            _raise_if_multiple_components(n_significant)
            if close_holes:
                _close_all_holes(ms, verbose=verbose)
        except ValueError:
            raise
        except Exception as e:
            if verbose:
                print(f"    Decimation failed: {e}")
    elif target_edge_length is not None or target_faces is None:
        # Isotropic remeshing for better triangle quality
        if verbose:
            print("  Remeshing for better triangle quality...")

        # Compute target edge length if not specified
        if target_edge_length is None:
            bbox = ms.current_mesh().bounding_box()
            diagonal = np.sqrt(
                (bbox.max()[0] - bbox.min()[0])**2 +
                (bbox.max()[1] - bbox.min()[1])**2 +
                (bbox.max()[2] - bbox.min()[2])**2
            )
            target_edge_length = diagonal / 50

        if verbose:
            print(f"    Target edge length: {target_edge_length:.6f}")

        try:
            ms.meshing_isotropic_explicit_remeshing(
                targetlen=pymeshlab.PercentageValue(1.0),  # 1% of bbox diagonal
                iterations=remesh_iterations,
                adaptive=True
            )
        except Exception as e:
            if verbose:
                print(f"    Remeshing failed: {e}")
                print("    Skipping remeshing step...")

    # Step 9: Final cleanup
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()

    # Step 9b: Final non-destructive safety pass before RSP.
    # Strip any debris reintroduced by remeshing and confirm the mesh is a
    # single significant component. We never fill holes here (RSP tolerates
    # boundaries) so intentional openings survive untouched.
    n_significant = _remove_small_debris(ms, debris_fraction=debris_fraction,
                                         verbose=verbose)
    _raise_if_multiple_components(n_significant)
    ms.meshing_remove_unreferenced_vertices()

    # Get final stats
    mesh = ms.current_mesh()
    n_verts_final = mesh.vertex_number()
    n_faces_final = mesh.face_number()

    if verbose:
        print(f"  Final: {n_verts_final} vertices, {n_faces_final} faces")
        n_border_final = _count_boundary_edges(ms)
        if n_border_final:
            print(f"  Note: mesh has {n_border_final} boundary edge(s) "
                  f"(open holes preserved; RSP supports boundaries).")

    # Save result
    ms.save_current_mesh(str(output_path))

    if verbose:
        print(f"  Saved to: {output_path}")

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
        'vertices': mesh.vertex_number(),
        'faces': mesh.face_number(),
        'issues': [],
        'is_manifold': True,
        'is_closed': True,
        'has_consistent_orientation': True,
    }

    # Check for non-manifold edges
    try:
        ms.compute_selection_by_non_manifold_edges_per_face()
        n_non_manifold = mesh.selected_face_number()
        if n_non_manifold > 0:
            result['issues'].append(f"Non-manifold edges: {n_non_manifold} faces affected")
            result['is_manifold'] = False
        ms.set_selection_none()
    except Exception:
        pass

    # Check for boundary edges (holes)
    try:
        ms.compute_selection_by_border()
        n_border = mesh.selected_vertex_number()
        if n_border > 0:
            result['issues'].append(f"Boundary vertices: {n_border} (mesh has holes)")
            result['is_closed'] = False
        ms.set_selection_none()
    except Exception:
        pass

    # Compute quality metrics
    try:
        quality = ms.get_topological_measures()
        result['euler_characteristic'] = quality.get('euler_characteristic', None)
        result['genus'] = quality.get('genus', None)
        result['connected_components'] = quality.get('connected_components_number', None)
    except Exception:
        pass

    if verbose:
        print(f"Mesh: {mesh_path}")
        print(f"  Vertices: {result['vertices']}, Faces: {result['faces']}")
        print(f"  Manifold: {result['is_manifold']}, Closed: {result['is_closed']}")
        if result.get('genus') is not None:
            print(f"  Genus: {result['genus']}, Euler char: {result['euler_characteristic']}")
        if result['issues']:
            print(f"  Issues:")
            for issue in result['issues']:
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
        # Fallback: just return original
        ms.save_current_mesh(str(output_path))
        return str(output_path)

    ms.save_current_mesh(str(output_path))

    if verbose:
        print(f"  Saved to: {output_path}")

    return str(output_path)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Preprocess mesh for RSP pipeline',
        epilog='''
Examples:
  python preprocess_mesh.py bunny.obj bunny_clean.obj
  python preprocess_mesh.py bunny.obj bunny_clean.obj --target-faces 10000
        '''
    )
    parser.add_argument('input', help='Input mesh file')
    parser.add_argument('output', nargs='?', help='Output mesh file (default: input_clean.obj)')
    parser.add_argument('--target-faces', type=int, help='Target face count for decimation')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

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
    result_path = preprocess_mesh(input_path, output_path, target_faces=args.target_faces, verbose=verbose)

    # Check quality after
    print("\n" + "=" * 50)
    print("QUALITY CHECK (after)")
    print("=" * 50)
    check_mesh_quality(result_path)
