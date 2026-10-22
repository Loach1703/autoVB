"""Three-dimensional ball-and-stick scenes with schematic VB electron pairs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .molecule_bond_variant_drawer import ValenceBondStructureInfo
from .electron_distribution import charge_notes, electron_counts

if TYPE_CHECKING:
    from ..io.xmo_output_parser import XmoParsedData


@dataclass
class PyVistaScene:
    symbols: list[str]
    coordinates: np.ndarray
    bonds: list[tuple[int, int]]  # Zero-based geometry indices.
    visible_atoms: list[int]
    structures: list[ValenceBondStructureInfo]  # Original, one-based geometry indices.
    charge_notes: list[dict[int, str]]  # Per-structure notes, zero-based geometry indices.


def prepare_pyvista_scene(
    parsed: XmoParsedData,
    structures: list[ValenceBondStructureInfo],
    *,
    show_hydrogens: bool = True,
    baseline_electron_counts: dict[int, int] | None = None,
) -> PyVistaScene:
    """Keep the input geometry and build the framework from one-/two-center $orb labels."""
    symbols = [atom.symbol for atom in parsed.geo]
    coordinates = np.array([[atom.x, atom.y, atom.z] for atom in parsed.geo], dtype=float)
    if not symbols or not np.isfinite(coordinates).all():
        raise ValueError("A finite, nonempty XMO geometry is required")
    if not structures:
        raise ValueError("No weighted VB structures were selected")
    if baseline_electron_counts is None:
        baseline_electron_counts = electron_counts(structures[0].bond_pairs,
                                                   structures[0].unpaired_atoms)
    if any(atom < 1 or atom > len(symbols) for atom in baseline_electron_counts):
        raise ValueError("Reference atom number is outside the geometry")

    bonds = set()
    for row in parsed.orb:
        if not 1 <= len(row) <= 2:
            raise ValueError("PyVista drawing currently requires one-/two-center $orb atom labels")
        if any(atom < 1 or atom > len(symbols) for atom in row):
            raise ValueError("$orb atom number is outside the geometry")
        if len(row) == 2 and row[0] != row[1]:
            bonds.add(tuple(sorted((row[0] - 1, row[1] - 1))))

    # Retain a reference H even if it has lost all active electrons in a panel.
    active_atoms = {atom - 1 for atom in baseline_electron_counts}
    for structure in structures:
        active_atoms.update(atom - 1 for pair in structure.bond_pairs for atom in pair)
        active_atoms.update(atom - 1 for atom in structure.unpaired_atoms)
    if any(atom < 0 or atom >= len(symbols) for atom in active_atoms):
        raise ValueError("VB structure atom number is outside the geometry")
    visible = [i for i, symbol in enumerate(symbols)
               if show_hydrogens or symbol.upper() != "H" or i in active_atoms]
    visible_set = set(visible)
    bonds = sorted(pair for pair in bonds if set(pair) <= visible_set)
    for a, b in bonds:
        if np.linalg.norm(coordinates[a] - coordinates[b]) < 1e-8:
            raise ValueError("Bond endpoints have coincident coordinates")
    notes = [charge_notes(baseline_electron_counts, structure.bond_pairs,
                          structure.unpaired_atoms) for structure in structures]
    return PyVistaScene(symbols, coordinates, bonds, visible, structures, notes)


def scene_camera(
    scene: PyVistaScene, azimuth: float | None = None, elevation: float | None = None,
    *, projection: str = "pca", show_charges: bool = True,
):
    """One camera for every panel, oriented relative to the molecular principal plane."""
    if projection not in ("pca", "optimized3d"):
        raise ValueError("PyVista projection must be 'pca' or 'optimized3d'")
    automatic = projection == "optimized3d" and azimuth is None and elevation is None
    azimuth = 35 if azimuth is None else azimuth
    elevation = 55 if elevation is None else elevation
    if not np.isfinite([azimuth, elevation]).all():
        raise ValueError("Camera angles must be finite")
    points = scene.coordinates[scene.visible_atoms]
    center = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - center, full_matrices=True)
    u, v, normal = axes
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    azimuth, elevation = np.deg2rad([azimuth, elevation])
    direction = (normal * np.sin(elevation)
                 + (u * np.cos(azimuth) + v * np.sin(azimuth)) * np.cos(elevation))
    if automatic:
        direction = _optimized_view_direction(scene, center, u, v, normal, direction,
                                               show_charges=show_charges)
    extent = max(float(np.linalg.norm(np.ptp(points, axis=0))), 2.0)
    view_up = v
    if abs(np.dot(direction, view_up)) > 0.98:
        view_up = normal if abs(np.dot(direction, normal)) < 0.98 else u
    return (center + direction * extent * 3, center, view_up), extent * 0.58, normal, u


def _screen_axes(direction: np.ndarray, view_up: np.ndarray):
    if abs(np.dot(direction, view_up)) > 0.98:
        view_up = np.eye(3)[np.argmin(np.abs(direction))]
    right = np.cross(view_up, direction)
    right /= np.linalg.norm(right)
    return right, np.cross(direction, right)


def _charge_offset(direction: np.ndarray, right: np.ndarray, up: np.ndarray):
    return 0.38 * right + 0.42 * up + 0.1 * direction


def _optimized_view_direction(scene, center, u, v, normal, initial, *, show_charges):
    """Search cameras using the SVG score plus sphere/marker occlusion penalties.

    A temporary, unsanitized topology is used only by the existing geometric SVG
    scorer. No bond-order perception is performed and no coordinates are changed.
    """
    from rdkit import Chem
    from .orbital_connectivity_molecule_drawer import OrbitalConnectivityMoleculeDrawer

    scorer = OrbitalConnectivityMoleculeDrawer
    atoms = scene.visible_atoms
    points = scene.coordinates[atoms] - center
    mapping = {atom: index for index, atom in enumerate(atoms)}
    topology = Chem.RWMol()
    for _ in atoms:
        topology.AddAtom(Chem.Atom(0))  # Only topology is needed by the geometric scorer.
    for a, b in scene.bonds:
        topology.AddBond(mapping[a], mapping[b], Chem.BondType.SINGLE)
    mol = topology.GetMol()
    radii = np.array([ATOM_RADII.get(scene.symbols[atom].upper(), 0.26) for atom in atoms])
    atom_pairs = np.triu_indices(len(atoms), k=1)
    markers = []
    framework = set(scene.bonds)
    # Score all panels, including lifted electron pairs, using representative points.
    for structure in scene.structures:
        panel = []
        occurrences = Counter()
        for begin, end in structure.bond_pairs:
            a, b = begin - 1, end - 1
            pair = tuple(sorted((a, b)))
            offset = 0.29 + occurrences[pair] * 0.18
            occurrences[pair] += 1
            if a == b:
                point = scene.coordinates[a] + normal * (0.50 + (occurrences[pair] - 1) * 0.2)
                panel.extend([point - u * 0.09, point + u * 0.09])
            else:
                t = np.array([0.25, 0.5, 0.75])
                lift = offset if pair in framework else offset + 0.65 * np.sin(np.pi * t)
                panel.extend((1 - t[:, None]) * scene.coordinates[a]
                             + t[:, None] * scene.coordinates[b]
                             + np.asarray(lift)[..., None] * normal)
        for atom, count in Counter(structure.unpaired_atoms).items():
            panel.extend(scene.coordinates[atom - 1] + normal * 0.55 + u * i * 0.18
                         for i in range(count))
        markers.append(np.array(panel, dtype=float).reshape(-1, 3) - center)

    def score(direction):
        projected = scorer._orthographic_projection(points, direction)
        result = scorer._optimized_projection_score(mol, points, projected)
        view_up = v
        if abs(np.dot(direction, view_up)) > 0.98:
            view_up = normal if abs(np.dot(direction, normal)) < 0.98 else u
        right, up = _screen_axes(direction, view_up)
        basis = np.column_stack((right, up))
        screen = points @ basis
        depths = points @ direction
        # Account for finite atom spheres, rather than just atom-center distances.
        a, b = atom_pairs
        distance = np.linalg.norm(screen[a] - screen[b], axis=1)
        required = radii[a] + radii[b]
        result += 8 * float(np.sum(np.maximum(0, 1 - distance / required) ** 2))

        def obscured(samples, padding):
            if not len(samples):
                return 0.0
            distances = np.linalg.norm((samples @ basis)[:, None] - screen[None], axis=2)
            behind = (samples @ direction)[:, None] < depths[None]
            overlap = np.maximum(0, 1 - distances / (radii + padding)) ** 2
            return float(np.sum(overlap * behind))

        penalty = 0.0
        for index, panel in enumerate(markers):
            penalty += obscured(panel, 0.06)
            if show_charges and scene.charge_notes[index]:
                charged_atoms = list(scene.charge_notes[index])
                labels = (scene.coordinates[charged_atoms] - center
                          + _charge_offset(direction, right, up))
                penalty += obscured(labels, 0.12)
        result += 6 * penalty / len(scene.structures)
        # Prefer a moderately oblique view, while allowing readability to win.
        result += 0.25 * ((np.dot(direction, normal) - np.sin(np.deg2rad(55))) / 0.2) ** 2
        return result

    candidates = [initial, normal, u, v]
    candidates.extend(x * u + y * v + z * normal
                      for x, y, z in scorer._hemisphere_view_directions(240))
    return min(candidates, key=score)


ATOM_COLORS = {
    "H": "#E8E9ED", "C": "#394352", "N": "#376FCE", "O": "#D44747",
    "F": "#75BE65", "CL": "#5AB45A", "S": "#E3BC3D", "P": "#DF8C36",
}
ATOM_RADII = {"H": 0.14, "C": 0.23, "N": 0.25, "O": 0.24}
PAIR_COLOR = "#BB1938"


def render_pyvista_scene(
    scene: PyVistaScene,
    output: str | Path,
    *,
    structures_per_row: int = 3,
    panel_size: int = 640,
    show_atom_labels: bool = False,
    show_charges: bool = True,
    transparent_background: bool = False,
    projection: str = "pca",
    azimuth: float | None = None,
    elevation: float | None = None,
) -> Path:
    """Render real spheres/tubes and schematic pairs to PNG or interactive HTML.

    Gray/element-colored sticks are the $orb framework. Red sticks represent VB
    pairs, lifted slightly off the principal plane for visibility; nonlocal pairs
    use dashed arcs. Dots indicate self-pairs or radicals, not orbital isosurfaces.
    Charge labels show electron redistribution relative to the reference structure.
    PyVista is optional and imported only when a scene is actually rendered.
    """
    if structures_per_row < 1 or panel_size < 128:
        raise ValueError("structures_per_row must be positive and panel_size must be at least 128")
    output = Path(output)
    html_output = output.suffix.lower() == ".html"
    if output.suffix.lower() not in (".png", ".html"):
        raise ValueError("PyVista output must use the .png or .html extension")
    if html_output:
        if transparent_background:
            raise ValueError("transparent_background is only supported for PNG output")
        try:
            import trame_pyvista  # noqa: F401 - registers the PyVista HTML exporter
            import nest_asyncio2  # noqa: F401 - required by synchronous trame export
        except ImportError as exc:
            raise ImportError("Interactive HTML requires: pip install 'autoVB[visualization-html]'") from exc
    camera, parallel_scale, normal, tangent = scene_camera(
        scene, azimuth, elevation, projection=projection, show_charges=show_charges,
    )
    view_direction = np.asarray(camera[0]) - np.asarray(camera[1])
    view_direction /= np.linalg.norm(view_direction)
    screen_right, screen_up = _screen_axes(view_direction, camera[2])
    charge_offset = _charge_offset(view_direction, screen_right, screen_up)
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("PyVista rendering requires: pip install 'autoVB[visualization]'") from exc

    columns = min(structures_per_row, len(scene.structures))
    rows = ceil(len(scene.structures) / columns)
    plotter = pv.Plotter(shape=(rows, columns), off_screen=True,
                         window_size=(panel_size * columns, panel_size * rows), border=False)
    mesh_style = dict(smooth_shading=True, ambient=0.25, diffuse=0.75,
                      specular=0.35, specular_power=25)
    visible = set(scene.visible_atoms)
    framework = set(scene.bonds)
    html_panels = []

    def sphere(point, radius, color):
        mesh = pv.Sphere(center=point, radius=radius, theta_resolution=28, phi_resolution=28)
        plotter.add_mesh(mesh, color=color, **mesh_style)

    def stick(start, end, radius, color):
        distance = np.linalg.norm(end - start)
        if distance < 1e-8:
            raise ValueError("Electron-pair or bond endpoints have coincident coordinates")
        mesh = pv.Cylinder(center=(start + end) / 2, direction=end - start,
                           radius=radius, height=distance, resolution=20)
        plotter.add_mesh(mesh, color=color, **mesh_style)

    try:
        for index in range(rows * columns):
            plotter.subplot(index // columns, index % columns)
            plotter.set_background("white")
            if index >= len(scene.structures):
                continue
            structure = scene.structures[index]
            if html_output:
                row, column = divmod(index, columns)
                labels = []
                if show_atom_labels:
                    labels.extend(dict(text=f"{scene.symbols[atom]}{atom + 1}", kind="atom",
                                       position=(scene.coordinates[atom] + normal * 0.35).tolist())
                                  for atom in scene.visible_atoms)
                if show_charges:
                    labels.extend(dict(text=note.replace("-", "−"), kind="charge",
                                       position=scene.coordinates[atom].tolist(),
                                       radius=ATOM_RADII.get(scene.symbols[atom].upper(), 0.26))
                                  for atom, note in scene.charge_notes[index].items())
                html_panels.append(dict(rank=index + 1, legend=structure.legend, labels=labels,
                                        viewport=[column / columns, 1 - (row + 1) / rows,
                                                  (column + 1) / columns, 1 - row / rows],
                                        camera=dict(position=np.asarray(camera[0]).tolist(),
                                                    focalPoint=np.asarray(camera[1]).tolist(),
                                                    viewUp=np.asarray(camera[2]).tolist(),
                                                    parallelProjection=True,
                                                    parallelScale=float(parallel_scale))))
            for a, b in scene.bonds:
                start, end = scene.coordinates[[a, b]]
                middle = (start + end) / 2
                stick(start, middle, 0.065, ATOM_COLORS.get(scene.symbols[a].upper(), "#888888"))
                stick(middle, end, 0.065, ATOM_COLORS.get(scene.symbols[b].upper(), "#888888"))
            for atom in scene.visible_atoms:
                symbol = scene.symbols[atom].upper()
                sphere(scene.coordinates[atom], ATOM_RADII.get(symbol, 0.26),
                       ATOM_COLORS.get(symbol, "#888888"))

            occurrences = Counter()
            for begin, end in structure.bond_pairs:
                a, b = begin - 1, end - 1
                if a not in visible or b not in visible:
                    raise ValueError("Active electrons cannot be placed on hidden atoms")
                pair = tuple(sorted((a, b)))
                offset = 0.29 + occurrences[pair] * 0.18
                occurrences[pair] += 1
                if a == b:
                    point = scene.coordinates[a] + normal * (0.50 + (occurrences[pair] - 1) * 0.2)
                    for sign in (-1, 1):
                        sphere(point + tangent * sign * 0.09, 0.065, PAIR_COLOR)
                elif pair in framework:
                    stick(scene.coordinates[a] + normal * offset,
                          scene.coordinates[b] + normal * offset, 0.047, PAIR_COLOR)
                else:
                    # Cross-ring/fragment pairings are separate from framework bonds.
                    start, end = scene.coordinates[[a, b]]
                    t = np.linspace(0, 1, 25)
                    points = ((1 - t[:, None]) * start + t[:, None] * end
                              + normal * (offset + 0.65 * np.sin(np.pi * t))[:, None])
                    for segment in range(len(points) - 1):
                        if segment % 3 != 2:
                            stick(points[segment], points[segment + 1], 0.04, PAIR_COLOR)
            for atom, count in Counter(structure.unpaired_atoms).items():
                for number in range(count):
                    sphere(scene.coordinates[atom - 1] + normal * 0.55
                           + tangent * number * 0.18, 0.075, PAIR_COLOR)

            if show_atom_labels:
                labels = [f"{scene.symbols[i]}{i + 1}" for i in scene.visible_atoms]
                plotter.add_point_labels(scene.coordinates[scene.visible_atoms] + normal * 0.35,
                                         labels, show_points=False, always_visible=True,
                                         font_size=14, shape_opacity=0, text_color="#111111")
            if show_charges and scene.charge_notes[index]:
                notes = scene.charge_notes[index]
                atoms = list(notes)
                plotter.add_point_labels(
                    scene.coordinates[atoms] + charge_offset,
                    [notes[atom].replace("-", "−") for atom in atoms],
                    show_points=False, always_visible=True, font_size=20, bold=True,
                    shape_opacity=0, text_color=PAIR_COLOR,
                )
            plotter.add_text(f"Rank {index + 1}", position="upper_left", font_size=13,
                             color="#333333", font="arial")
            plotter.add_text(structure.legend, position="lower_left", font_size=13,
                             color="#222222", font="arial")
            plotter.camera_position = camera
            plotter.enable_parallel_projection()
            plotter.camera.parallel_scale = parallel_scale
        plotter.enable_anti_aliasing("ssaa")
        output.parent.mkdir(parents=True, exist_ok=True)
        if html_output:
            from .html_export import export_interactive_html
            export_interactive_html(plotter, output, html_panels)
        else:
            plotter.screenshot(str(output), transparent_background=transparent_background,
                               return_img=False)
    finally:
        plotter.close()
    return output
