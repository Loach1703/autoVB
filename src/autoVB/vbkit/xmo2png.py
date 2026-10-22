"""Prepare XMO structures for optional PyVista PNG/interactive HTML rendering."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from ..draw_xmo.pyvista_drawer import prepare_pyvista_scene, render_pyvista_scene
from ..draw_xmo.electron_distribution import electron_counts
from ..draw_xmo.xmo_drawer_input_converter import XmoToDrawerInputConverter
from ..io.xmo_output_parser import XmoParser


@dataclass
class Xmo2PngResult:
    output_file: Path
    structure_count: int
    displayed_atom_count: int
    weight_table: str


def xmo2png_file(
    xmo_file: str | Path,
    output: str | Path | None = None,
    *,
    weight_table: str = "both",
    max_structures: int = 3,
    structures_per_row: int = 3,
    panel_size: int = 640,
    show_hydrogens: bool = True,
    show_atom_labels: bool = False,
    show_connection_labels: bool = False,
    show_charges: bool = True,
    baseline_index: int | None = None,
    transparent_background: bool = False,
    projection: str = "pca",
    azimuth: float | None = None,
    elevation: float | None = None,
) -> Xmo2PngResult:
    """Draw the highest-weight structures at the original geometry; no calculations."""
    if max_structures <= 0 or structures_per_row <= 0 or panel_size < 128:
        raise ValueError("Structure counts must be positive and panel_size must be at least 128")
    if projection not in ("pca", "optimized3d"):
        raise ValueError("PyVista projection must be 'pca' or 'optimized3d'")
    source = Path(xmo_file)
    output = Path(output) if output is not None else source.with_name(source.stem + "_3d.png")
    if output.suffix.lower() not in (".png", ".html"):
        raise ValueError("PyVista output must use the .png or .html extension")
    parsed = XmoParser(source).parse()
    # Reuse SVG structure selection/legends without rewriting XYZ beside the source.
    # Keep geometry indices intact; hydrogen visibility is handled by the 3D scene.
    with TemporaryDirectory(prefix="autovb-pyvista-xyz-") as directory:
        converter = XmoToDrawerInputConverter(
            parsed, directory, hide_hydrogens=False, max_structures=max_structures,
            weight_table=weight_table, show_connection_labels=show_connection_labels,
            baseline_index=baseline_index,
        )
        drawer_input = converter.convert()
    scene = prepare_pyvista_scene(
        parsed, drawer_input.active_space, show_hydrogens=show_hydrogens,
        baseline_electron_counts=electron_counts(drawer_input.active_bond_atom,
                                                 drawer_input.baseline_unpaired_atoms),
    )
    render_pyvista_scene(
        scene, output, structures_per_row=structures_per_row, panel_size=panel_size,
        show_atom_labels=show_atom_labels, transparent_background=transparent_background,
        show_charges=show_charges,
        projection=projection, azimuth=azimuth, elevation=elevation,
    )
    return Xmo2PngResult(output, len(scene.structures), len(scene.visible_atoms), drawer_input.weight_table)
