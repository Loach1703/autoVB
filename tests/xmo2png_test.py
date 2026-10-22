from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import numpy as np
import pytest

from autoVB.draw_xmo.molecule_bond_variant_drawer import ValenceBondStructureInfo
from autoVB.draw_xmo.pyvista_drawer import prepare_pyvista_scene, render_pyvista_scene, scene_camera
from autoVB.io.xmo_output_parser import XmoGeometryAtom
from autoVB.vbkit import xmo2png as workflow


def test_scene_preserves_original_atom_numbers_and_keeps_active_hydrogens():
    parsed = SimpleNamespace(
        geo=[XmoGeometryAtom("C", 0, 0, 0), XmoGeometryAtom("H", 0, 0, 1),
             XmoGeometryAtom("N", 1.4, 0, 0), XmoGeometryAtom("H", 2, 0, 0)],
        orb=[[1, 3], [1, 2], [3, 4]],
    )
    structure = ValenceBondStructureInfo("1", "Löwdin 1 w=1", [(3, 3)], [2])
    scene = prepare_pyvista_scene(parsed, [structure], show_hydrogens=False)
    assert scene.visible_atoms == [0, 1, 2]
    assert scene.bonds == [(0, 1), (0, 2)]
    assert scene.structures[0].bond_pairs == [(3, 3)]
    assert scene.structures[0].unpaired_atoms == [2]
    np.testing.assert_array_equal(scene.coordinates[2], [1.4, 0, 0])
    camera, scale, normal, _ = scene_camera(scene, 35, 55)
    assert scale > 0 and np.isfinite(camera).all()
    assert np.linalg.norm(normal) == pytest.approx(1)


@pytest.mark.parametrize("row", [[1, 2, 3], [0], [4]])
def test_invalid_or_unsupported_orbital_atom_labels_are_rejected(row):
    parsed = SimpleNamespace(geo=[XmoGeometryAtom("C", i, 0, 0) for i in range(3)], orb=[row])
    with pytest.raises(ValueError, match=r"\$orb"):
        prepare_pyvista_scene(parsed, [ValenceBondStructureInfo("1", "test", [(1, 2)])])


def test_bad_geometry_and_empty_structures_are_rejected():
    parsed = SimpleNamespace(geo=[XmoGeometryAtom("C", float("nan"), 0, 0)], orb=[[1]])
    with pytest.raises(ValueError, match="finite"):
        prepare_pyvista_scene(parsed, [ValenceBondStructureInfo("1", "test", [(1, 1)])])
    parsed.geo[0].x = 0
    with pytest.raises(ValueError, match="No weighted"):
        prepare_pyvista_scene(parsed, [])


@pytest.fixture
def xmo_file(tmp_path):
    source = tmp_path / "ethene.xmo"
    source.write_text("""$ctrl
vbscf
nae=2
nao=2
basis=sto-3g
$end
$orb
2*1 1*2
1 2
1
2
$end
$geo
C -0.67 0 0
C 0.67 0 0
$end
****** WEIGHTS OF STRUCTURES ******
1 0.80 ****** 1:1 2-3
2 0.20 ****** 1:1 2-2
Lowdin Weights
1 0.30 ****** 1:1 2-3
2 0.70 ****** 1:1 2-2
""")
    return source


def test_workflow_reuses_weight_ranking_without_writing_xyz_beside_input(xmo_file, monkeypatch):
    captured = {}
    original = xmo_file.read_bytes()
    def render(scene, output, **options):
        captured.update(scene=scene, output=output, options=options)
    monkeypatch.setattr(workflow, "render_pyvista_scene", render)
    result = workflow.xmo2png_file(xmo_file, max_structures=1, weight_table="both")
    assert result.structure_count == 1 and result.displayed_atom_count == 2
    assert result.output_file == xmo_file.with_name("ethene_3d.png")
    structure = captured["scene"].structures[0]
    assert structure.bond_pairs == [(1, 1)]
    assert "CC 2" in structure.legend and "Löwdin 2" in structure.legend
    assert not xmo_file.with_suffix(".xyz").exists()
    assert xmo_file.read_bytes() == original


@pytest.mark.parametrize("options", [dict(max_structures=0), dict(structures_per_row=0),
                                         dict(panel_size=64), dict(output=Path("invalid.svg")),
                                         dict(color_scheme="invalid"), dict(projection="rdkit")])
def test_invalid_output_options_fail_before_rendering(xmo_file, options):
    with pytest.raises(ValueError):
        workflow.xmo2png_file(xmo_file, **options)


@pytest.mark.parametrize("reference,pairs,radicals,expected", [
    ({1: 1, 2: 1}, [(1, 2)], [], {}),
    ({1: 1, 2: 1}, [(1, 1)], [], {0: "-", 1: "+"}),
    ({1: 1, 2: 1}, [(2, 2)], [], {0: "+", 1: "-"}),
    ({1: 2, 2: 2}, [(1, 1), (1, 1)], [], {0: "2-", 1: "2+"}),
    ({1: 1, 2: 1, 3: 1}, [(1, 3)], [2], {}),
    ({1: 1, 2: 1, 3: 1}, [(1, 1)], [3], {0: "-", 1: "+"}),
])
def test_scene_charges_match_svg_bookkeeping(reference, pairs, radicals, expected, tmp_path):
    from autoVB.draw_xmo.molecule_bond_variant_drawer import MoleculeBondVariantDrawer

    parsed = SimpleNamespace(
        geo=[XmoGeometryAtom("C", i, 0, 0) for i in range(3)], orb=[[1, 2], [2, 3]],
    )
    structure = ValenceBondStructureInfo("1", "test", pairs, radicals)
    scene = prepare_pyvista_scene(parsed, [structure], baseline_electron_counts=reference)
    drawer = MoleculeBondVariantDrawer(
        xyz_file=tmp_path / "unused.xyz", output_dir=tmp_path,
        active_bond_atom=[[atom for atom, count in reference.items() for _ in range(count)]],
        active_space=[],
    )
    assert scene.charge_notes == [expected]
    assert drawer._charge_notes_from_valence_structure(structure) == expected


def test_positive_reference_hydrogen_is_retained_when_hiding_hydrogens():
    parsed = SimpleNamespace(
        geo=[XmoGeometryAtom("H", 0, 0, 0), XmoGeometryAtom("C", 1, 0, 0)], orb=[[1, 2]],
    )
    structure = ValenceBondStructureInfo("1", "ionic", [(2, 2)])
    scene = prepare_pyvista_scene(parsed, [structure], show_hydrogens=False,
                                  baseline_electron_counts={1: 1, 2: 1})
    assert scene.visible_atoms == [0, 1]
    assert scene.charge_notes == [{0: "+", 1: "-"}]


def test_reference_is_independent_of_weight_ranking(xmo_file, monkeypatch):
    captured = {}
    monkeypatch.setattr(workflow, "render_pyvista_scene",
                        lambda scene, output, **options: captured.update(scene=scene, options=options))
    workflow.xmo2png_file(xmo_file, max_structures=2, weight_table="both", baseline_index=1)
    assert captured["scene"].charge_notes == [{0: "-", 1: "+"}, {}]
    assert captured["options"]["show_charges"] is True


@pytest.mark.parametrize("color_scheme,framework_color,pair_color", [
    ("element", "#394352", "#BB1938"), ("highlight", "#C8CDD3", "#FF3030"),
])
def test_charge_labels_are_independent_of_atom_labels(tmp_path, monkeypatch, color_scheme, framework_color, pair_color):
    from autoVB.draw_xmo import pyvista_drawer

    annotations = Mock()
    monkeypatch.setattr(pyvista_drawer, "add_png_charge_labels", annotations)
    plotter = Mock()
    monkeypatch.setitem(sys.modules, "pyvista", SimpleNamespace(
        Plotter=Mock(return_value=plotter), Sphere=Mock(), Cylinder=Mock(),
    ))
    parsed = SimpleNamespace(
        geo=[XmoGeometryAtom("C", 0, 0, 0), XmoGeometryAtom("C", 1.4, 0, 0)], orb=[[1, 2]],
    )
    structure = ValenceBondStructureInfo("1", "ionic", [(1, 1)])
    scene = prepare_pyvista_scene(parsed, [structure], baseline_electron_counts={1: 1, 2: 1})
    render_pyvista_scene(scene, tmp_path / "charged.png", show_atom_labels=False,
                         color_scheme=color_scheme, show_legend=False)
    colors = [call.kwargs["color"] for call in plotter.add_mesh.call_args_list]
    assert colors[:2] == [framework_color, framework_color]
    assert colors[2:4] == ["#394352", "#394352"]  # Atom colors are preserved.
    assert colors[4:] == [pair_color, pair_color]
    assert annotations.call_count == 1
    args, kwargs = annotations.call_args
    assert args[2] == ["-", "+"]
    assert kwargs["color"] == pair_color
    assert kwargs["circle"] is False
    assert np.isfinite(args[1]).all()
    annotations.reset_mock()
    plotter.reset_mock()
    render_pyvista_scene(scene, tmp_path / "hidden.png", show_charges=False, show_legend=False)
    plotter.add_point_labels.assert_not_called()
    annotations.assert_not_called()


@pytest.mark.parametrize("projection", ["pca", "optimized3d"])
def test_cli_passes_projection(xmo_file, projection, monkeypatch):
    from autoVB.cli import xmo2png as cli

    captured = {}
    def draw(source, output, **options):
        captured.update(options)
        return SimpleNamespace(output_file=xmo_file.with_suffix(".png"), structure_count=2,
                               displayed_atom_count=2, weight_table="both")
    monkeypatch.setattr(cli, "xmo2png_file", draw)
    cli.xmo2png([str(xmo_file), "--projection", projection])
    assert captured["projection"] == projection
    assert captured["azimuth"] is None and captured["elevation"] is None


@pytest.fixture
def nonplanar_scene():
    xyz = [(0, 0, 0), (1.4, 0, 0), (1.4, 1.4, 0), (0, 1.4, 0.8), (0.5, 0.5, 1.8)]
    parsed = SimpleNamespace(geo=[XmoGeometryAtom("C", *p) for p in xyz],
                             orb=[[1, 2], [2, 3], [3, 4], [4, 1], [4, 5]])
    structures = [ValenceBondStructureInfo("1", "covalent", [(1, 2), (3, 4)], [5]),
                  ValenceBondStructureInfo("2", "ionic", [(1, 1), (3, 4)], [5])]
    return prepare_pyvista_scene(parsed, structures)


def test_optimized_camera_is_finite_repeatable_and_keeps_geometry(nonplanar_scene):
    scene = nonplanar_scene
    original = scene.coordinates.copy()
    camera, scale, normal, tangent = scene_camera(scene, projection="optimized3d")
    assert np.isfinite(camera).all() and scale > 0
    np.testing.assert_array_equal(scene.coordinates, original)
    np.testing.assert_allclose(camera, scene_camera(scene, projection="optimized3d")[0])
    np.testing.assert_allclose(normal, scene_camera(scene, projection="pca")[2])
    np.testing.assert_allclose(tangent, scene_camera(scene, projection="pca")[3])


@pytest.mark.parametrize("angles", [dict(azimuth=0), dict(elevation=90),
                                     dict(azimuth=40, elevation=60)])
def test_explicit_angles_override_optimized_search(nonplanar_scene, angles, monkeypatch):
    from autoVB.draw_xmo import pyvista_drawer

    search = Mock(side_effect=AssertionError("manual view must bypass search"))
    monkeypatch.setattr(pyvista_drawer, "_optimized_view_direction", search)
    expected = scene_camera(nonplanar_scene, projection="pca", **angles)[0]
    actual = scene_camera(nonplanar_scene, projection="optimized3d", **angles)[0]
    np.testing.assert_allclose(actual, expected)
    search.assert_not_called()


def test_default_camera_retains_old_angles_and_rejects_invalid_values(nonplanar_scene):
    np.testing.assert_allclose(scene_camera(nonplanar_scene)[0],
                               scene_camera(nonplanar_scene, 35, 55)[0])
    with pytest.raises(ValueError, match="projection"):
        scene_camera(nonplanar_scene, projection="rdkit")
    with pytest.raises(ValueError, match="finite"):
        scene_camera(nonplanar_scene, azimuth=float("nan"), projection="optimized3d")


def test_optimized_camera_handles_single_atom():
    parsed = SimpleNamespace(geo=[XmoGeometryAtom("He", 0, 0, 0)], orb=[[1]])
    scene = prepare_pyvista_scene(parsed, [ValenceBondStructureInfo("1", "pair", [(1, 1)])])
    camera, scale, *_ = scene_camera(scene, projection="optimized3d")
    assert np.isfinite(camera).all() and scale > 0


def test_optimized_camera_separates_atoms_overlapping_in_default_view():
    from itertools import product

    azimuth, elevation = np.deg2rad([35, 55])
    direction = np.array([np.cos(azimuth) * np.cos(elevation),
                          np.sin(azimuth) * np.cos(elevation), np.sin(elevation)])
    # Opposite corners overlap in the default view; distant anchors fix PCA axes.
    points = [direction * signs for signs in product([-1, 1], repeat=3)]
    points.extend([[10, 0, 0], [-10, 0, 0], [0, 8, 0], [0, -8, 0]])
    parsed = SimpleNamespace(geo=[XmoGeometryAtom("C", *p) for p in points], orb=[[9, 10]])
    scene = prepare_pyvista_scene(parsed, [ValenceBondStructureInfo("1", "test", [(9, 10)])])

    def minimum_separation(mode):
        camera, *_ = scene_camera(scene, projection=mode)
        direction = np.asarray(camera[0]) - camera[1]
        direction /= np.linalg.norm(direction)
        delta = scene.coordinates[:8, None, :] - scene.coordinates[None, :8, :]
        projected = delta - np.einsum("ijk,k->ij", delta, direction)[:, :, None] * direction
        return np.linalg.norm(projected, axis=2)[np.triu_indices(8, 1)].min()

    assert minimum_separation("pca") < 1e-6
    assert minimum_separation("optimized3d") > 0.1


@pytest.mark.parametrize("color_scheme,pair_color,circle", [
    ("element", "#BB1938", False), ("highlight", "#FF3030", True),
])
def test_html_output_uses_exporter_and_retains_all_annotations(xmo_file, tmp_path, monkeypatch, color_scheme, pair_color, circle):
    from autoVB.draw_xmo import html_export

    plotter = Mock()
    monkeypatch.setitem(sys.modules, "pyvista", SimpleNamespace(
        Plotter=Mock(return_value=plotter), Sphere=Mock(), Cylinder=Mock(),
    ))
    monkeypatch.setitem(sys.modules, "trame_pyvista", Mock())
    monkeypatch.setitem(sys.modules, "nest_asyncio2", Mock())
    captured = {}
    def export(plotter, output, panels, *, atom_legend):
        captured.update(panels=panels, atom_legend=atom_legend)
        output.write_text("<html>interactive</html>")
    monkeypatch.setattr(html_export, "export_interactive_html", export)
    output = tmp_path / "interactive.html"
    result = workflow.xmo2png_file(xmo_file, output=output, max_structures=2,
                                    baseline_index=1, show_atom_labels=True, color_scheme=color_scheme,
                                    circle_charges=circle)
    assert result.output_file == output and output.exists()
    assert captured["atom_legend"] == [dict(symbol="C", color="#394352")]
    panels = captured["panels"]
    assert len(panels) == 2
    assert "Löwdin 2" in panels[0]["legend"] and "CC 2" in panels[0]["legend"]
    assert [label["text"] for label in panels[0]["labels"]] == ["C1", "C2", "−", "+"]
    assert [label["color"] for label in panels[0]["labels"] if label["kind"] == "charge"] == [pair_color, pair_color]
    assert [label["circle"] for label in panels[0]["labels"] if label["kind"] == "charge"] == [circle, circle]
    assert panels[0]["viewport"] == [0, 0, 0.5, 1]
    assert panels[1]["viewport"] == [0.5, 0, 1, 1]
    assert panels[0]["camera"]["parallelProjection"] is True
    assert panels[0]["camera"]["parallelScale"] > 0
    plotter.screenshot.assert_not_called()
    plotter.close.assert_called_once()


def test_html_dependency_error_and_transparency_rejection(nonplanar_scene, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "trame_pyvista", None)
    with pytest.raises(ImportError, match="visualization-html"):
        render_pyvista_scene(nonplanar_scene, tmp_path / "missing.html")
    with pytest.raises(ValueError, match="only supported for PNG"):
        render_pyvista_scene(nonplanar_scene, tmp_path / "transparent.html", transparent_background=True)


def test_html_annotation_payload_preserves_unicode_and_escapes_script(tmp_path):
    import io
    import json
    import re
    from autoVB.draw_xmo.html_export import export_interactive_html

    plotter = Mock()
    plotter.trame.export_html.return_value = io.StringIO("<html><body>VTK scene</body></html>")
    panels = [dict(rank=1, legend="Löwdin </script><script>alert(1)</script>",
                   viewport=[0, 0, 1, 1], labels=[dict(text="−", kind="charge", position=[0, 0, 0])])]
    output = tmp_path / "safe.html"
    export_interactive_html(plotter, output, panels)
    document = output.read_text()
    payload = re.search(r'<script id="autovb-panels" type="application/json">(.*?)</script>',document).group(1)
    assert json.loads(payload) == panels
    assert "<script>alert(1)</script>" not in document
    assert '<div class="content" id="autovb-scene"' in document
