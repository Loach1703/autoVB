"""Public CLI routing, validation and real SVG/shared preparation regressions."""

from types import SimpleNamespace
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from autoVB.cli.draw_xmo import draw_xmo
from autoVB.cli.xmo2html import xmo2html
from autoVB.cli.xmo2png import xmo2png
from autoVB.cli.xmo2svg import xmo2svg
from autoVB.cli.xmoview import xmoview


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "ethene.xmo"
    path.write_text("""$ctrl
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
    return path


@pytest.fixture
def routed(monkeypatch):
    from autoVB.cli import xmo2png as png, xmo2svg as svg

    calls = []
    def render(source, output=None, **options):
        calls.append(dict(source=source, output=output, **options))
        return SimpleNamespace(output_file=output, structure_count=2,
                               displayed_atom_count=2, weight_table=options["weight_table"])
    monkeypatch.setattr(svg, "xmo2svg_file", render)
    monkeypatch.setattr(svg, "xmo2svg_report_lines", lambda result: [])
    monkeypatch.setattr(png, "xmo2png_file", render)
    return calls


@pytest.mark.parametrize("command,format", [
    (xmoview, "svg"), (draw_xmo, "svg"), (xmo2svg, "svg"), (xmo2png, "png"),
    (xmo2html, "html"),
])
def test_aliases_share_defaults(command, format, source, routed):
    assert command([str(source)]) == 0
    call, = routed
    assert call["output"] == source.with_suffix("." + format)
    assert call["weight_table"] == "both"
    assert call["max_structures"] == 6 and call["structures_per_row"] == 3
    assert not call["show_atom_labels"] and not call["show_connection_labels"]
    assert call["show_charges"] and call["show_lone_pairs"]
    assert call["circle_charges"] is False
    if format == "svg":
        assert call["hide_hydrogens"] is True
    else:
        assert call["show_hydrogens"] is True
        assert call["color_scheme"] == "element"
        assert call["show_legend"] is True


@pytest.mark.parametrize("format", ["svg", "png", "html"])
def test_explicit_format_selects_backend(format, source, routed):
    xmoview([str(source), "-f", format])
    assert routed[0]["output"] == source.with_suffix("." + format)
    if format == "svg":
        assert routed[0]["hide_hydrogens"] is True
    else:
        assert routed[0]["show_hydrogens"] is True


@pytest.mark.parametrize("suffix", ["svg", "PNG", "html"])
def test_output_path_infers_format(source, routed, tmp_path, suffix):
    output = tmp_path / "figures" / ("custom." + suffix)
    xmoview([str(source), "-o", str(output)])
    assert routed[0]["output"] == output
    if suffix.lower() == "svg":
        assert routed[0]["hide_hydrogens"] is True
    else:
        assert routed[0]["show_hydrogens"] is True


@pytest.mark.parametrize("options", [
    ["-f", "png", "-o", "wrong.svg"], ["-o", "wrong.pdf"],
    ["-f", "svg", "--view", "pca"],
    ["-f", "svg", "--hide-legend"],
    ["-f", "svg", "--connectivity", "rdkit", "--layout", "contact"],
    ["-f", "svg", "--layout", "pca", "--projection", "contact"],
    ["-f", "png", "--layout", "pca"], ["-f", "png", "--charge", "0"],
    ["-f", "png", "--write-individual"],
    ["-f", "png", "--projection", "contact"],
    ["-f", "png", "--view", "pca", "--projection", "optimized3d"],
    ["-f", "html", "-n", "3"],
    ["-f", "html", "--transparent-background"],
    ["-m", "0"], ["-n", "0"], ["--panel-size", "64"], ["--baseline-index", "0"],
    ["--indices", "1,1"], ["--indices", "0"], ["--indices", "a"],
    ["--indices", "1", "-m", "all"],
])
def test_invalid_combinations_fail_before_rendering(source, routed, options):
    with pytest.raises(SystemExit) as error:
        xmoview([str(source), *options])
    assert error.value.code == 2 and routed == []


@pytest.mark.parametrize("format,key,hydrogen_option,show_hydrogens", [
    ("svg", "hide_hydrogens", "--show-hydrogens", True),
    ("png", "show_hydrogens", "--hide-hydrogens", False),
    ("html", "show_hydrogens", "--hide-hydrogens", False),
])
def test_common_display_controls(source, routed, format, key, hydrogen_option, show_hydrogens):
    xmoview([str(source), "-f", format, "--show-atom-labels", "--show-connection-labels",
             hydrogen_option, "--hide-lone-pairs", "--hide-charges", "--circle-charges"])
    call, = routed
    assert call["show_atom_labels"] and call["show_connection_labels"]
    assert call[key] == (not show_hydrogens if key == "hide_hydrogens" else show_hydrogens)
    assert not call["show_charges"] and not call["show_lone_pairs"]
    assert call["circle_charges"] is True


def test_three_d_view_and_manual_angles(source, routed):
    xmoview([str(source), "-f", "png", "--view", "optimized3d", "--elevation", "30",
             "--color-scheme", "highlight", "--hide-legend"])
    assert routed[0]["projection"] == "optimized3d"
    assert routed[0]["elevation"] == 30 and routed[0]["azimuth"] is None
    assert routed[0]["color_scheme"] == "highlight"
    assert routed[0]["show_legend"] is False


def test_real_svg_custom_output_and_transparency(source, tmp_path):
    output = tmp_path / "figures" / "custom.svg"
    assert xmoview([str(source), "-o", str(output), "-m", "all", "--panel-size", "300",
                    "--transparent-background", "--hide-charges"]) == 0
    svg = output.read_text()
    root = ET.fromstring(svg)
    assert root.attrib["height"] == "300px"
    assert "charge-label" not in svg and "fill:#FFFFFF" not in svg
    assert not source.with_suffix(".svg").exists()
    assert not (output.parent / "ethene_grid.svg").exists()


@pytest.mark.parametrize("format", ["svg", "png"])
def test_explicit_selection_keeps_order_and_reference(source, monkeypatch, format):
    from autoVB.vbkit import xmo2png as png
    from autoVB.draw_xmo.molecule_bond_variant_drawer import MoleculeBondVariantDrawer

    captured = []
    if format == "svg":
        original_draw = MoleculeBondVariantDrawer.draw
        def draw(self):
            captured.extend(self.active_space)
            assert self.active_bond_atom == [[1, 1]]
            return original_draw(self)
        monkeypatch.setattr(MoleculeBondVariantDrawer, "draw", draw)
    else:
        def render(scene, output, **options):
            captured.extend(scene.structures)
            assert scene.charge_notes == [{0: "+", 1: "-"}, {}]
        monkeypatch.setattr(png, "render_pyvista_scene", render)
    xmoview([str(source), "-f", format, "--indices", "1,2"])
    assert "CC 1" in captured[0].legend and "CC 2" in captured[1].legend
    if format == "svg":
        root = ET.parse(source.with_suffix(".svg")).getroot()
        labels = root.findall(".//{http://www.w3.org/2000/svg}text")
        assert "−" in [label.text for label in labels]


@pytest.mark.parametrize("format", ["svg", "png"])
def test_unknown_structure_index_fails(source, format):
    with pytest.raises(ValueError, match="Cannot find.*999"):
        xmoview([str(source), "-f", format, "--indices", "999"])


def test_all_structures_sorted_by_weight_in_three_d(source, monkeypatch):
    from autoVB.vbkit import xmo2png as png
    captured = []
    monkeypatch.setattr(png, "render_pyvista_scene", lambda scene, output, **kw: captured.extend(scene.structures))
    xmoview([str(source), "-f", "png", "-m", "all"])
    assert "CC 2" in captured[0].legend and "CC 1" in captured[1].legend


def test_help_does_not_import_pyvista():
    code = """import sys
from autoVB.cli.xmoview import xmoview
try:
    xmoview(['--help'])
except SystemExit as e:
    assert e.code == 0
assert 'pyvista' not in sys.modules
assert 'autoVB.draw_xmo.pyvista_drawer' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
