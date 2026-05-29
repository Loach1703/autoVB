from autoVB.draw_xmo.molecule_bond_variant_drawer import MoleculeBondVariantDrawer
from autoVB.draw_xmo.xmo_drawer_input_converter import XmoToDrawerInputConverter
from autoVB.io.xmo_output_parser import XmoParser


def write_xmo(tmp_path, text: str):
    xmo_path = tmp_path / "sample.xmo"
    xmo_path.write_text(text.strip() + "\n", encoding="utf-8")
    return xmo_path


def test_converter_uses_parser_structure_connections_for_doublet(tmp_path):
    xmo_path = write_xmo(
        tmp_path,
        """
$ctrl
vbscf
nae=3
nao=3
nmul=2
basis=6-31g*
$end

$orb
1*3
1
3
5
$end

$geo
C 0.0 0.0 0.0
C 1.0 0.0 0.0
C 2.0 0.0 0.0
C 3.0 0.0 0.0
C 4.0 0.0 0.0
$end

******  WEIGHTS OF STRUCTURES ******
1 0.50 ****** 2 3 1
""",
    )

    parsed = XmoParser(xmo_path).parse()
    converter = XmoToDrawerInputConverter(
        parsed,
        tmp_path,
        hide_hydrogens=False,
    )

    drawer_input = converter.convert()

    assert parsed.cc_weights[0].atom_connections == [(3, 5)]
    assert parsed.cc_weights[0].unpaired_atoms == [1]
    assert drawer_input.active_space[0].bond_pairs == [(3, 5)]
    assert drawer_input.active_space[0].unpaired_atoms == [1]
    assert drawer_input.active_bond_atom == [[3, 5]]
    assert drawer_input.orbital_to_atom == {1: 1, 2: 3, 3: 5}


def test_converter_falls_back_to_highest_weight_baseline(tmp_path):
    xmo_path = write_xmo(
        tmp_path,
        """
$ctrl
vbscf
nae=2
nao=2
basis=cc-pVDZ
$end

$orb
1*2
1
2
$end

$geo
C 0.0 0.0 0.0
C 1.0 0.0 0.0
$end

******  WEIGHTS OF STRUCTURES ******
1 0.10 ****** 1-2
2 0.90 ****** 1 1
""",
    )

    parsed = XmoParser(xmo_path).parse()
    converter = XmoToDrawerInputConverter(
        parsed,
        tmp_path,
        baseline_index=99,
        hide_hydrogens=False,
    )

    drawer_input = converter.convert()

    assert drawer_input.active_bond_atom == [[1, 1]]


def test_drawer_renders_unpaired_electron_as_radical_dot(tmp_path):
    drawer = MoleculeBondVariantDrawer(
        xyz_file=tmp_path / "dummy.xyz",
        output_dir=tmp_path,
        active_bond_atom=[],
        active_space=[],
    )

    svg = drawer._add_svg_annotations(
        "<svg></svg>",
        atom_coords=[(50.0, 50.0)],
        charge_notes={},
        lone_pair_counts={},
        radical_counts={0: 1},
        width=100,
        height=100,
    )

    assert "radical-dot" in svg
    assert "fill='#E00000'" in svg
