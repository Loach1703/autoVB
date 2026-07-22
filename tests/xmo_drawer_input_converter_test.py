from rdkit import Chem

from autoVB.draw_xmo.molecule_bond_variant_drawer import MoleculeBondVariantDrawer
from autoVB.draw_xmo.molecule_bond_variant_drawer import ValenceBondStructureInfo
from autoVB.draw_xmo.xmo_drawer_input_converter import XmoToDrawerInputConverter
from autoVB.io.xmo_output_parser import XmoParser


def write_xmo(tmp_path, text: str):
    xmo_path = tmp_path / "sample.xmo"
    xmo_path.write_text(text.strip() + "\n", encoding="utf-8")
    return xmo_path


def write_xyz(tmp_path, name: str, text: str):
    xyz_path = tmp_path / name
    xyz_path.write_text(text.strip() + "\n", encoding="utf-8")
    return xyz_path


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
    assert drawer_input.baseline_unpaired_atoms == [1]
    assert drawer_input.orbital_to_atom == {1: 1, 2: 3, 3: 5}


def test_converter_uses_highest_weight_baseline_by_default(tmp_path):
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


def test_unpaired_electron_movement_does_not_create_charge_notes(tmp_path):
    drawer = MoleculeBondVariantDrawer(
        xyz_file=tmp_path / "dummy.xyz",
        output_dir=tmp_path,
        active_bond_atom=[[2, 3, 5, 4]],
        baseline_unpaired_atoms=[1],
        active_space=[],
    )
    structure = ValenceBondStructureInfo(
        file_suffix="moved-radical",
        legend="moved radical",
        bond_pairs=[(1, 2), (3, 5)],
        unpaired_atoms=[4],
    )

    assert drawer._charge_notes_from_valence_structure(structure) == {}


def test_c5h5_uses_clean_connectivity_without_breaking_ring(tmp_path):
    xyz_path = write_xyz(
        tmp_path,
        "c5h5.xyz",
        """
10
C5H5 radical
C    0.000000000  -1.188906820   0.000000000
H    0.028216860  -2.271094080   0.000000000
C    1.173670220  -0.318446820   0.000000000
H    2.199403080  -0.657181880   0.000000000
C    0.730543480   0.965374770   0.000000000
H    1.330235910   1.864439300   0.000000000
C   -1.159644260  -0.390925350   0.000000000
H   -2.178784100  -0.746796070   0.000000000
C   -0.744078340   0.933422120   0.000000000
H   -1.382018380   1.807525230   0.000000000
""",
    )
    structure = ValenceBondStructureInfo(
        file_suffix="covalent",
        legend="covalent",
        bond_pairs=[(2, 3), (5, 4)],
        unpaired_atoms=[1],
    )
    drawer = MoleculeBondVariantDrawer(
        xyz_file=xyz_path,
        output_dir=tmp_path,
        charge=0,
        active_bond_atom=[[2, 3, 5, 4]],
        baseline_unpaired_atoms=[1],
        active_space=[structure],
    )

    base_mol = drawer.build_base_molecule()

    assert drawer.bond_perception_mode == "connectivity_only"
    assert sum(atom.GetFormalCharge() for atom in base_mol.GetAtoms()) == 0
    assert sum(atom.GetNumRadicalElectrons() for atom in base_mol.GetAtoms()) == 0
    assert base_mol.GetNumBonds() == 5

    variant, _ = drawer.apply_variant(base_mol, structure)
    bond_types = [bond.GetBondType() for bond in variant.GetBonds()]

    assert variant.GetNumBonds() == 5
    assert bond_types.count(Chem.BondType.DOUBLE) == 2
    assert bond_types.count(Chem.BondType.SINGLE) == 3
    assert drawer._charge_notes_from_mol(variant) == {}
    assert drawer._radical_counts_from_mol(variant) == {0: 1}


def test_closed_shell_molecule_keeps_bond_order_perception(tmp_path):
    xyz_path = write_xyz(
        tmp_path,
        "ethene.xyz",
        """
6
ethene
C -0.6700  0.0000 0.0000
C  0.6700  0.0000 0.0000
H -1.2300  0.9200 0.0000
H -1.2300 -0.9200 0.0000
H  1.2300  0.9200 0.0000
H  1.2300 -0.9200 0.0000
""",
    )
    drawer = MoleculeBondVariantDrawer(
        xyz_file=xyz_path,
        output_dir=tmp_path,
        charge=0,
        active_bond_atom=[],
        baseline_unpaired_atoms=[],
        active_space=[],
    )

    base_mol = drawer.build_base_molecule()

    assert drawer.bond_perception_mode == "bond_order"
    assert base_mol.GetNumAtoms() == 2
    assert base_mol.GetNumBonds() == 1
    assert base_mol.GetBondWithIdx(0).GetBondType() == Chem.BondType.DOUBLE
