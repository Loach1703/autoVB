import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from autoVB.cli.xmo2svg import xmo2svg
from autoVB.draw_xmo.orbital_connectivity_molecule_drawer import (
    OrbitalConnectivityMoleculeDrawer,
)
from autoVB.draw_xmo.molecule_bond_variant_drawer import (
    ValenceBondStructureInfo,
)


def test_orbital_connectivity_keeps_close_molecules_separate(tmp_path):
    xyz_path = tmp_path / "two_h2.xyz"
    xyz_path.write_text(
        """4
two close hydrogen molecules
H 0.00 0.0 0.0
H 0.74 0.0 0.0
H 1.64 0.0 0.0
H 2.38 0.0 0.0
""",
        encoding="utf-8",
    )
    drawer = OrbitalConnectivityMoleculeDrawer(
        xyz_file=xyz_path,
        output_dir=tmp_path,
        charge=0,
        orbital_atom_rows=[[1, 2], [3, 4]],
        active_bond_atom=[],
        active_space=[],
        baseline_unpaired_atoms=[],
        hide_hydrogens=False,
    )

    mol = drawer.build_base_molecule()

    assert Chem.GetMolFrags(mol) == ((0, 1), (2, 3))
    assert mol.GetBondBetweenAtoms(1, 2) is None
    assert sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) == 0
    assert sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()) == 0


def test_pca_projection_preserves_relative_fragment_distances(tmp_path):
    xyz_path = tmp_path / "two_h2.xyz"
    xyz_path.write_text(
        """4
two close hydrogen molecules
H 0.00 0.0 0.0
H 0.74 0.0 0.0
H 1.64 0.0 0.0
H 2.38 0.0 0.0
""",
        encoding="utf-8",
    )
    drawer = OrbitalConnectivityMoleculeDrawer(
        xyz_file=xyz_path,
        output_dir=tmp_path,
        charge=0,
        orbital_atom_rows=[[1, 2], [3, 4]],
        active_bond_atom=[],
        active_space=[],
        baseline_unpaired_atoms=[],
        hide_hydrogens=False,
        projection="pca",
    )

    mol = drawer.build_base_molecule()
    conformer = mol.GetConformer()

    assert conformer.Is3D() is False
    assert conformer.GetAtomPosition(0).Distance(
        conformer.GetAtomPosition(1)
    ) == pytest.approx(0.74)
    assert conformer.GetAtomPosition(1).Distance(
        conformer.GetAtomPosition(2)
    ) == pytest.approx(0.90)


def test_pca_projection_avoids_collapsing_bond_along_third_axis():
    editable_mol = Chem.RWMol()
    for _ in range(6):
        editable_mol.AddAtom(Chem.Atom("C"))
    editable_mol.AddBond(0, 1, Chem.BondType.SINGLE)
    mol = editable_mol.GetMol()
    conformer = Chem.Conformer(6)
    for atom_index, coordinates in enumerate(
        (
            (0.0, 0.0, -1.0),
            (0.0, 0.0, 1.0),
            (-5.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
            (0.0, -3.0, 0.0),
            (0.0, 3.0, 0.0),
        )
    ):
        conformer.SetAtomPosition(atom_index, Point3D(*coordinates))
    mol.AddConformer(conformer)

    OrbitalConnectivityMoleculeDrawer._apply_pca_projection(mol)

    projected_conformer = mol.GetConformer()
    assert projected_conformer.GetAtomPosition(0).Distance(
        projected_conformer.GetAtomPosition(1)
    ) == pytest.approx(2.0)


def test_contact_projection_uses_rdkit_for_one_fragment(tmp_path):
    xyz_path = tmp_path / "h2.xyz"
    xyz_path.write_text(
        """2
hydrogen
H 0.00 0.0 0.0
H 0.74 0.0 0.0
""",
        encoding="utf-8",
    )
    drawer = OrbitalConnectivityMoleculeDrawer(
        xyz_file=xyz_path,
        output_dir=tmp_path,
        charge=0,
        orbital_atom_rows=[[1, 2]],
        active_bond_atom=[],
        active_space=[],
        baseline_unpaired_atoms=[],
        hide_hydrogens=False,
        projection="contact",
    )

    mol = drawer.build_base_molecule()
    conformer = mol.GetConformer()

    assert conformer.Is3D() is False
    assert conformer.GetAtomPosition(0).Distance(
        conformer.GetAtomPosition(1)
    ) == pytest.approx(1.5)


def test_contact_projection_places_nearest_fragment_atoms_facing(tmp_path):
    xyz_path = tmp_path / "two_h2.xyz"
    xyz_path.write_text(
        """4
two close hydrogen molecules
H 0.00 0.0 0.0
H 0.74 0.0 0.0
H 1.64 0.0 0.0
H 2.38 0.0 0.0
""",
        encoding="utf-8",
    )
    drawer = OrbitalConnectivityMoleculeDrawer(
        xyz_file=xyz_path,
        output_dir=tmp_path,
        charge=0,
        orbital_atom_rows=[[1, 2], [3, 4]],
        active_bond_atom=[],
        active_space=[],
        baseline_unpaired_atoms=[],
        hide_hydrogens=False,
        projection="contact",
    )

    mol = drawer.build_base_molecule()
    conformer = mol.GetConformer()
    positions = [
        conformer.GetAtomPosition(atom_index)
        for atom_index in range(mol.GetNumAtoms())
    ]

    assert positions[0].Distance(positions[1]) == pytest.approx(1.5)
    assert positions[2].Distance(positions[3]) == pytest.approx(1.5)
    assert positions[1].Distance(positions[2]) == pytest.approx(1.5)
    first_fragment_center = Point3D(
        (positions[0].x + positions[1].x) / 2,
        (positions[0].y + positions[1].y) / 2,
        0.0,
    )
    second_fragment_center = Point3D(
        (positions[2].x + positions[3].x) / 2,
        (positions[2].y + positions[3].y) / 2,
        0.0,
    )
    assert positions[1].Distance(second_fragment_center) < positions[0].Distance(
        second_fragment_center
    )
    assert positions[2].Distance(first_fragment_center) < positions[3].Distance(
        first_fragment_center
    )


def test_contact_projection_aligns_multiple_active_bonds(tmp_path):
    xyz_path = tmp_path / "two_h2.xyz"
    xyz_path.write_text(
        """4
two hydrogen fragments with two active contacts
H 0.00 0.0 0.0
H 0.74 0.0 0.0
H 1.64 0.0 0.0
H 2.38 0.0 0.0
""",
        encoding="utf-8",
    )
    drawer = OrbitalConnectivityMoleculeDrawer(
        xyz_file=xyz_path,
        output_dir=tmp_path,
        charge=0,
        orbital_atom_rows=[[1, 2], [3, 4]],
        active_bond_atom=[],
        active_space=[
            ValenceBondStructureInfo(
                file_suffix="1",
                legend="",
                bond_pairs=[(1, 3), (2, 4)],
            )
        ],
        baseline_unpaired_atoms=[],
        hide_hydrogens=False,
        projection="contact",
    )

    mol = drawer.build_base_molecule()
    conformer = mol.GetConformer()

    assert conformer.GetAtomPosition(0).Distance(
        conformer.GetAtomPosition(2)
    ) == pytest.approx(1.5)
    assert conformer.GetAtomPosition(1).Distance(
        conformer.GetAtomPosition(3)
    ) == pytest.approx(1.5)


def test_contact_projection_uses_bonds_from_all_displayed_structures(tmp_path):
    drawer = OrbitalConnectivityMoleculeDrawer(
        xyz_file=tmp_path / "unused.xyz",
        output_dir=tmp_path,
        charge=0,
        orbital_atom_rows=[],
        active_bond_atom=[],
        active_space=[
            ValenceBondStructureInfo(
                file_suffix="1",
                legend="",
                bond_pairs=[(1, 3), (2, 4)],
            ),
            ValenceBondStructureInfo(
                file_suffix="2",
                legend="",
                bond_pairs=[(1, 3), (2, 3)],
            ),
        ],
        baseline_unpaired_atoms=[],
        hide_hydrogens=False,
        projection="contact",
    )

    assert drawer._active_contact_pairs({0, 1}, {2, 3}) == [
        (0, 2),
        (1, 3),
        (0, 2),
        (1, 2),
    ]


@pytest.mark.parametrize("projection", ["pca", "contact"])
def test_xmo2svg_generates_grid_from_orbital_connectivity(
    tmp_path,
    projection,
):
    xmo_path = tmp_path / "ethene.xmo"
    xmo_path.write_text(
        """$ctrl
vbscf
nae=2
nao=2
ncharge=0
nmul=1
basis=6-31g
$end

$orb
2*5 1*2
1 2
1 3
1 4
2 5
2 6
1
2
$end

$geo
C -0.6700  0.0000 0.0000
C  0.6700  0.0000 0.0000
H -1.2300  0.9200 0.0000
H -1.2300 -0.9200 0.0000
H  1.2300  0.9200 0.0000
H  1.2300 -0.9200 0.0000
$end

******  WEIGHTS OF STRUCTURES ******
1 1.00 ****** 1:5 6-7

Lowdin Weights
1 1.00 ****** 1:5 6-7
""",
        encoding="utf-8",
    )

    assert xmo2svg([str(xmo_path), "--projection", projection]) == 0

    grid_path = tmp_path / "ethene_grid.svg"
    assert grid_path.exists()
    assert "<svg" in grid_path.read_text(encoding="utf-8")
