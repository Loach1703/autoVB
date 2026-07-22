from rdkit import Chem

from autoVB.cli.xmo2svg import xmo2svg
from autoVB.draw_xmo.orbital_connectivity_molecule_drawer import (
    OrbitalConnectivityMoleculeDrawer,
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


def test_xmo2svg_generates_grid_from_orbital_connectivity(tmp_path):
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

    assert xmo2svg([str(xmo_path)]) == 0

    grid_path = tmp_path / "ethene_grid.svg"
    assert grid_path.exists()
    assert "<svg" in grid_path.read_text(encoding="utf-8")
