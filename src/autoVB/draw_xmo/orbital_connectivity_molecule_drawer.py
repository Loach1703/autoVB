from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdDepictor, rdDetermineBonds

from .molecule_bond_variant_drawer import (
    MoleculeBondVariantDrawer,
    ValenceBondStructureInfo,
)


class OrbitalConnectivityMoleculeDrawer(MoleculeBondVariantDrawer):
    """使用 XMO ``$orb`` 原子标签建立基础键连的价键结构绘图器。"""

    def __init__(
        self,
        xyz_file: str | Path,
        output_dir: str | Path,
        charge: int = 0,
        *,
        orbital_atom_rows: Sequence[Sequence[int]],
        active_bond_atom: Sequence[Sequence[int]],
        active_space: Sequence[ValenceBondStructureInfo],
        baseline_unpaired_atoms: Sequence[int] | None = None,
        active_space_color: str | None = None,
        active_space_width: float | None = None,
        color_active_space: bool = True,
        charge_note_scale: float | None = None,
        show_atom_labels: bool = True,
        atom_label_color: str | None = None,
        atom_label_font_size: float | None = None,
        hide_hydrogens: bool = True,
        show_lone_pairs: bool = True,
        lone_pair_color: str | None = None,
        lone_pair_dot_radius: float | None = None,
        write_individual_svgs: bool = False,
        structures_per_row: int = 2,
    ) -> None:
        super().__init__(
            xyz_file=xyz_file,
            output_dir=output_dir,
            charge=charge,
            active_bond_atom=active_bond_atom,
            active_space=active_space,
            baseline_unpaired_atoms=baseline_unpaired_atoms,
            active_space_color=active_space_color,
            active_space_width=active_space_width,
            color_active_space=color_active_space,
            charge_note_scale=charge_note_scale,
            show_atom_labels=show_atom_labels,
            atom_label_color=atom_label_color,
            atom_label_font_size=atom_label_font_size,
            hide_hydrogens=hide_hydrogens,
            show_lone_pairs=show_lone_pairs,
            lone_pair_color=lone_pair_color,
            lone_pair_dot_radius=lone_pair_dot_radius,
            write_individual_svgs=write_individual_svgs,
            structures_per_row=structures_per_row,
        )
        self.orbital_atom_rows = self._copy_atom_groups(orbital_atom_rows)

    def build_base_molecule(self) -> Chem.Mol:
        """从双原子 ``$orb`` 标签建立连接，再由 RDKit 分配键级。"""
        if not self.xyz_file.exists():
            raise FileNotFoundError(f"XYZ file not found: {self.xyz_file}")

        raw_mol = Chem.MolFromXYZFile(str(self.xyz_file))
        if raw_mol is None:
            raise ValueError(f"Could not parse XYZ file: {self.xyz_file}")

        orbital_bonds = self._orbital_bonds()
        if not orbital_bonds:
            raise ValueError(
                f"Cannot build connectivity for {self.xyz_file.name}: "
                "$orb contains no two-atom labels."
            )

        editable_mol = Chem.RWMol(raw_mol)
        for begin_atom, end_atom in orbital_bonds:
            editable_mol.AddBond(
                begin_atom - 1,
                end_atom - 1,
                Chem.BondType.SINGLE,
            )
        connectivity_mol = editable_mol.GetMol()

        mol = None
        first_exc: Exception | None = None
        allow_charged_fragments_options = (
            (False, True) if self.baseline_unpaired_atoms else (True, False)
        )
        for allow_charged_fragments in allow_charged_fragments_options:
            candidate = Chem.Mol(connectivity_mol)
            try:
                rdDetermineBonds.DetermineBondOrders(
                    candidate,
                    charge=self.charge,
                    allowChargedFragments=allow_charged_fragments,
                )
            except Exception as exc:
                if first_exc is None:
                    first_exc = exc
                continue

            if self._electron_assignment_matches(candidate):
                mol = candidate
                break

        if mol is None:
            raise ValueError(
                f"Bond-order perception from $orb failed for {self.xyz_file.name}. "
                f"First RDKit error: {first_exc}"
            )

        visible_mol = (
            Chem.RemoveHs(mol, sanitize=False)
            if self.hide_hydrogens
            else Chem.Mol(mol)
        )
        try:
            Chem.Kekulize(visible_mol, clearAromaticFlags=True)
        except Exception as exc:
            raise ValueError(f"Kekulization failed for {self.xyz_file.name}.") from exc

        for atom in visible_mol.GetAtoms():
            atom.SetNoImplicit(True)
            atom.SetNumRadicalElectrons(0)

        visible_mol.UpdatePropertyCache(strict=False)
        rdDepictor.Compute2DCoords(visible_mol)
        return visible_mol

    def _orbital_bonds(self) -> list[tuple[int, int]]:
        """提取恰好包含两个不同原子的 ``$orb`` 标签并去重。"""
        return sorted(
            {
                self._normalize_bond((row[0], row[1]))
                for row in self.orbital_atom_rows
                if len(row) == 2 and row[0] != row[1]
            }
        )
