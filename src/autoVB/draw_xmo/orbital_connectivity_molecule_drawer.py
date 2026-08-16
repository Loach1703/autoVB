from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdDetermineBonds
from rdkit.Geometry import Point3D

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
        layout_bond_pairs: Sequence[tuple[int, int]] | None = None,
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
        projection: str = "rdkit",
        condense_hydrogens: bool = True,
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
        self.layout_bond_pairs = (
            list(layout_bond_pairs)
            if layout_bond_pairs is not None
            else [
                pair
                for structure in self.active_space
                for pair in structure.bond_pairs
            ]
        )
        self.projection = projection
        self.condense_hydrogens = condense_hydrogens

    def build_base_molecule(self) -> Chem.Mol:
        """从双原子 ``$orb`` 标签建立连接，再由 RDKit 分配键级。"""
        if not self.xyz_file.exists():
            raise FileNotFoundError(f"XYZ file not found: {self.xyz_file}")

        raw_mol = Chem.MolFromXYZFile(str(self.xyz_file))
        if raw_mol is None:
            raise ValueError(f"Could not parse XYZ file: {self.xyz_file}")
        for atom in raw_mol.GetAtoms():
            atom.SetIntProp(self.ORIGINAL_ATOM_NUMBER_PROP, atom.GetIdx() + 1)

        orbital_bonds = self._orbital_bonds()

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

        if self.hide_hydrogens and self.condense_hydrogens:
            self._set_condensed_hydrogen_labels(mol)
        visible_mol = (
            Chem.RemoveHs(mol, sanitize=True)
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
        if self.projection == "optimized3d":
            self._apply_optimized_3d_projection(visible_mol)
        elif self.projection == "pca":
            self._apply_pca_projection(visible_mol)
        elif self.projection == "contact":
            self._apply_contact_projection(visible_mol)
        elif self.projection == "rdkit":
            rdDepictor.Compute2DCoords(visible_mol)
        else:
            raise ValueError(
                "projection must be 'rdkit', 'pca', 'optimized3d' or 'contact', "
                f"but got {self.projection!r}."
            )
        return visible_mol

    def _set_condensed_hydrogen_labels(self, mol: Chem.Mol) -> None:
        """为杂原子和孤立碳记录删氢后的紧凑显示标签。"""
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue

            hydrogen_count = sum(
                neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors()
            )
            heavy_atom_count = sum(
                neighbor.GetAtomicNum() != 1 for neighbor in atom.GetNeighbors()
            )
            if hydrogen_count == 0:
                continue
            if atom.GetAtomicNum() == 6 and heavy_atom_count > 0:
                continue

            hydrogen_label = "H"
            if hydrogen_count > 1:
                hydrogen_label += f"<sub>{hydrogen_count}</sub>"
            atom.SetProp(
                self.CONDENSED_ATOM_LABEL_PROP,
                f"{atom.GetSymbol()}{hydrogen_label}",
            )

    def _apply_contact_projection(self, mol: Chem.Mol) -> None:
        """优先用价键结构中的跨片段键安排多个分子片段。"""
        fragments = list(Chem.GetMolFrags(mol))
        if len(fragments) == 1:
            rdDepictor.Compute2DCoords(mol)
            return

        conformer = mol.GetConformer()
        coordinates_3d = np.array(
            [
                list(conformer.GetAtomPosition(atom_index))
                for atom_index in range(mol.GetNumAtoms())
            ],
            dtype=float,
        )

        # RDKit 先把每个片段画规整；后面只整体移动片段。
        rdDepictor.Compute2DCoords(mol)
        conformer = mol.GetConformer()
        coordinates_2d = np.array(
            [
                list(conformer.GetAtomPosition(atom_index))[:2]
                for atom_index in range(mol.GetNumAtoms())
            ],
            dtype=float,
        )
        initial_centers = [
            coordinates_2d[list(fragment)].mean(axis=0)
            for fragment in fragments
        ]

        core_index = max(range(len(fragments)), key=lambda index: len(fragments[index]))
        core_atoms = list(fragments[core_index])
        coordinates_2d[core_atoms] -= coordinates_2d[core_atoms].mean(axis=0)

        atom_to_fragment = {
            atom_index: fragment_index
            for fragment_index, fragment in enumerate(fragments)
            for atom_index in fragment
        }
        placed_atoms = set(fragments[core_index])
        remaining_fragments = set(range(len(fragments))) - {core_index}

        while remaining_fragments:
            _, fragment_index, placed_anchor, fragment_anchor = min(
                (
                    np.linalg.norm(
                        coordinates_3d[placed_atom]
                        - coordinates_3d[fragment_atom]
                    ),
                    candidate_index,
                    placed_atom,
                    fragment_atom,
                )
                for candidate_index in remaining_fragments
                for placed_atom in placed_atoms
                for fragment_atom in fragments[candidate_index]
            )
            fragment_atoms = list(fragments[fragment_index])
            placed_fragment = atom_to_fragment[placed_anchor]
            contact_pairs = self._active_contact_pairs(
                placed_atoms,
                set(fragment_atoms),
            )
            use_multiple_contacts = (
                len({pair[0] for pair in contact_pairs}) > 1
                and len({pair[1] for pair in contact_pairs}) > 1
            )

            fragment_coordinates = coordinates_2d[fragment_atoms]
            fragment_center = fragment_coordinates.mean(axis=0)
            centered_fragment = fragment_coordinates - fragment_center

            if use_multiple_contacts:
                placed_anchors = [pair[0] for pair in contact_pairs]
                fragment_anchors = [pair[1] for pair in contact_pairs]
                placed_anchor_coordinates = coordinates_2d[placed_anchors]
                fragment_anchor_coordinates = coordinates_2d[fragment_anchors]

                centered_placed_anchors = (
                    placed_anchor_coordinates
                    - placed_anchor_coordinates.mean(axis=0)
                )
                centered_fragment_anchors = (
                    fragment_anchor_coordinates
                    - fragment_anchor_coordinates.mean(axis=0)
                )
                left_axes, _, right_axes = np.linalg.svd(
                    centered_fragment_anchors.T @ centered_placed_anchors
                )

                _, _, interface_axes = np.linalg.svd(
                    centered_placed_anchors,
                    full_matrices=False,
                )
                interface_axis = interface_axes[0]
                direction = np.array(
                    [-interface_axis[1], interface_axis[0]]
                )
                fragment_anchor_positions = [
                    fragment_atoms.index(atom) for atom in fragment_anchors
                ]
                placement_options = []
                for reflect_fragment in (False, True):
                    candidate_left_axes = left_axes.copy()
                    if reflect_fragment:
                        candidate_left_axes[:, -1] *= -1
                    candidate_rotation = candidate_left_axes @ right_axes
                    candidate_fragment = (
                        centered_fragment @ candidate_rotation
                    )
                    candidate_anchors = candidate_fragment[
                        fragment_anchor_positions
                    ]
                    for candidate_direction in (direction, -direction):
                        candidate_coordinates = candidate_fragment + (
                            placed_anchor_coordinates
                            + candidate_direction * 1.5
                            - candidate_anchors
                        ).mean(axis=0)
                        placement_options.append(
                            (
                                self._fragment_clearance(
                                    candidate_coordinates,
                                    coordinates_2d,
                                    placed_atoms,
                                ),
                                candidate_fragment,
                                candidate_anchors,
                                candidate_direction,
                            )
                        )
                (
                    _,
                    rotated_fragment,
                    rotated_anchor_coordinates,
                    direction,
                ) = max(
                    placement_options,
                    key=lambda option: option[0],
                )
            else:
                if contact_pairs:
                    placed_anchor, fragment_anchor = contact_pairs[0]
                    placed_fragment = atom_to_fragment[placed_anchor]

                placed_center = coordinates_2d[
                    list(fragments[placed_fragment])
                ].mean(axis=0)
                direction = coordinates_2d[placed_anchor] - placed_center
                if np.linalg.norm(direction) == 0:
                    direction = (
                        initial_centers[fragment_index]
                        - initial_centers[placed_fragment]
                    )
                direction /= np.linalg.norm(direction)

                anchor_vector = (
                    coordinates_2d[fragment_anchor] - fragment_center
                )
                angle = np.arctan2(-direction[1], -direction[0]) - np.arctan2(
                    anchor_vector[1],
                    anchor_vector[0],
                )
                rotation = np.array(
                    [
                        [np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)],
                    ]
                )
                rotated_fragment = centered_fragment @ rotation.T
                anchor_position = rotated_fragment[
                    fragment_atoms.index(fragment_anchor)
                ]

            gap = 1.5
            while True:
                if use_multiple_contacts:
                    placed_fragment_coordinates = rotated_fragment + (
                        placed_anchor_coordinates
                        + direction * gap
                        - rotated_anchor_coordinates
                    ).mean(axis=0)
                else:
                    target_anchor = (
                        coordinates_2d[placed_anchor] + direction * gap
                    )
                    placed_fragment_coordinates = (
                        rotated_fragment + target_anchor - anchor_position
                    )
                distances = np.linalg.norm(
                    placed_fragment_coordinates[:, np.newaxis, :]
                    - coordinates_2d[list(placed_atoms)][np.newaxis, :, :],
                    axis=2,
                )
                if distances.min() >= 1.2:
                    break
                gap += 0.25

            coordinates_2d[fragment_atoms] = placed_fragment_coordinates
            placed_atoms.update(fragment_atoms)
            remaining_fragments.remove(fragment_index)

        for atom_index, (x_coord, y_coord) in enumerate(coordinates_2d):
            conformer.SetAtomPosition(
                atom_index,
                Point3D(float(x_coord), float(y_coord), 0.0),
            )
        conformer.Set3D(False)

    def _active_contact_pairs(
        self,
        placed_atoms: set[int],
        fragment_atoms: set[int],
    ) -> list[tuple[int, int]]:
        """提取所有展示结构中连接当前两个片段的原子对。"""
        if not self.layout_bond_pairs:
            return []

        contact_pairs = []
        for begin_atom, end_atom in self.layout_bond_pairs:
            begin_index = begin_atom - 1
            end_index = end_atom - 1
            if begin_index in placed_atoms and end_index in fragment_atoms:
                contact_pairs.append((begin_index, end_index))
            elif end_index in placed_atoms and begin_index in fragment_atoms:
                contact_pairs.append((end_index, begin_index))
        return contact_pairs

    @staticmethod
    def _fragment_clearance(
        fragment_coordinates: np.ndarray,
        coordinates_2d: np.ndarray,
        placed_atoms: set[int],
    ) -> float:
        """返回待放置片段与已有原子的最短二维距离。"""
        distances = np.linalg.norm(
            fragment_coordinates[:, np.newaxis, :]
            - coordinates_2d[list(placed_atoms)][np.newaxis, :, :],
            axis=2,
        )
        return float(distances.min())

    @classmethod
    def _apply_optimized_3d_projection(cls, mol: Chem.Mol) -> None:
        """搜索最清晰的正交观察方向并投影三维坐标。"""
        conformer = mol.GetConformer()
        coordinates = np.array(
            [
                list(conformer.GetAtomPosition(atom_index))
                for atom_index in range(mol.GetNumAtoms())
            ],
            dtype=float,
        )
        centered_coordinates = coordinates - coordinates.mean(axis=0)
        _, _, principal_axes = np.linalg.svd(
            centered_coordinates,
            full_matrices=True,
        )
        view_directions = cls._hemisphere_view_directions(240)
        view_directions.extend(principal_axes)

        selected_coordinates = None
        selected_score = None
        for view_direction in view_directions:
            projected_coordinates = cls._orthographic_projection(
                centered_coordinates,
                view_direction,
            )
            score = cls._optimized_projection_score(
                mol,
                centered_coordinates,
                projected_coordinates,
            )
            if selected_score is None or score < selected_score:
                selected_score = score
                selected_coordinates = projected_coordinates

        projected_coordinates = cls._orient_projected_coordinates(
            selected_coordinates
        )
        for atom_index, (x_coord, y_coord) in enumerate(projected_coordinates):
            conformer.SetAtomPosition(
                atom_index,
                Point3D(float(x_coord), float(y_coord), 0.0),
            )
        conformer.Set3D(False)

    @staticmethod
    def _hemisphere_view_directions(count: int) -> list[np.ndarray]:
        """使用 Fibonacci 网格在半球面均匀生成观察方向。"""
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        directions = []
        for index in range(count):
            z_coord = (index + 0.5) / count
            radius = np.sqrt(1.0 - z_coord * z_coord)
            angle = index * golden_angle
            directions.append(
                np.array(
                    [
                        radius * np.cos(angle),
                        radius * np.sin(angle),
                        z_coord,
                    ]
                )
            )
        return directions

    @staticmethod
    def _orthographic_projection(
        coordinates: np.ndarray,
        view_direction: np.ndarray,
    ) -> np.ndarray:
        """把三维坐标正交投影到指定观察方向的垂直平面。"""
        view_direction = view_direction / np.linalg.norm(view_direction)
        reference_axis = (
            np.array([0.0, 0.0, 1.0])
            if abs(view_direction[2]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        horizontal_axis = np.cross(reference_axis, view_direction)
        horizontal_axis /= np.linalg.norm(horizontal_axis)
        vertical_axis = np.cross(view_direction, horizontal_axis)
        return coordinates @ np.column_stack(
            (horizontal_axis, vertical_axis)
        )

    @classmethod
    def _optimized_projection_score(
        cls,
        mol: Chem.Mol,
        coordinates_3d: np.ndarray,
        coordinates_2d: np.ndarray,
    ) -> float:
        """根据键压缩、重叠和交叉程度评价一个投影视角。"""
        bond_lengths_3d = []
        bond_ratios = []
        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            distance_3d = np.linalg.norm(
                coordinates_3d[begin_atom] - coordinates_3d[end_atom]
            )
            distance_2d = np.linalg.norm(
                coordinates_2d[begin_atom] - coordinates_2d[end_atom]
            )
            bond_lengths_3d.append(distance_3d)
            bond_ratios.append(distance_2d / distance_3d)

        bond_scale = (
            float(np.median(bond_lengths_3d))
            if bond_lengths_3d
            else 1.5
        )
        collapsed_bond_penalty = sum(
            max(0.0, 0.65 - ratio) ** 2 for ratio in bond_ratios
        )

        atom_overlap_penalty = 0.0
        minimum_atom_distance = 0.55 * bond_scale
        for first_atom in range(mol.GetNumAtoms()):
            for second_atom in range(first_atom + 1, mol.GetNumAtoms()):
                distance = np.linalg.norm(
                    coordinates_2d[first_atom] - coordinates_2d[second_atom]
                )
                atom_overlap_penalty += (
                    max(0.0, minimum_atom_distance - distance)
                    / bond_scale
                ) ** 2

        fragment_overlap_penalty = cls._fragment_overlap_penalty(
            mol,
            coordinates_2d,
            bond_scale,
        )
        crossing_count = cls._bond_crossing_count(mol, coordinates_2d)
        spread = float(np.var(coordinates_2d, axis=0).sum()) / (
            bond_scale * bond_scale
        )
        return (
            24.0 * collapsed_bond_penalty
            + 16.0 * atom_overlap_penalty
            + 10.0 * fragment_overlap_penalty
            + 4.0 * crossing_count
            - 0.05 * spread
        )

    @staticmethod
    def _fragment_overlap_penalty(
        mol: Chem.Mol,
        coordinates_2d: np.ndarray,
        bond_scale: float,
    ) -> float:
        """惩罚不同分子片段在投影平面中过度重叠。"""
        fragments = list(Chem.GetMolFrags(mol))
        centers = [
            coordinates_2d[list(fragment)].mean(axis=0)
            for fragment in fragments
        ]
        radii = [
            max(
                0.5 * bond_scale,
                float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                (
                                    coordinates_2d[list(fragment)]
                                    - centers[index]
                                )
                                ** 2,
                                axis=1,
                            )
                        )
                    )
                ),
            )
            for index, fragment in enumerate(fragments)
        ]

        penalty = 0.0
        for first_fragment in range(len(fragments)):
            for second_fragment in range(first_fragment + 1, len(fragments)):
                separation = np.linalg.norm(
                    centers[first_fragment] - centers[second_fragment]
                )
                required_separation = (
                    radii[first_fragment] + radii[second_fragment]
                )
                penalty += max(
                    0.0,
                    0.8 - separation / required_separation,
                ) ** 2
        return penalty

    @classmethod
    def _bond_crossing_count(
        cls,
        mol: Chem.Mol,
        coordinates_2d: np.ndarray,
    ) -> int:
        """统计没有共享原子的键线之间发生的交叉。"""
        bonds = [
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in mol.GetBonds()
        ]
        crossing_count = 0
        for first_index, first_bond in enumerate(bonds):
            for second_bond in bonds[first_index + 1 :]:
                if set(first_bond) & set(second_bond):
                    continue
                if cls._segments_cross(
                    coordinates_2d[first_bond[0]],
                    coordinates_2d[first_bond[1]],
                    coordinates_2d[second_bond[0]],
                    coordinates_2d[second_bond[1]],
                ):
                    crossing_count += 1
        return crossing_count

    @staticmethod
    def _segments_cross(
        first_start: np.ndarray,
        first_end: np.ndarray,
        second_start: np.ndarray,
        second_end: np.ndarray,
    ) -> bool:
        """判断两条二维线段是否在各自内部相交。"""
        def orientation(
            origin: np.ndarray,
            first_point: np.ndarray,
            second_point: np.ndarray,
        ) -> float:
            first_vector = first_point - origin
            second_vector = second_point - origin
            return float(
                first_vector[0] * second_vector[1]
                - first_vector[1] * second_vector[0]
            )

        first_side = orientation(first_start, first_end, second_start)
        second_side = orientation(first_start, first_end, second_end)
        third_side = orientation(second_start, second_end, first_start)
        fourth_side = orientation(second_start, second_end, first_end)
        return first_side * second_side < 0 and third_side * fourth_side < 0

    @staticmethod
    def _orient_projected_coordinates(
        projected_coordinates: np.ndarray,
    ) -> np.ndarray:
        """在投影平面内固定主方向和正负号，保证输出稳定。"""
        centered_coordinates = (
            projected_coordinates - projected_coordinates.mean(axis=0)
        )
        _, _, principal_axes = np.linalg.svd(
            centered_coordinates,
            full_matrices=False,
        )
        oriented_coordinates = centered_coordinates @ principal_axes.T
        for axis in range(2):
            anchor_index = np.argmax(np.abs(oriented_coordinates[:, axis]))
            if oriented_coordinates[anchor_index, axis] < 0:
                oriented_coordinates[:, axis] *= -1
        return oriented_coordinates

    @classmethod
    def _apply_pca_projection(cls, mol: Chem.Mol) -> None:
        """从 PCA 主轴平面中选择适合绘图的平面并投影三维坐标。"""
        conformer = mol.GetConformer()
        coordinates = np.array(
            [
                list(conformer.GetAtomPosition(atom_index))
                for atom_index in range(mol.GetNumAtoms())
            ],
            dtype=float,
        )
        centered_coordinates = coordinates - coordinates.mean(axis=0)
        _, _, principal_axes = np.linalg.svd(
            centered_coordinates,
            full_matrices=True,
        )
        axis_pairs = ((0, 1), (0, 2), (1, 2))
        selected_axes = max(
            axis_pairs,
            key=lambda axes: cls._projection_score(
                mol,
                centered_coordinates,
                principal_axes,
                axes,
            ),
        )
        projected_coordinates = (
            centered_coordinates @ principal_axes[list(selected_axes)].T
        )

        # PCA 主轴的正负号不唯一；固定方向可以避免同一输入
        # 偶尔发生镜像翻转。
        for axis in range(2):
            anchor_index = np.argmax(np.abs(projected_coordinates[:, axis]))
            if projected_coordinates[anchor_index, axis] < 0:
                projected_coordinates[:, axis] *= -1

        for atom_index, (x_coord, y_coord) in enumerate(projected_coordinates):
            conformer.SetAtomPosition(
                atom_index,
                Point3D(float(x_coord), float(y_coord), 0.0),
            )
        conformer.Set3D(False)

    @staticmethod
    def _projection_score(
        mol: Chem.Mol,
        coordinates: np.ndarray,
        principal_axes: np.ndarray,
        axes: tuple[int, int],
    ) -> float:
        """兼顾整体展开程度和真实键投影长度，为主轴平面评分。"""
        projected_coordinates = coordinates @ principal_axes[list(axes)].T
        spread = float(np.var(projected_coordinates, axis=0).sum())
        bond_length_ratios = []
        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            distance_3d = np.linalg.norm(
                coordinates[begin_atom] - coordinates[end_atom]
            )
            if distance_3d > 0:
                distance_2d = np.linalg.norm(
                    projected_coordinates[begin_atom]
                    - projected_coordinates[end_atom]
                )
                bond_length_ratios.append(distance_2d / distance_3d)

        shortest_bond_ratio = min(bond_length_ratios, default=1.0)
        return spread * shortest_bond_ratio

    def _orbital_bonds(self) -> list[tuple[int, int]]:
        """提取恰好包含两个不同原子的 ``$orb`` 标签并去重。"""
        return sorted(
            {
                self._normalize_bond((row[0], row[1]))
                for row in self.orbital_atom_rows
                if len(row) == 2 and row[0] != row[1]
            }
        )
