from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdDepictor, rdDetermineBonds
from rdkit.Chem.Draw import rdMolDraw2D


@dataclass
class ValenceBondStructureInfo:
    """单个价键结构的输入信息。

    Attributes:
        file_suffix: 输出 SVG 文件名后缀，例如 `"12"` 会生成 `water_12.svg`。
        legend: 图片下方显示的图例文本。
        bond_pairs: 成键电子配对。该列表用于键升阶与电荷计算；
            例如 `[(1, 2)]` 表示一对电子分布在 1-2 之间。
            自配对 `[(1, 1)]` 表示一对电子都分布在 1 号原子上。
        unpaired_atoms: 未成对电子所在的原子编号。每个编号表示一个单电子，
            绘图时显示为一个红色点。
    """

    file_suffix: str
    legend: str
    bond_pairs: list[tuple[int, int]]
    unpaired_atoms: list[int] = field(default_factory=list)


class MoleculeBondVariantDrawer:
    """从 XYZ 文件生成多种价键结构 SVG 图。

    原子编号使用“当前可见原子”在 XYZ 文件中的顺序。默认会隐藏氢原子；
    如果需要把氢纳入电子分布与绘图，请设置 `hide_hydrogens=False`。
    芳香键会先转换为 Kekule 单双键交替结构，再执行键降级与键升级。

    Args:
        xyz_file: 输入 XYZ 文件路径。
        output_dir: SVG 输出目录。
        charge: RDKit 从 XYZ 推断键连时使用的总电荷。
        active_bond_atom: 原始电子区域。每个原子编号出现一次代表
            该原子贡献 1 个电子；例如 `[1, 2]` 代表 1-2 区域有 2 个电子。
            重复编号可表示同一键多级降级，例如 `[1, 2, 1, 2]`。
        active_space: 要绘制的价键结构信息列表，其中 `bond_pairs` 管成键、电荷
            和自配对电子。
        active_space_color: 活性空间键、电荷符号、孤对电子点的默认颜色。
        active_space_width: 活性空间键线条宽度。
        color_active_space: 是否高亮活性空间中新升高键级的键线。
        charge_note_scale: 电荷符号字号缩放。
        show_atom_labels: 是否显示原子编号小标签。
        atom_label_color: 原子编号标签颜色。
        atom_label_font_size: 原子编号标签字号。
        hide_hydrogens: 是否隐藏氢原子。
        show_lone_pairs: 是否显示价键结构中的孤对电子点。
        lone_pair_color: 孤对电子点颜色；默认跟随 `active_space_color`。
        lone_pair_dot_radius: 孤对电子点半径。
        write_individual_svgs: 是否为每个价键结构额外写出单张 SVG；
            默认只写出 grid 汇总图。
    """

    DEFAULT_ACTIVE_SPACE_COLOR = "#E00000"
    DEFAULT_ACTIVE_SPACE_WIDTH = 3.0
    DEFAULT_CHARGE_NOTE_SCALE = 1.2
    DEFAULT_ATOM_LABEL_COLOR = "#000000"
    DEFAULT_ATOM_LABEL_FONT_SIZE = 12.0
    DEFAULT_CHARGE_LABEL_BASE_FONT_SIZE = 22.0
    DEFAULT_LONE_PAIR_DOT_RADIUS = 3.4
    DEFAULT_RADICAL_DOT_RADIUS = 4.0
    ATOM_LABEL_OFFSET = 13.0
    CHARGE_LABEL_OFFSET = 34.0
    LONE_PAIR_OFFSET = 23.0
    LONE_PAIR_DOT_GAP = 9.0
    CHARGE_NOTE_PROP = "_bondVariantChargeNote"
    LONE_PAIR_COUNT_PROP = "_bondVariantLonePairCount"
    RADICAL_COUNT_PROP = "_bondVariantRadicalCount"

    @dataclass
    class Result:
        """一次绘图运行写出的文件信息。

        Attributes:
            xyz_file: 实际读取的 XYZ 文件路径。
            output_dir: SVG 输出目录。
            displayed_atom_count: 图中可见原子数量。
            written_files: 本次运行实际写出的 SVG 文件路径列表。
        """

        xyz_file: Path
        output_dir: Path
        displayed_atom_count: int
        written_files: list[Path]

    def __init__(
        self,
        xyz_file: str | Path = "xyz_inputs/benzene.xyz",
        output_dir: str | Path = "output_benzene_bonds",
        charge: int = 0,
        *,
        active_bond_atom: Sequence[Sequence[int]],
        active_space: Sequence[ValenceBondStructureInfo],
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
        """初始化绘图器。

        Args:
            xyz_file: 输入 XYZ 文件路径。
            output_dir: SVG 输出目录。
            charge: RDKit 从 XYZ 推断键连时使用的总电荷。
            active_bond_atom: 需要先降级的原始电子区域。
            active_space: 价键结构输入。
            active_space_color: 活性空间键、电荷、孤对电子点的默认颜色。
            active_space_width: 活性空间键线条宽度。
            color_active_space: 是否高亮活性空间中新升高键级的键线。
            charge_note_scale: 电荷符号字号缩放。
            show_atom_labels: 是否显示原子编号标签。
            atom_label_color: 原子编号标签颜色。
            atom_label_font_size: 原子编号标签字号。
            hide_hydrogens: 是否隐藏氢原子。
            show_lone_pairs: 是否显示孤对电子点。
            lone_pair_color: 孤对电子点颜色。
            lone_pair_dot_radius: 孤对电子点半径。
            write_individual_svgs: 是否写出每个结构的单张 SVG；
                如果为 `False`，则只写出 grid 汇总图。
            structures_per_row: grid 汇总图中每行显示的结构数量。

        Returns:
            None.
        """
        self.xyz_file = Path(xyz_file)
        self.output_dir = Path(output_dir)
        self.charge = charge
        self.active_bond_atom = self._copy_atom_groups(active_bond_atom)
        self.active_space = self._copy_active_space(active_space)
        self.active_space_color = active_space_color or self.DEFAULT_ACTIVE_SPACE_COLOR
        self.active_space_width = (
            self.DEFAULT_ACTIVE_SPACE_WIDTH
            if active_space_width is None
            else active_space_width
        )
        self.color_active_space = color_active_space
        self.charge_note_scale = (
            self.DEFAULT_CHARGE_NOTE_SCALE
            if charge_note_scale is None
            else charge_note_scale
        )
        self.show_atom_labels = show_atom_labels
        self.atom_label_color = atom_label_color or self.DEFAULT_ATOM_LABEL_COLOR
        self.atom_label_font_size = (
            self.DEFAULT_ATOM_LABEL_FONT_SIZE
            if atom_label_font_size is None
            else atom_label_font_size
        )
        self.hide_hydrogens = hide_hydrogens
        self.show_lone_pairs = show_lone_pairs
        self.lone_pair_color = lone_pair_color or self.active_space_color
        self.lone_pair_dot_radius = (
            self.DEFAULT_LONE_PAIR_DOT_RADIUS
            if lone_pair_dot_radius is None
            else lone_pair_dot_radius
        )
        self.write_individual_svgs = write_individual_svgs
        self.structures_per_row = structures_per_row

    @staticmethod
    def _copy_atom_groups(atom_groups: Sequence[Sequence[int]]) -> list[list[int]]:
        """复制原始电子区域配置。

        Args:
            atom_groups: 原始电子区域原子编号列表。

        Returns:
            深拷贝后的二维列表。
        """
        return [list(atom_group) for atom_group in atom_groups]

    @classmethod
    def _copy_active_space(
        cls,
        structures: Sequence[ValenceBondStructureInfo],
    ) -> list[ValenceBondStructureInfo]:
        """复制价键结构输入，避免后续修改调用方的列表。

        Args:
            structures: `ValenceBondStructureInfo` 形式的价键结构输入。

        Returns:
            标准化后的 `ValenceBondStructureInfo` 列表。
        """
        return [cls._copy_valence_bond_structure(structure) for structure in structures]

    @staticmethod
    def _copy_valence_bond_structure(
        structure: ValenceBondStructureInfo,
    ) -> ValenceBondStructureInfo:
        """复制单个价键结构输入。

        Args:
            structure: 单个 `ValenceBondStructureInfo` 实例。

        Returns:
            深拷贝列表字段后的 `ValenceBondStructureInfo`。
        """
        return ValenceBondStructureInfo(
            file_suffix=structure.file_suffix,
            legend=structure.legend,
            bond_pairs=list(structure.bond_pairs),
            unpaired_atoms=list(structure.unpaired_atoms),
        )

    def draw(self) -> MoleculeBondVariantDrawer.Result:
        """生成价键结构 SVG 文件。

        默认只写出 grid 汇总图。若初始化时设置
        `write_individual_svgs=True`，则额外为每个价键结构写出单张 SVG。

        Returns:
            `MoleculeBondVariantDrawer.Result`，包含输入文件、输出目录、
            可见原子数量以及本次实际写出的 SVG 文件路径。
        """
        base_mol = self.build_base_molecule()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        mols: list[Chem.Mol] = []
        legends: list[str] = []
        upgraded_bond_counts_list: list[dict[int, int]] = []
        written_files: list[Path] = []
        output_prefix = self.xyz_file.stem

        for valence_bond_structure in self.active_space:
            mol, upgraded_bond_counts = self.apply_variant(
                base_mol,
                valence_bond_structure,
            )
            mols.append(mol)
            legends.append(valence_bond_structure.legend)
            upgraded_bond_counts_list.append(upgraded_bond_counts)

            if self.write_individual_svgs:
                svg_file = (
                    self.output_dir
                    / f"{output_prefix}_{valence_bond_structure.file_suffix}.svg"
                )
                svg_file.write_text(
                    self.draw_variant_svg(
                        mol,
                        valence_bond_structure.legend,
                        upgraded_bond_counts,
                    ),
                    encoding="utf-8",
                )
                written_files.append(svg_file)

        grid_file = self.output_dir / f"{output_prefix}_grid.svg"
        grid_file.write_text(
            self.draw_grid_svg(mols, legends, upgraded_bond_counts_list),
            encoding="utf-8",
        )
        written_files.append(grid_file)

        return self.Result(
            xyz_file=self.xyz_file,
            output_dir=self.output_dir,
            displayed_atom_count=base_mol.GetNumAtoms(),
            written_files=written_files,
        )

    def build_base_molecule(self) -> Chem.Mol:
        """读取并预处理基础分子。

        处理顺序是：读取 XYZ、调用 RDKit 推断键连、按需隐藏氢原子、
        将芳香键 Kekulize 为单双键交替，并关闭隐式氢。

        Returns:
            预处理后的 RDKit `Chem.Mol` 分子对象。

        Raises:
            FileNotFoundError: `xyz_file` 不存在。
            ValueError: XYZ 解析、键连推断或 Kekulize 失败。
        """
        if not self.xyz_file.exists():
            raise FileNotFoundError(f"XYZ file not found: {self.xyz_file}")

        mol = Chem.MolFromXYZFile(str(self.xyz_file))
        if mol is None:
            raise ValueError(f"Could not parse XYZ file: {self.xyz_file}")

        try:
            rdDetermineBonds.DetermineBonds(mol, charge=self.charge)
        except Exception as first_exc:
            try:
                # 自由基体系常因形式电荷/价态分配失败；改用 radical 标记后
                # 通常仍能得到可用于绘图的键级。
                rdDetermineBonds.DetermineBonds(
                    mol,
                    charge=self.charge,
                    allowChargedFragments=False,
                )
            except Exception:
                try:
                    # 价键结构绘图至少需要连接拓扑；若键级无法可靠推断，
                    # 退回到只根据坐标推断连接关系。
                    rdDetermineBonds.DetermineConnectivity(mol, charge=self.charge)
                except Exception as exc:
                    raise ValueError(
                        f"Bond perception failed for {self.xyz_file.name}. "
                        f"Try a different charge value. First RDKit error: "
                        f"{first_exc}"
                    ) from exc

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

        visible_mol.UpdatePropertyCache(strict=False)
        rdDepictor.Compute2DCoords(visible_mol)
        return visible_mol

    def apply_variant(
        self,
        base_mol: Chem.Mol,
        valence_bond_structure: ValenceBondStructureInfo,
    ) -> tuple[Chem.Mol, dict[int, int]]:
        """应用一个价键结构：先降级原结构，再按结构信息升阶。

        Args:
            base_mol: 已从 XYZ 读入并完成预处理的基础分子。
            valence_bond_structure: 单个价键结构信息。

        Returns:
            二元组：处理后的分子，以及每根被升级键的升级次数。
        """
        structure_name = valence_bond_structure.legend
        self._validate_atom_groups(
            base_mol,
            self.active_bond_atom,
            "active_bond_atom",
        )
        self._validate_bonds(
            base_mol,
            valence_bond_structure.bond_pairs,
            f"{structure_name}.bond_pairs",
        )
        self._validate_atom_numbers(
            base_mol,
            valence_bond_structure.unpaired_atoms,
            f"{structure_name}.unpaired_atoms",
        )

        editable_mol = Chem.RWMol(base_mol)
        for atom in editable_mol.GetAtoms():
            atom.SetNoImplicit(True)

        self._decrease_bond_orders_in_atom_groups(editable_mol)

        upgraded_bond_counts: dict[int, int] = {}
        for begin_atom, end_atom in valence_bond_structure.bond_pairs:
            if begin_atom == end_atom:
                # self-pair 可以参与电荷计算，但不会生成自键。
                continue
            upgraded_bond_id = self._increase_bond_order(
                editable_mol,
                begin_atom,
                end_atom,
            )
            if upgraded_bond_id is not None:
                upgraded_bond_counts[upgraded_bond_id] = (
                    upgraded_bond_counts.get(upgraded_bond_id, 0) + 1
                )

        self._clear_orphan_aromatic_atom_flags(editable_mol)

        mol = editable_mol.GetMol()
        charge_notes = self._charge_notes_from_valence_structure(
            valence_bond_structure
        )
        self._apply_charge_note_props(
            mol,
            charge_notes,
        )
        self._apply_lone_pair_count_props(
            mol,
            self._neutral_self_pair_counts_from_valence_structure(
                valence_bond_structure,
                charge_notes,
            ),
        )
        self._apply_radical_count_props(
            mol,
            self._radical_counts_from_valence_structure(valence_bond_structure),
        )
        mol.UpdatePropertyCache(strict=False)
        return mol, upgraded_bond_counts

    def draw_variant_svg(
        self,
        mol: Chem.Mol,
        legend: str,
        upgraded_bond_counts: dict[int, int],
    ) -> str:
        """绘制单个价键结构的 SVG。

        Args:
            mol: 已应用价键结构后的 RDKit 分子。
            legend: SVG 下方显示的图例。
            upgraded_bond_counts: 每根升级键的升级次数，用于决定标红几根键线。

        Returns:
            SVG 文本。
        """
        return self._draw_molecule_svg(mol, legend, upgraded_bond_counts, 620, 460)

    def draw_grid_svg(
        self,
        mols: list[Chem.Mol],
        legends: list[str],
        upgraded_bond_counts_list: list[dict[int, int]],
    ) -> str:
        """把多个价键结构拼成一个 grid SVG。

        Args:
            mols: 每个价键结构对应的 RDKit 分子。
            legends: 每个子图下方显示的图例。
            upgraded_bond_counts_list: 每个分子的升级键统计。

        Returns:
            包含所有子图的 SVG 文本。
        """
        sub_img_width = 520
        sub_img_height = 390
        mols_per_row = self.structures_per_row
        row_count = (len(mols) + mols_per_row - 1) // mols_per_row
        grid_width = sub_img_width * mols_per_row
        grid_height = sub_img_height * row_count

        parts = [
            "<?xml version='1.0' encoding='iso-8859-1'?>\n",
            "<svg version='1.1' baseProfile='full'\n",
            "              xmlns='http://www.w3.org/2000/svg'\n",
            "                      xmlns:rdkit='http://www.rdkit.org/xml'\n",
            "                      xmlns:xlink='http://www.w3.org/1999/xlink'\n",
            "                  xml:space='preserve'\n",
            f"width='{grid_width}px' height='{grid_height}px' ",
            f"viewBox='0 0 {grid_width} {grid_height}'>\n",
            "<!-- END OF HEADER -->\n",
            (
                "<rect style='opacity:1.0;fill:#FFFFFF;stroke:none' "
                "width='100%' height='100%' x='0' y='0'> </rect>\n"
            ),
        ]

        for idx, (mol, legend, upgraded_bond_counts) in enumerate(
            zip(mols, legends, upgraded_bond_counts_list)
        ):
            col_idx = idx % mols_per_row
            row_idx = idx // mols_per_row
            x_offset = col_idx * sub_img_width
            y_offset = row_idx * sub_img_height
            sub_svg = self._draw_molecule_svg(
                mol,
                legend,
                upgraded_bond_counts,
                sub_img_width,
                sub_img_height,
            )
            parts.append(f"<g transform='translate({x_offset},{y_offset})'>\n")
            parts.append(self._extract_svg_body(sub_svg))
            parts.append("</g>\n")

        parts.append("</svg>\n")
        return "".join(parts)

    def _draw_molecule_svg(
        self,
        mol: Chem.Mol,
        legend: str,
        upgraded_bond_counts: dict[int, int],
        width: int,
        height: int,
    ) -> str:
        """绘制单个分子的 SVG，并插入自定义标注。

        Args:
            mol: 要绘制的 RDKit 分子。
            legend: 图片下方图例。
            upgraded_bond_counts: 每根升级键的升级次数。
            width: SVG 宽度。
            height: SVG 高度。

        Returns:
            已完成键高亮、原子编号、电荷和孤对电子标注的 SVG 文本。
        """
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol, legend=legend)
        atom_coords = self._atom_draw_coords(drawer, mol)
        drawer.FinishDrawing()

        svg = drawer.GetDrawingText()
        if self.color_active_space:
            svg = self._color_upgraded_bond_lines_in_svg(
                svg,
                mol,
                upgraded_bond_counts,
            )
        return self._add_svg_annotations(
            svg,
            atom_coords,
            self._charge_notes_from_mol(mol),
            self._lone_pair_counts_from_mol(mol),
            self._radical_counts_from_mol(mol),
            width,
            height,
        )

    @staticmethod
    def _atom_draw_coords(
        drawer: rdMolDraw2D.MolDraw2DSVG,
        mol: Chem.Mol,
    ) -> list[tuple[float, float]]:
        """读取 RDKit 绘图坐标。

        Args:
            drawer: 已调用 `DrawMolecule` 的 SVG drawer。
            mol: 被绘制的分子。

        Returns:
            每个原子在 SVG 坐标系中的 `(x, y)` 坐标。
        """
        coords: list[tuple[float, float]] = []
        for atom_idx in range(mol.GetNumAtoms()):
            point = drawer.GetDrawCoords(atom_idx)
            coords.append((float(point.x), float(point.y)))
        return coords

    def _add_svg_annotations(
        self,
        svg: str,
        atom_coords: list[tuple[float, float]],
        charge_notes: dict[int, str],
        lone_pair_counts: dict[int, int],
        radical_counts: dict[int, int],
        width: int,
        height: int,
    ) -> str:
        """向 SVG 追加原子编号、电荷和孤对电子标注。

        Args:
            svg: RDKit 生成的原始 SVG 文本。
            atom_coords: 原子的 SVG 坐标。
            charge_notes: 原子索引到电荷符号的映射。
            lone_pair_counts: 原子索引到孤对电子对数的映射。
            radical_counts: 原子索引到未成对电子数的映射。
            width: SVG 宽度。
            height: SVG 高度。

        Returns:
            插入自定义标注后的 SVG 文本。
        """
        annotation_parts: list[str] = []
        centroid = self._point_centroid(atom_coords)

        if self.show_lone_pairs:
            annotation_parts.extend(
                self._svg_lone_pair_dots(
                    atom_coords,
                    centroid,
                    charge_notes,
                    lone_pair_counts,
                    width,
                    height,
                )
            )

        annotation_parts.extend(
            self._svg_radical_dots(
                atom_coords,
                centroid,
                charge_notes,
                lone_pair_counts,
                radical_counts,
                width,
                height,
            )
        )

        if self.show_atom_labels:
            for atom_idx, atom_coord in enumerate(atom_coords):
                atom_label_offset = (
                    -self.CHARGE_LABEL_OFFSET
                    if atom_idx in charge_notes
                    else self.ATOM_LABEL_OFFSET
                )
                annotation_parts.append(
                    self._svg_text_label(
                        css_class=f"atom-number-label atom-{atom_idx}",
                        text=str(atom_idx + 1),
                        position=self._label_position(
                            atom_coord,
                            centroid,
                            atom_label_offset,
                            self.atom_label_font_size,
                            width,
                            height,
                        ),
                        color=self.atom_label_color,
                        font_size=self.atom_label_font_size,
                        font_weight="400",
                    )
                )

        charge_font_size = self.DEFAULT_CHARGE_LABEL_BASE_FONT_SIZE * (
            self.charge_note_scale
        )
        for atom_idx, note in charge_notes.items():
            atom_coord = atom_coords[atom_idx]
            annotation_parts.append(
                self._svg_text_label(
                    css_class=f"charge-label atom-{atom_idx}",
                    text=note,
                    position=self._label_position(
                        atom_coord,
                        centroid,
                        self.CHARGE_LABEL_OFFSET,
                        charge_font_size,
                        width,
                        height,
                    ),
                    color=self.active_space_color,
                    font_size=charge_font_size,
                    font_weight="700",
                )
            )

        if not annotation_parts:
            return svg

        annotations = (
            "<g class='atom-annotations'>\n" + "".join(annotation_parts) + "</g>\n"
        )
        return svg.replace("</svg>", annotations + "</svg>", 1)

    def _svg_lone_pair_dots(
        self,
        atom_coords: list[tuple[float, float]],
        centroid: tuple[float, float],
        charge_notes: dict[int, str],
        lone_pair_counts: dict[int, int],
        width: int,
        height: int,
    ) -> list[str]:
        """生成孤对电子点的 SVG 片段。

        Args:
            atom_coords: 原子的 SVG 坐标。
            centroid: 分子绘图坐标中心。
            charge_notes: 原子索引到电荷符号的映射。
            lone_pair_counts: 原子索引到孤对电子对数的映射。
            width: SVG 宽度。
            height: SVG 高度。

        Returns:
            每个孤对电子点对应的 `<circle>` SVG 文本列表。
        """
        dot_parts: list[str] = []

        for atom_idx, pair_count in lone_pair_counts.items():
            if atom_idx < 0 or atom_idx >= len(atom_coords):
                continue
            atom_coord = atom_coords[atom_idx]
            for pair_idx, angle in enumerate(
                self._lone_pair_angles(atom_coord, centroid, pair_count)
            ):
                dot_parts.extend(
                    self._svg_lone_pair_dot_pair(
                        atom_idx,
                        pair_idx,
                        atom_coord,
                        angle,
                        -self.LONE_PAIR_OFFSET
                        if atom_idx in charge_notes
                        else self.LONE_PAIR_OFFSET,
                        width,
                        height,
                    )
                )

        return dot_parts

    @staticmethod
    def _lone_pair_angles(
        atom_coord: tuple[float, float],
        centroid: tuple[float, float],
        pair_count: int,
    ) -> list[float]:
        """计算孤对电子相对原子的绘制角度。

        Args:
            atom_coord: 原子的 SVG 坐标。
            centroid: 分子绘图坐标中心。
            pair_count: 该原子上要显示的孤对电子对数。

        Returns:
            每对孤对电子的极角，单位为弧度。
        """
        x_delta = atom_coord[0] - centroid[0]
        y_delta = atom_coord[1] - centroid[1]
        if x_delta == 0 and y_delta == 0:
            base_angle = -math.pi / 4.0
        else:
            base_angle = math.atan2(y_delta, x_delta)

        if pair_count <= 1:
            return [base_angle]

        angle_step = math.radians(52.0)
        middle = (pair_count - 1) / 2.0
        return [
            base_angle + (pair_idx - middle) * angle_step
            for pair_idx in range(pair_count)
        ]

    def _svg_lone_pair_dot_pair(
        self,
        atom_idx: int,
        pair_idx: int,
        atom_coord: tuple[float, float],
        angle: float,
        distance: float,
        width: int,
        height: int,
    ) -> list[str]:
        """生成一对孤对电子点的 SVG circle。

        Args:
            atom_idx: 0-based 原子索引。
            pair_idx: 该原子上的第几对孤对电子。
            atom_coord: 原子的 SVG 坐标。
            angle: 孤对电子方向角，单位为弧度。
            distance: 孤对电子点中心相对原子的距离。
            width: SVG 宽度。
            height: SVG 高度。

        Returns:
            两个 `<circle>` SVG 文本。
        """
        x_unit = math.cos(angle)
        y_unit = math.sin(angle)
        x_perp = -y_unit
        y_perp = x_unit
        center_x = atom_coord[0] + x_unit * distance
        center_y = atom_coord[1] + y_unit * distance
        half_gap = self.LONE_PAIR_DOT_GAP / 2.0

        dots: list[str] = []
        for dot_idx, side in enumerate((-1.0, 1.0), start=1):
            x_pos = center_x + x_perp * half_gap * side
            y_pos = center_y + y_perp * half_gap * side
            x_pos = self._clamp(
                x_pos,
                self.lone_pair_dot_radius,
                width - self.lone_pair_dot_radius,
            )
            y_pos = self._clamp(
                y_pos,
                self.lone_pair_dot_radius,
                height - self.lone_pair_dot_radius,
            )
            dots.append(
                (
                    f"<circle class='lone-pair-dot atom-{atom_idx} "
                    f"pair-{pair_idx} dot-{dot_idx}' cx='{x_pos:.1f}' "
                    f"cy='{y_pos:.1f}' r='{self.lone_pair_dot_radius:.1f}' "
                    f"fill='{self.lone_pair_color}' stroke='none'/>\n"
                )
            )

        return dots

    def _svg_radical_dots(
        self,
        atom_coords: list[tuple[float, float]],
        centroid: tuple[float, float],
        charge_notes: dict[int, str],
        lone_pair_counts: dict[int, int],
        radical_counts: dict[int, int],
        width: int,
        height: int,
    ) -> list[str]:
        """生成未成对电子红点的 SVG 片段。"""
        dot_parts: list[str] = []
        for atom_idx, radical_count in radical_counts.items():
            if atom_idx < 0 or atom_idx >= len(atom_coords):
                continue
            atom_coord = atom_coords[atom_idx]
            for radical_idx, angle in enumerate(
                self._radical_angles(
                    atom_coord,
                    centroid,
                    radical_count,
                    lone_pair_counts.get(atom_idx, 0),
                )
            ):
                dot_parts.append(
                    self._svg_radical_dot(
                        atom_idx,
                        radical_idx,
                        atom_coord,
                        angle,
                        -self.LONE_PAIR_OFFSET
                        if atom_idx in charge_notes
                        else self.LONE_PAIR_OFFSET,
                        width,
                        height,
                    )
                )
        return dot_parts

    @staticmethod
    def _radical_angles(
        atom_coord: tuple[float, float],
        centroid: tuple[float, float],
        radical_count: int,
        lone_pair_count: int,
    ) -> list[float]:
        """计算未成对电子相对原子的绘制角度。"""
        x_delta = atom_coord[0] - centroid[0]
        y_delta = atom_coord[1] - centroid[1]
        if x_delta == 0 and y_delta == 0:
            base_angle = -math.pi / 4.0
        else:
            base_angle = math.atan2(y_delta, x_delta)

        if lone_pair_count:
            base_angle += math.radians(52.0)
        if radical_count <= 1:
            return [base_angle]

        angle_step = math.radians(30.0)
        middle = (radical_count - 1) / 2.0
        return [
            base_angle + (radical_idx - middle) * angle_step
            for radical_idx in range(radical_count)
        ]

    def _svg_radical_dot(
        self,
        atom_idx: int,
        radical_idx: int,
        atom_coord: tuple[float, float],
        angle: float,
        distance: float,
        width: int,
        height: int,
    ) -> str:
        """生成一个未成对电子红点的 SVG circle。"""
        x_pos = atom_coord[0] + math.cos(angle) * distance
        y_pos = atom_coord[1] + math.sin(angle) * distance
        x_pos = self._clamp(
            x_pos,
            self.DEFAULT_RADICAL_DOT_RADIUS,
            width - self.DEFAULT_RADICAL_DOT_RADIUS,
        )
        y_pos = self._clamp(
            y_pos,
            self.DEFAULT_RADICAL_DOT_RADIUS,
            height - self.DEFAULT_RADICAL_DOT_RADIUS,
        )
        return (
            f"<circle class='radical-dot atom-{atom_idx} radical-{radical_idx}' "
            f"cx='{x_pos:.1f}' cy='{y_pos:.1f}' "
            f"r='{self.DEFAULT_RADICAL_DOT_RADIUS:.1f}' "
            f"fill='{self.active_space_color}' stroke='none'/>\n"
        )

    @staticmethod
    def _point_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
        """计算一组点的几何中心。

        Args:
            points: SVG 坐标点列表。

        Returns:
            点集的 `(x, y)` 中心；空列表返回 `(0.0, 0.0)`。
        """
        if not points:
            return (0.0, 0.0)
        x_sum = sum(point[0] for point in points)
        y_sum = sum(point[1] for point in points)
        return (x_sum / len(points), y_sum / len(points))

    @classmethod
    def _label_position(
        cls,
        atom_coord: tuple[float, float],
        centroid: tuple[float, float],
        distance: float,
        font_size: float,
        width: int,
        height: int,
    ) -> tuple[float, float]:
        """计算文字标签位置。

        Args:
            atom_coord: 原子的 SVG 坐标。
            centroid: 分子绘图坐标中心。
            distance: 标签相对原子的距离；负值表示朝分子内侧。
            font_size: 标签字号，用于边界裁剪。
            width: SVG 宽度。
            height: SVG 高度。

        Returns:
            标签在 SVG 坐标系中的 `(x, y)` 坐标。
        """
        x_pos, y_pos = atom_coord
        x_delta = x_pos - centroid[0]
        y_delta = y_pos - centroid[1]
        length = (x_delta * x_delta + y_delta * y_delta) ** 0.5
        if length == 0:
            x_unit, y_unit = 0.7, -0.7
        else:
            x_unit, y_unit = x_delta / length, y_delta / length

        label_x = x_pos + x_unit * distance
        label_y = y_pos + y_unit * distance
        return (
            cls._clamp(label_x, font_size, width - font_size),
            cls._clamp(label_y, font_size, height - font_size),
        )

    @staticmethod
    def _clamp(value: float, lower_bound: float, upper_bound: float) -> float:
        """把数值限制在指定范围内。

        Args:
            value: 原始数值。
            lower_bound: 下界。
            upper_bound: 上界。

        Returns:
            被限制在 `[lower_bound, upper_bound]` 范围内的数值。
        """
        return max(lower_bound, min(value, upper_bound))

    @staticmethod
    def _svg_text_label(
        css_class: str,
        text: str,
        position: tuple[float, float],
        color: str,
        font_size: float,
        font_weight: str,
    ) -> str:
        """生成一个 SVG text 标签。

        Args:
            css_class: SVG 元素 class。
            text: 显示文本。
            position: 文本中心坐标。
            color: 文本颜色。
            font_size: 字号。
            font_weight: 字重。

        Returns:
            `<text>` SVG 文本。
        """
        x_pos, y_pos = position
        return (
            f"<text class='{css_class}' x='{x_pos:.1f}' y='{y_pos:.1f}' "
            f"fill='{color}' stroke='none' font-family='Arial, Helvetica, sans-serif' "
            f"font-size='{font_size:.1f}px' font-weight='{font_weight}' "
            "text-anchor='middle' dominant-baseline='middle'>"
            f"{text}</text>\n"
        )

    @staticmethod
    def _extract_svg_body(svg: str) -> str:
        """提取子 SVG 的正文部分。

        Args:
            svg: 完整 SVG 文本。

        Returns:
            去掉顶层 `<svg>` 包装和结尾标签后的正文。
        """
        header_marker = "<!-- END OF HEADER -->\n"
        if header_marker in svg:
            body = svg.split(header_marker, 1)[1]
        else:
            svg_tag_end = svg.find(">")
            body = svg[svg_tag_end + 1 :] if svg_tag_end != -1 else svg

        closing_tag = "</svg>"
        if closing_tag in body:
            body = body.rsplit(closing_tag, 1)[0]
        return body

    @staticmethod
    def _normalize_bond(bond: tuple[int, int]) -> tuple[int, int]:
        """规范化键端点顺序。

        Args:
            bond: 1-based 原子编号组成的二元组。

        Returns:
            按从小到大排序后的键端点。
        """
        return tuple(sorted(bond))

    @staticmethod
    def _validate_bonds(
        mol: Chem.Mol,
        bonds: list[tuple[int, int]],
        source_name: str,
    ) -> None:
        """检查成键配对中的原子编号是否有效。

        Args:
            mol: 用于确定可见原子数量的分子。
            bonds: 1-based 原子编号配对列表。
            source_name: 错误信息中显示的来源名称。

        Returns:
            None.

        Raises:
            ValueError: 任一原子编号超出可见原子范围。
        """
        atom_count = mol.GetNumAtoms()

        for begin_atom, end_atom in bonds:
            for atom_number in (begin_atom, end_atom):
                if atom_number < 1 or atom_number > atom_count:
                    raise ValueError(
                        f"{source_name}: atom number must be 1~{atom_count}, "
                        f"but got {atom_number}."
                    )

    @staticmethod
    def _validate_atom_numbers(
        mol: Chem.Mol,
        atom_numbers: list[int],
        source_name: str,
    ) -> None:
        """检查一组原子编号是否都在可见原子范围内。

        Args:
            mol: 用于确定可见原子数量的分子。
            atom_numbers: 1-based 原子编号列表。
            source_name: 错误信息中显示的来源名称。

        Returns:
            None.

        Raises:
            ValueError: 任一原子编号超出可见原子范围。
        """
        atom_count = mol.GetNumAtoms()
        for atom_number in atom_numbers:
            if atom_number < 1 or atom_number > atom_count:
                raise ValueError(
                    f"{source_name}: atom number must be 1~{atom_count}, "
                    f"but got {atom_number}."
                )

    @staticmethod
    def _validate_atom_groups(
        mol: Chem.Mol,
        atom_groups: list[list[int]],
        source_name: str,
    ) -> None:
        """检查原始电子区域配置。

        Args:
            mol: 用于确定可见原子数量的分子。
            atom_groups: 原始电子区域列表。
            source_name: 错误信息中显示的来源名称。

        Returns:
            None.

        Raises:
            ValueError: 原子组过短，或任一原子编号超出范围。
        """
        atom_count = mol.GetNumAtoms()

        for group_idx, atom_group in enumerate(atom_groups, start=1):
            if len(atom_group) < 2:
                raise ValueError(
                    f"{source_name}[{group_idx}]: at least two atoms are required."
                )
            if len(atom_group) % 2 != 0:
                raise ValueError(
                    f"{source_name}[{group_idx}]: atom numbers must form electron "
                    f"pairs, but got odd length {len(atom_group)}."
                )

            for atom_number in atom_group:
                if atom_number < 1 or atom_number > atom_count:
                    raise ValueError(
                        f"{source_name}[{group_idx}]: atom number must be "
                        f"1~{atom_count}, but got {atom_number}."
                    )

    def _decrease_bond_orders_in_atom_groups(self, editable_mol: Chem.RWMol) -> None:
        """按原始电子区域对键阶进行降级。

        Args:
            editable_mol: 可编辑 RDKit 分子。

        Returns:
            None.
        """
        for atom_group in self.active_bond_atom:
            decrease_count = self._bond_decrease_count_from_atom_group(atom_group)
            candidate_bond_counts = Counter(
                self._collect_existing_bonds_in_atom_group(
                    editable_mol,
                    atom_group,
                )
            )

            decreased_in_group = 0
            while decreased_in_group < decrease_count:
                bond_spec = self._best_decrease_candidate(
                    editable_mol,
                    candidate_bond_counts,
                )
                if bond_spec is None:
                    break

                begin_atom, end_atom = bond_spec
                self._decrease_bond_order(editable_mol, begin_atom, end_atom)
                candidate_bond_counts[bond_spec] -= 1
                if candidate_bond_counts[bond_spec] <= 0:
                    del candidate_bond_counts[bond_spec]
                decreased_in_group += 1

    @classmethod
    def _bond_decrease_count_from_atom_group(cls, atom_group: list[int]) -> int:
        """计算一个原始电子区域需要降级几次真实键。

        `active_bond_atom` 中连续两个原子编号表示一对原始电子。`(a, a)`
        这种 self-pair 表示电子都在同一个原子上，不对应需要降级的真实键。

        Args:
            atom_group: 1-based 原子编号列表，长度必须为偶数。

        Returns:
            需要执行真实键降级的次数。
        """
        decrease_count = 0
        for begin_atom, end_atom in cls._atom_group_electron_pairs(atom_group):
            if begin_atom != end_atom:
                decrease_count += 1
        return decrease_count

    @staticmethod
    def _atom_group_electron_pairs(atom_group: list[int]) -> list[tuple[int, int]]:
        """把原始电子区域按连续两位拆成电子对。

        Args:
            atom_group: 1-based 原子编号列表，长度必须为偶数。

        Returns:
            连续两位组成的电子对列表。
        """
        return [
            (atom_group[index], atom_group[index + 1])
            for index in range(0, len(atom_group), 2)
        ]

    def _best_decrease_candidate(
        self,
        editable_mol: Chem.RWMol,
        candidate_bond_counts: Counter[tuple[int, int]],
    ) -> tuple[int, int] | None:
        """选择下一根最应该降级的键。

        Args:
            editable_mol: 可编辑 RDKit 分子。
            candidate_bond_counts: 候选键及其剩余可降级次数。

        Returns:
            1-based 键端点；如果没有可降级键则返回 `None`。
        """
        candidates = [
            bond_spec
            for bond_spec, remaining_count in candidate_bond_counts.items()
            if remaining_count > 0
            and self._bond_decrease_priority(editable_mol, bond_spec) > 0
        ]
        if not candidates:
            return None

        return max(
            candidates,
            key=lambda bond_spec: (
                self._bond_decrease_priority(editable_mol, bond_spec),
                -bond_spec[0],
                -bond_spec[1],
            ),
        )

    @classmethod
    def _collect_existing_bonds_in_atom_group(
        cls,
        mol: Chem.Mol,
        atom_group: list[int],
    ) -> list[tuple[int, int]]:
        """收集原子组内部当前存在的键。

        Args:
            mol: RDKit 分子。
            atom_group: 1-based 原子编号列表，允许重复。

        Returns:
            原子组内部存在的键列表；重复原子会产生重复候选键。
        """
        atom_number_counts = Counter(atom_group)
        candidate_bonds: list[tuple[int, int]] = []

        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtomIdx() + 1
            end_atom = bond.GetEndAtomIdx() + 1
            multiplicity = min(
                atom_number_counts.get(begin_atom, 0),
                atom_number_counts.get(end_atom, 0),
            )
            if multiplicity > 0:
                candidate_bonds.extend(
                    [cls._normalize_bond((begin_atom, end_atom))] * multiplicity
                )

        return candidate_bonds

    @staticmethod
    def _decrease_bond_order(
        editable_mol: Chem.RWMol,
        begin_atom: int,
        end_atom: int,
    ) -> None:
        """将指定键降一级。

        Args:
            editable_mol: 可编辑 RDKit 分子。
            begin_atom: 起点原子的 1-based 编号。
            end_atom: 终点原子的 1-based 编号。

        Returns:
            None.

        Raises:
            ValueError: 遇到暂不支持的键类型。
        """
        bond = editable_mol.GetBondBetweenAtoms(begin_atom - 1, end_atom - 1)
        if bond is None:
            return

        bond_type = bond.GetBondType()
        if bond_type == Chem.BondType.TRIPLE:
            bond.SetBondType(Chem.BondType.DOUBLE)
        elif bond_type in (Chem.BondType.DOUBLE, Chem.BondType.AROMATIC):
            bond.SetBondType(Chem.BondType.SINGLE)
        elif bond_type == Chem.BondType.SINGLE:
            editable_mol.RemoveBond(begin_atom - 1, end_atom - 1)
            return
        else:
            raise ValueError(
                f"Unsupported bond type for decreasing {begin_atom}-{end_atom}: "
                f"{bond_type}"
            )

        bond.SetIsAromatic(False)

    @staticmethod
    def _bond_decrease_priority(
        editable_mol: Chem.RWMol,
        bond_spec: tuple[int, int],
    ) -> int:
        """计算候选键的降级优先级。

        Args:
            editable_mol: 可编辑 RDKit 分子。
            bond_spec: 1-based 键端点。

        Returns:
            降级优先级；三键最高，双键/芳香键其次，单键最低。
        """
        begin_atom, end_atom = bond_spec
        bond = editable_mol.GetBondBetweenAtoms(begin_atom - 1, end_atom - 1)
        if bond is None:
            return 0

        bond_type = bond.GetBondType()
        if bond_type == Chem.BondType.TRIPLE:
            return 3
        if bond_type in (Chem.BondType.DOUBLE, Chem.BondType.AROMATIC):
            return 2
        if bond_type == Chem.BondType.SINGLE:
            return 1
        return 0

    @staticmethod
    def _increase_bond_order(
        editable_mol: Chem.RWMol,
        begin_atom: int,
        end_atom: int,
    ) -> int | None:
        """将指定原子对的键阶升一级。

        Args:
            editable_mol: 可编辑 RDKit 分子。
            begin_atom: 起点原子的 1-based 编号。
            end_atom: 终点原子的 1-based 编号。

        Returns:
            被升级键的 RDKit bond id；如果已经是三键无法继续升级，则返回 `None`。

        Raises:
            ValueError: 遇到暂不支持的键类型。
        """
        begin_idx = begin_atom - 1
        end_idx = end_atom - 1
        bond = editable_mol.GetBondBetweenAtoms(begin_idx, end_idx)
        upgraded = False

        if bond is None:
            editable_mol.AddBond(begin_idx, end_idx, Chem.BondType.SINGLE)
            bond = editable_mol.GetBondBetweenAtoms(begin_idx, end_idx)
            upgraded = True
        else:
            bond_type = bond.GetBondType()
            if bond_type == Chem.BondType.SINGLE:
                bond.SetBondType(Chem.BondType.DOUBLE)
                upgraded = True
            elif bond_type in (Chem.BondType.DOUBLE, Chem.BondType.AROMATIC):
                bond.SetBondType(Chem.BondType.TRIPLE)
                upgraded = True
            elif bond_type == Chem.BondType.TRIPLE:
                bond.SetBondType(Chem.BondType.TRIPLE)
            else:
                raise ValueError(
                    f"Unsupported bond type for increasing {begin_atom}-{end_atom}: "
                    f"{bond_type}"
                )

        bond.SetIsAromatic(False)
        if not upgraded:
            return None
        return bond.GetIdx()

    @staticmethod
    def _clear_orphan_aromatic_atom_flags(editable_mol: Chem.RWMol) -> None:
        """清理不再连接芳香键的原子芳香标记。

        Args:
            editable_mol: 可编辑 RDKit 分子。

        Returns:
            None.
        """
        for atom in editable_mol.GetAtoms():
            if not any(bond.GetIsAromatic() for bond in atom.GetBonds()):
                atom.SetIsAromatic(False)

    def _charge_notes_from_valence_structure(
        self,
        valence_bond_structure: ValenceBondStructureInfo,
    ) -> dict[int, str]:
        """比较当前价键结构与原始电子分布，得到电荷标签。

        Args:
            valence_bond_structure: 当前价键结构信息。

        Returns:
            0-based 原子索引到电荷符号的映射。
        """
        expected_counts = self._expected_electron_counts()

        actual_counts = {atom_number: 0 for atom_number in expected_counts}
        for begin_atom, end_atom in valence_bond_structure.bond_pairs:
            actual_counts.setdefault(begin_atom, 0)
            actual_counts.setdefault(end_atom, 0)
            if begin_atom in actual_counts:
                actual_counts[begin_atom] += 1
            if end_atom in actual_counts:
                actual_counts[end_atom] += 1

        charge_notes: dict[int, str] = {}
        atom_numbers = sorted(set(expected_counts) | set(actual_counts))
        for atom_number in atom_numbers:
            expected_count = expected_counts.get(atom_number, 0)
            delta = actual_counts.get(atom_number, 0) - expected_count
            note = self._charge_note_from_delta(delta)
            if note:
                charge_notes[atom_number - 1] = note

        return charge_notes

    def _neutral_self_pair_counts_from_valence_structure(
        self,
        valence_bond_structure: ValenceBondStructureInfo,
        charge_notes: dict[int, str],
    ) -> dict[int, int]:
        """统计需要画成孤对电子点的中性 self-pair。

        Args:
            valence_bond_structure: 当前价键结构信息。
            charge_notes: 已按 `active_bond_atom` 比较得到的电荷标签。

        Returns:
            0-based 原子索引到孤对电子对数的映射。只有 self-pair 对应原子
            没有电荷标签时，才把该 self-pair 画成孤对电子。
        """
        lone_pair_counts: dict[int, int] = {}
        for begin_atom, end_atom in valence_bond_structure.bond_pairs:
            if begin_atom != end_atom:
                continue

            atom_number = begin_atom
            atom_idx = atom_number - 1
            if atom_idx in charge_notes:
                continue
            lone_pair_counts[atom_idx] = lone_pair_counts.get(atom_idx, 0) + 1
        return lone_pair_counts

    @staticmethod
    def _radical_counts_from_valence_structure(
        valence_bond_structure: ValenceBondStructureInfo,
    ) -> dict[int, int]:
        """统计当前结构中每个原子上的未成对电子数量。"""
        radical_counts: dict[int, int] = {}
        for atom_number in valence_bond_structure.unpaired_atoms:
            atom_idx = atom_number - 1
            radical_counts[atom_idx] = radical_counts.get(atom_idx, 0) + 1
        return radical_counts

    def _expected_electron_counts(self) -> dict[int, int]:
        """计算原始电子分布。

        Returns:
            1-based 原子编号到原始电子数的映射。
        """
        expected_counts: dict[int, int] = {}
        for atom_group in self.active_bond_atom:
            for atom_number in atom_group:
                expected_counts[atom_number] = expected_counts.get(atom_number, 0) + 1

        return expected_counts

    @staticmethod
    def _charge_note_from_delta(delta: int) -> str:
        """把电子数差值转换为电荷符号。

        Args:
            delta: 当前电子数减去原始电子数。

        Returns:
            电荷标签文本，例如 `"+"`、`"-"`、`"2+"`；无差异返回空字符串。
        """
        if delta > 0:
            return "-" if delta == 1 else f"{delta}-"
        if delta < 0:
            missing_count = abs(delta)
            return "+" if missing_count == 1 else f"{missing_count}+"
        return ""

    def _apply_charge_note_props(
        self,
        mol: Chem.Mol,
        charge_notes: dict[int, str],
    ) -> None:
        """把电荷标签暂存到原子属性里。

        Args:
            mol: RDKit 分子。
            charge_notes: 0-based 原子索引到电荷符号的映射。

        Returns:
            None.
        """
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            if atom.HasProp(self.CHARGE_NOTE_PROP):
                atom.ClearProp(self.CHARGE_NOTE_PROP)
            if atom_idx in charge_notes:
                atom.SetProp(self.CHARGE_NOTE_PROP, charge_notes[atom_idx])

    def _apply_lone_pair_count_props(
        self,
        mol: Chem.Mol,
        lone_pair_counts: dict[int, int],
    ) -> None:
        """把当前价键结构要显示的孤对电子数量暂存到原子属性里。

        Args:
            mol: RDKit 分子。
            lone_pair_counts: 0-based 原子索引到孤对电子对数的映射。

        Returns:
            None.
        """
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            if atom.HasProp(self.LONE_PAIR_COUNT_PROP):
                atom.ClearProp(self.LONE_PAIR_COUNT_PROP)
            if atom_idx in lone_pair_counts:
                atom.SetProp(
                    self.LONE_PAIR_COUNT_PROP,
                    str(lone_pair_counts[atom_idx]),
                )

    def _apply_radical_count_props(
        self,
        mol: Chem.Mol,
        radical_counts: dict[int, int],
    ) -> None:
        """把当前价键结构要显示的未成对电子数量暂存到原子属性里。"""
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            if atom.HasProp(self.RADICAL_COUNT_PROP):
                atom.ClearProp(self.RADICAL_COUNT_PROP)
            if atom_idx in radical_counts:
                atom.SetProp(
                    self.RADICAL_COUNT_PROP,
                    str(radical_counts[atom_idx]),
                )

    def _charge_notes_from_mol(self, mol: Chem.Mol) -> dict[int, str]:
        """从分子原子属性中取出电荷标签。

        Args:
            mol: RDKit 分子。

        Returns:
            0-based 原子索引到电荷符号的映射。
        """
        charge_notes: dict[int, str] = {}
        for atom in mol.GetAtoms():
            if atom.HasProp(self.CHARGE_NOTE_PROP):
                charge_notes[atom.GetIdx()] = atom.GetProp(self.CHARGE_NOTE_PROP)
        return charge_notes

    def _lone_pair_counts_from_mol(self, mol: Chem.Mol) -> dict[int, int]:
        """从分子原子属性中取出孤对电子显示数量。

        Args:
            mol: RDKit 分子。

        Returns:
            0-based 原子索引到孤对电子对数的映射。
        """
        lone_pair_counts: dict[int, int] = {}
        for atom in mol.GetAtoms():
            if atom.HasProp(self.LONE_PAIR_COUNT_PROP):
                lone_pair_counts[atom.GetIdx()] = int(
                    atom.GetProp(self.LONE_PAIR_COUNT_PROP)
                )
        return lone_pair_counts

    def _radical_counts_from_mol(self, mol: Chem.Mol) -> dict[int, int]:
        """从分子原子属性中取出未成对电子显示数量。"""
        radical_counts: dict[int, int] = {}
        for atom in mol.GetAtoms():
            if atom.HasProp(self.RADICAL_COUNT_PROP):
                radical_counts[atom.GetIdx()] = int(
                    atom.GetProp(self.RADICAL_COUNT_PROP)
                )
        return radical_counts

    def _color_upgraded_bond_lines_in_svg(
        self,
        svg: str,
        mol: Chem.Mol,
        upgraded_bond_counts: dict[int, int],
    ) -> str:
        """在 SVG 中高亮被升级的键线。

        Args:
            svg: 原始 SVG 文本。
            mol: 已绘制的 RDKit 分子。
            upgraded_bond_counts: 每根升级键的升级次数。

        Returns:
            高亮升级键后的 SVG 文本。
        """
        if not upgraded_bond_counts:
            return svg

        matches = self._bond_path_matches(svg)
        if not matches:
            return svg

        matches_to_color = self._select_upgraded_line_matches(
            matches,
            mol,
            upgraded_bond_counts,
        )
        return self._recolor_matches(matches_to_color, svg)

    @staticmethod
    def _bond_path_matches(svg: str) -> list[re.Match[str]]:
        """查找 SVG 中所有 RDKit 键路径。

        Args:
            svg: SVG 文本。

        Returns:
            匹配到的键 path 正则匹配对象列表。
        """
        bond_path_pattern = re.compile(
            r"<path class='bond-(\d+) atom-\d+ atom-\d+'[^>]*>"
        )
        return list(bond_path_pattern.finditer(svg))

    def _select_upgraded_line_matches(
        self,
        matches: list[re.Match[str]],
        mol: Chem.Mol,
        upgraded_bond_counts: dict[int, int],
    ) -> list[re.Match[str]]:
        """选出每根升级键中新增加的键线。

        RDKit 可能把 C-O/C-N 等键拆成多个 SVG 片段，因此这里按“整根线”
        选择片段，而不是只改最后一个 path。

        Args:
            matches: 当前 SVG 中所有键 path 匹配。
            mol: 已绘制的 RDKit 分子。
            upgraded_bond_counts: 每根升级键的升级次数。

        Returns:
            需要被重新着色的 path 匹配对象列表。
        """
        selected_matches: list[re.Match[str]] = []
        current_group: list[re.Match[str]] = []
        current_bond_id: int | None = None

        def flush_group() -> None:
            """把当前键 path 组中需要高亮的片段加入结果列表。

            Returns:
                None.
            """
            if current_group and current_bond_id in upgraded_bond_counts:
                segment_count = self._upgraded_line_segment_count(
                    mol,
                    current_bond_id,
                    len(current_group),
                    upgraded_bond_counts[current_bond_id],
                )
                selected_matches.extend(current_group[-segment_count:])

        for match in matches:
            bond_id = int(match.group(1))
            if bond_id == current_bond_id:
                current_group.append(match)
            else:
                flush_group()
                current_bond_id = bond_id
                current_group = [match]

        flush_group()
        return selected_matches

    @classmethod
    def _upgraded_line_segment_count(
        cls,
        mol: Chem.Mol,
        bond_id: int,
        path_count: int,
        upgrade_count: int,
    ) -> int:
        """计算一根升级键需要重染色多少个 SVG path。

        Args:
            mol: 已绘制的 RDKit 分子。
            bond_id: RDKit bond id。
            path_count: 该键在 SVG 中对应的 path 数量。
            upgrade_count: 该键被升级的次数。

        Returns:
            需要从该键 path 组末尾选取的 path 数量。
        """
        line_count = cls._rendered_bond_line_count(mol, bond_id)
        if line_count <= 0 or path_count <= 0:
            return 1
        if path_count % line_count == 0:
            paths_per_line = max(1, path_count // line_count)
            selected_line_count = max(1, min(upgrade_count, line_count))
            return min(path_count, paths_per_line * selected_line_count)
        return min(path_count, max(1, upgrade_count))

    @staticmethod
    def _rendered_bond_line_count(mol: Chem.Mol, bond_id: int) -> int:
        """判断一根键在图中通常绘制为几条线。

        Args:
            mol: RDKit 分子。
            bond_id: RDKit bond id。

        Returns:
            单键为 1，双键/芳香键为 2，三键为 3。
        """
        bond = mol.GetBondWithIdx(bond_id)
        bond_type = bond.GetBondType()
        if bond_type == Chem.BondType.TRIPLE:
            return 3
        if bond_type in (Chem.BondType.DOUBLE, Chem.BondType.AROMATIC):
            return 2
        return 1

    def _recolor_matches(self, matches: list[re.Match[str]], svg: str) -> str:
        """替换一批 SVG path 的颜色和线宽。

        Args:
            matches: 需要重染色的 path 匹配对象。
            svg: SVG 文本。

        Returns:
            替换后的 SVG 文本。
        """
        for match in reversed(matches):
            original_path = match.group(0)
            colored_path = self._recolor_svg_path(original_path)
            svg = svg[: match.start()] + colored_path + svg[match.end() :]

        return svg

    def _recolor_svg_path(self, path_text: str) -> str:
        """修改单个 SVG path 的 stroke 颜色和线宽。

        Args:
            path_text: 单个 `<path>` SVG 文本。

        Returns:
            替换颜色和线宽后的 path 文本。
        """
        path_text = re.sub(
            r"stroke:#[0-9A-Fa-f]{6}",
            f"stroke:{self.active_space_color}",
            path_text,
        )
        return re.sub(
            r"stroke-width:[^;']+",
            f"stroke-width:{self.active_space_width:.1f}px",
            path_text,
        )
