from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..draw_xmo.molecule_bond_variant_drawer import MoleculeBondVariantDrawer
    from ..draw_xmo.xmo_drawer_input_converter import XmoDrawerInput
    from ..io.xmo_output_parser import XmoParsedData

DEFAULT_XMO_MAX_STRUCTURES = 20
DEFAULT_XMO_WEIGHT_TABLE = "lowdin"
DEFAULT_XMO_ACTIVE_SPACE_COLOR = "#B00000"
DEFAULT_XMO_ACTIVE_SPACE_WIDTH = 3.0
DEFAULT_XMO_STRUCTURES_PER_ROW = 2


@dataclass
class Xmo2SvgResult:
    """保存一次 XMO 绘图任务的结果和报告信息。"""

    draw_result: "MoleculeBondVariantDrawer.Result"
    parsed_data: "XmoParsedData"
    drawer_input: "XmoDrawerInput"
    connectivity: str
    projection: str

    @property
    def xyz_file(self) -> Path:
        return self.draw_result.xyz_file

    @property
    def output_dir(self) -> Path:
        return self.draw_result.output_dir

    @property
    def displayed_atom_count(self) -> int:
        return self.draw_result.displayed_atom_count

    @property
    def written_files(self) -> list[Path]:
        return self.draw_result.written_files


def xmo2svg_file(
    xmo_file: str | Path,
    *,
    connectivity: str = "orb",
    weight_table: str = DEFAULT_XMO_WEIGHT_TABLE,
    max_structures: int | None = DEFAULT_XMO_MAX_STRUCTURES,
    baseline_index: int | None = None,
    charge: int = 0,
    hide_hydrogens: bool = True,
    write_individual_svgs: bool = False,
    show_atom_labels: bool = True,
    show_lone_pairs: bool = True,
    structures_per_row: int = DEFAULT_XMO_STRUCTURES_PER_ROW,
    projection: str = "rdkit",
    condense_hydrogens: bool = True,
    show_connection_labels: bool = True,
    rename_grid: bool = True,
) -> Xmo2SvgResult:
    """读取 XMO 并按指定连接关系生成价键结构 SVG。"""
    from ..draw_xmo.molecule_bond_variant_drawer import MoleculeBondVariantDrawer
    from ..draw_xmo.orbital_connectivity_molecule_drawer import (
        OrbitalConnectivityMoleculeDrawer,
    )
    from ..draw_xmo.xmo_drawer_input_converter import XmoToDrawerInputConverter
    from ..io.xmo_output_parser import XmoParser

    xmo_path = Path(xmo_file)
    if not xmo_path.exists():
        raise FileNotFoundError(f"XMO file not found: {xmo_path}")
    if not xmo_path.is_file():
        raise ValueError(f"XMO path is not a file: {xmo_path}")

    output_dir = xmo_path.parent
    parsed_data = XmoParser(xmo_path).parse()
    converter = XmoToDrawerInputConverter(
        parsed_data,
        output_dir,
        hide_hydrogens=hide_hydrogens,
        max_structures=max_structures,
        baseline_index=baseline_index,
        weight_table=weight_table,
        show_connection_labels=show_connection_labels,
    )
    drawer_input = converter.convert()
    hide_hydrogens = converter.hide_hydrogens

    drawer_kwargs = dict(
        xyz_file=drawer_input.xyz_file,
        output_dir=output_dir,
        charge=charge,
        active_bond_atom=drawer_input.active_bond_atom,
        active_space=drawer_input.active_space,
        baseline_unpaired_atoms=drawer_input.baseline_unpaired_atoms,
        active_space_color=DEFAULT_XMO_ACTIVE_SPACE_COLOR,
        active_space_width=DEFAULT_XMO_ACTIVE_SPACE_WIDTH,
        color_active_space=True,
        show_atom_labels=show_atom_labels,
        hide_hydrogens=hide_hydrogens,
        show_lone_pairs=show_lone_pairs,
        write_individual_svgs=write_individual_svgs,
        structures_per_row=structures_per_row,
    )
    if connectivity == "orb":
        drawer = OrbitalConnectivityMoleculeDrawer(
            **drawer_kwargs,
            orbital_atom_rows=parsed_data.orb,
            projection=projection,
            condense_hydrogens=condense_hydrogens,
        )
    elif connectivity == "rdkit":
        drawer = MoleculeBondVariantDrawer(**drawer_kwargs)
    else:
        raise ValueError(f"Unsupported connectivity mode: {connectivity}")

    draw_result = drawer.draw()
    if rename_grid:
        grid_path = output_dir / f"{xmo_path.stem}_grid.svg"
        output_path = output_dir / f"{xmo_path.stem}.svg"
        grid_path.replace(output_path)
        draw_result.written_files = [
            output_path if written_file == grid_path else written_file
            for written_file in draw_result.written_files
        ]

    return Xmo2SvgResult(
        draw_result=draw_result,
        parsed_data=parsed_data,
        drawer_input=drawer_input,
        connectivity=connectivity,
        projection=projection,
    )


def xmo2svg_report_lines(result: Xmo2SvgResult) -> list[str]:
    """生成 XMO 绘图任务的用户可读报告行。"""
    from ..utils import constants

    if not isinstance(result, Xmo2SvgResult):
        return []

    return [
        f"XMO2SVG version: {constants.VERSION}",
        f"Read XMO from: {result.parsed_data.source_file.resolve()}",
        f"Generated XYZ: {result.drawer_input.xyz_file.resolve()}",
        f"Active orbital -> atom: {result.drawer_input.orbital_to_atom}",
        f"Weight table: {result.drawer_input.weight_table}",
        f"active_bond_atom: {result.drawer_input.active_bond_atom}",
        f"Connectivity source: {result.connectivity}",
        f"Projection: {result.projection}",
        f"Drawn structures: {len(result.drawer_input.active_space)}",
        f"Output directory: {result.output_dir.resolve()}",
        *[f" - {out_file.name}" for out_file in result.written_files],
    ]
