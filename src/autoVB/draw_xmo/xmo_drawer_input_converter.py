from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .molecule_bond_variant_drawer import ValenceBondStructureInfo
from ..io.xmo_output_parser import XmoParsedData, XmoStructureWeight


@dataclass(slots=True)
class XmoDrawerInput:
    """保存从 XMVb 输出转换得到的绘图输入。

    Args:
        xyz_file: 由 `$geo` 坐标块写出的临时 XYZ 文件路径。
        active_bond_atom: 用于初始键降级的活性成键原子区域。
        active_space: 需要绘制的价键结构列表。
        orbital_to_atom: XMVb 活性轨道编号到绘图原子编号的映射。
        weight_table: 当前使用的权重表，取值为 `"cc"` 或 `"lowdin"`。
    """

    xyz_file: Path
    active_bond_atom: list[list[int]]
    active_space: list[ValenceBondStructureInfo]
    orbital_to_atom: dict[int, int]
    weight_table: str


class XmoToDrawerInputConverter:
    """把 `XmoParser` 解析结果转换为 `MoleculeBondVariantDrawer` 所需输入。"""

    _ORBITAL_PAIR_RE = re.compile(r"^(?P<begin>\d+)-(?P<end>\d+)$")
    _INT_RE = re.compile(r"^\d+$")
    _FILENAME_SAFE_RE = re.compile(r"[^0-9A-Za-z]+")

    def __init__(
        self,
        parsed_data: XmoParsedData,
        output_dir: str | Path,
        *,
        hide_hydrogens: bool = True,
        max_structures: int | None = None,
        baseline_index: int = 1,
        weight_table: str = "cc",
    ) -> None:
        """初始化 XMO 到绘图输入的转换器。

        Args:
            parsed_data: `XmoParser` 解析出的 XMO 数据。
            output_dir: 临时 XYZ 和 SVG 输出目录。
            hide_hydrogens: 是否让绘图原子编号遵循“隐藏氢原子后”的编号。
            max_structures: 最多转换多少个 CC 结构；`None` 表示转换全部。
            baseline_index: 作为初始电子排布基准的 CC 权重序号，使用 1-based 编号。
            weight_table: 使用哪一种权重表，`"cc"` 表示 `WEIGHTS OF STRUCTURES`，
                `"lowdin"` 表示 `Lowdin Weights`。
        """
        self.parsed_data = parsed_data
        self.output_dir = Path(output_dir)
        self.hide_hydrogens = hide_hydrogens and not self._active_orbitals_include_hydrogen()
        self.max_structures = max_structures
        self.baseline_index = baseline_index
        self.weight_table = self._normalize_weight_table(weight_table)

    def convert(self) -> XmoDrawerInput:
        """执行转换，得到绘图器可直接使用的数据。

        Returns:
            转换后的 `XmoDrawerInput`。

        Raises:
            ValueError: XMO 中的轨道、原子或结构信息无法映射到绘图输入。
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        xyz_file = self._write_xyz_from_geo()
        orbital_to_atom = self._active_orbital_to_drawer_atom()
        baseline_weight = self._baseline_weight()
        baseline_structure = self._weight_to_structure_info(
            baseline_weight,
            orbital_to_atom,
        )
        active_bond_atom = self._active_bond_atom_from_structure(baseline_structure)
        return XmoDrawerInput(
            xyz_file=xyz_file,
            active_bond_atom=active_bond_atom,
            active_space=[
                self._weight_to_structure_info(weight, orbital_to_atom)
                for weight in self._selected_weights()
            ],
            orbital_to_atom=orbital_to_atom,
            weight_table=self.weight_table,
        )

    def _write_xyz_from_geo(self) -> Path:
        """把 `$geo` 坐标块写成 RDKit 可读取的 XYZ 文件。

        Returns:
            写出的临时 XYZ 文件路径。
        """
        xyz_file = self.output_dir / f"{self.parsed_data.source_file.stem}.xyz"
        xyz_text = "\n".join(
            [
                str(len(self.parsed_data.geo)),
                f"Generated from {self.parsed_data.source_file}",
                self.parsed_data.geo_text,
                "",
            ]
        )
        xyz_file.write_text(xyz_text, encoding="utf-8")
        return xyz_file

    def _active_orbitals_include_hydrogen(self) -> bool:
        data = self.parsed_data
        active_orb_rows = data.orb[-data.nao :] if data.nao > 0 else []
        for orb_row in active_orb_rows:
            if not orb_row:
                continue
            geo_atom_number = orb_row[0]
            if 1 <= geo_atom_number <= len(data.geo):
                if data.geo[geo_atom_number - 1].symbol.upper() == "H":
                    return True
        return False

    def _active_orbital_to_drawer_atom(self) -> dict[int, int]:
        """根据 `$orb` 最后 `nao` 行计算“活性轨道 -> 绘图原子”的映射。

        Returns:
            XMVb 活性轨道编号到绘图原子编号的映射。

        Raises:
            ValueError: `$orb` 行数不足、轨道指向不存在的原子，或活性轨道位于隐藏氢上。
        """
        data = self.parsed_data
        if data.nao <= 0:
            raise ValueError(f"nao must be positive, but got {data.nao}.")
        if len(data.orb) < data.nao:
            raise ValueError(
                f"$orb has {len(data.orb)} rows after skipping its count line, "
                f"but nao={data.nao}."
            )

        # XmoParser 已经跳过 `$orb` 的第一行数量说明；这里取最后 nao 行作为活性轨道。
        active_orb_rows = data.orb[-data.nao :]
        active_start_orbital = len(data.orb) - data.nao + 1
        geo_to_drawer_atom = self._geo_atom_to_drawer_atom()

        orbital_to_atom: dict[int, int] = {}
        for offset, orb_row in enumerate(active_orb_rows):
            if not orb_row:
                raise ValueError(f"Active $orb row {offset + 1} is empty.")

            orbital_number = active_start_orbital + offset
            geo_atom_number = orb_row[0]
            if geo_atom_number < 1 or geo_atom_number > len(data.geo):
                raise ValueError(
                    f"Active orbital {orbital_number} refers to missing $geo atom "
                    f"{geo_atom_number}."
                )
            try:
                orbital_to_atom[orbital_number] = geo_to_drawer_atom[geo_atom_number]
            except KeyError as exc:
                atom = data.geo[geo_atom_number - 1]
                raise ValueError(
                    f"Active orbital {orbital_number} is on hidden atom "
                    f"{geo_atom_number} ({atom.symbol}). Use --show-hydrogens."
                ) from exc

        return orbital_to_atom

    def _geo_atom_to_drawer_atom(self) -> dict[int, int]:
        """建立 `$geo` 原子编号到绘图原子编号的映射。

        Returns:
            1-based `$geo` 原子编号到 1-based 绘图原子编号的映射。
        """
        mapping: dict[int, int] = {}
        drawer_atom_number = 1

        for geo_atom_number, atom in enumerate(self.parsed_data.geo, start=1):
            if self.hide_hydrogens and atom.symbol.upper() == "H":
                continue
            mapping[geo_atom_number] = drawer_atom_number
            drawer_atom_number += 1

        return mapping

    def _baseline_weight(self) -> XmoStructureWeight:
        """取得用于初始电子排布基准的 CC 权重行。

        Returns:
            被选中的权重行。

        Raises:
            ValueError: 找不到 `baseline_index` 指定的权重行。
        """
        for weight in self._weights():
            if weight.index == self.baseline_index:
                return weight

        raise ValueError(
            f"Cannot find {self.weight_table} weight with index {self.baseline_index}."
        )

    def _selected_weights(self) -> list[XmoStructureWeight]:
        """取得需要绘制的权重行。

        Returns:
            权重行列表；如果设置了 `max_structures`，则按权重从大到小排序后返回前 N 个。
        """
        weights = self._weights()
        if self.max_structures is None:
            return list(weights)
        return sorted(
            weights,
            key=lambda weight: (-weight.weight, weight.index),
        )[: self.max_structures]

    def _weights(self) -> list[XmoStructureWeight]:
        """根据 `weight_table` 返回当前要使用的权重表。

        Returns:
            CC 权重或 Lowdin 权重列表。
        """
        if self.weight_table == "lowdin":
            return self.parsed_data.lowdin_weights
        return self.parsed_data.cc_weights

    @staticmethod
    def _normalize_weight_table(weight_table: str) -> str:
        """标准化并检查权重表名称。

        Args:
            weight_table: 用户传入的权重表名称。

        Returns:
            标准化后的权重表名称。

        Raises:
            ValueError: 权重表名称不是 `"cc"` 或 `"lowdin"`。
        """
        normalized_name = weight_table.strip().lower()
        if normalized_name in {"cc", "lowdin"}:
            return normalized_name
        raise ValueError(
            f"weight_table must be 'cc' or 'lowdin', but got {weight_table!r}."
        )

    def _weight_to_structure_info(
        self,
        weight: XmoStructureWeight,
        orbital_to_atom: dict[int, int],
    ) -> ValenceBondStructureInfo:
        """把一行 CC 权重转换成 `ValenceBondStructureInfo`。

        Args:
            weight: XMVb 输出中的一行 CC 权重。
            orbital_to_atom: 活性轨道编号到绘图原子编号的映射。

        Returns:
            转换后的价键结构信息。

        Raises:
            ValueError: 按 `nae - nao` 判断应为孤对电子的位置不是自配对。
        """
        orbital_pairs = weight.orbital_connections or self._active_orbital_pairs(
            weight.structure_name
        )
        leading_lone_pair_count = self._leading_lone_pair_count()
        bond_pairs: list[tuple[int, int]] = []

        for pair_index, (begin_orbital, end_orbital) in enumerate(orbital_pairs):
            begin_atom = self._map_orbital(begin_orbital, orbital_to_atom)
            end_atom = self._map_orbital(end_orbital, orbital_to_atom)

            # 如果 nae > nao，结构名前面的 nae-nao 个自配对项才按孤对电子显示。
            if pair_index < leading_lone_pair_count:
                if begin_atom != end_atom:
                    raise ValueError(
                        "Expected a leading lone-pair orbital pair, but got "
                        f"{begin_orbital}-{end_orbital} in structure "
                        f"{weight.index}: {weight.structure_name}."
                    )
            bond_pairs.append((begin_atom, end_atom))

            # 所有 self-pair 都保留在 bond_pairs 中。绘图类会根据 active_bond_atom
            # 判断它最终应显示为电荷还是孤对电子点。

        return ValenceBondStructureInfo(
            file_suffix=self._file_suffix(weight),
            legend=self._legend(weight, bond_pairs),
            bond_pairs=bond_pairs,
        )

    def _active_orbital_pairs(self, structure_name: str) -> list[tuple[int, int]]:
        """从 XMVb 结构名中解析活性轨道配对。

        Args:
            structure_name: 结构名文本，例如 `1:18   21-22 20-23 19-24`。

        Returns:
            解析得到的活性轨道配对列表。

        Raises:
            ValueError: 结构名中存在没有配成对的自配对轨道编号。
        """
        pairs: list[tuple[int, int]] = []
        pending_single_orbital: int | None = None

        for token in structure_name.split():
            # `1:18` 这类字段表示非活性/闭壳层范围，不参与成键映射。
            if ":" in token:
                continue

            pair_match = self._ORBITAL_PAIR_RE.match(token)
            if pair_match:
                if pending_single_orbital is not None:
                    raise ValueError(
                        f"Incomplete orbital self-pair before {token!r} "
                        f"in structure {structure_name!r}."
                    )
                pairs.append(
                    (
                        int(pair_match.group("begin")),
                        int(pair_match.group("end")),
                    )
                )
                continue

            # XMVb 用 `19 19` 这种两个连续整数表示同一轨道上的一对电子。
            if self._INT_RE.match(token):
                orbital_number = int(token)
                if pending_single_orbital is None:
                    pending_single_orbital = orbital_number
                else:
                    pairs.append((pending_single_orbital, orbital_number))
                    pending_single_orbital = None

        if pending_single_orbital is not None:
            raise ValueError(
                f"Incomplete orbital self-pair at the end of structure "
                f"{structure_name!r}."
            )

        return pairs

    def _leading_lone_pair_count(self) -> int:
        """根据 `nae - nao` 计算结构名前面有几个孤对电子自配对项。

        Returns:
            结构名前面应解释为孤对电子的自配对项数量。
        """
        return max(0, self.parsed_data.nae - self.parsed_data.nao)

    def _map_orbital(
        self,
        orbital_number: int,
        orbital_to_atom: dict[int, int],
    ) -> int:
        """把单个活性轨道编号映射为绘图原子编号。

        Args:
            orbital_number: XMVb 轨道编号。
            orbital_to_atom: 活性轨道编号到绘图原子编号的映射。

        Returns:
            绘图原子编号。

        Raises:
            ValueError: 该轨道不属于最后 `nao` 个活性轨道。
        """
        try:
            return orbital_to_atom[orbital_number]
        except KeyError as exc:
            raise ValueError(
                f"Orbital {orbital_number} is not one of the last nao="
                f"{self.parsed_data.nao} active orbitals."
            ) from exc

    @staticmethod
    def _active_bond_atom_from_structure(
        structure: ValenceBondStructureInfo,
    ) -> list[list[int]]:
        """由基准价键结构生成 `active_bond_atom`。

        Args:
            structure: 作为初始电子排布基准的价键结构。

        Returns:
            展平后的活性成键原子组；如果基准结构只有孤对电子，则返回空列表。
        """
        active_atoms: list[int] = []
        for begin_atom, end_atom in structure.bond_pairs:
            active_atoms.extend([begin_atom, end_atom])

        return [active_atoms] if active_atoms else []

    def _file_suffix(self, weight: XmoStructureWeight) -> str:
        """为单个 CC 结构生成安全的 SVG 文件名后缀。

        Args:
            weight: 一行 CC 权重。

        Returns:
            可以放入文件名的后缀字符串。
        """
        safe_name = self._FILENAME_SAFE_RE.sub("_", weight.structure_name).strip("_")
        return f"{self.weight_table}_{weight.index:03d}_{safe_name}"

    def _legend(
        self,
        weight: XmoStructureWeight,
        bond_pairs: list[tuple[int, int]],
    ) -> str:
        """生成显示在单个结构下方的图例文本。

        Args:
            weight: 一行权重。
            bond_pairs: 已映射到原子编号的成键配对。

        Returns:
            简短图例文本。
        """
        pair_text = " ".join(f"{begin}-{end}" for begin, end in bond_pairs)
        table_label = "Lowdin" if self.weight_table == "lowdin" else "CC"
        return f"{table_label} {weight.index} w={weight.weight:.5f}: {pair_text}"
