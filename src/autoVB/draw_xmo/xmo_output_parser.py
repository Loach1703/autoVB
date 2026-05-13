from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from ..constants import SUPPORTED_METHODS


@dataclass(slots=True)
class XmoGeometryAtom:
    """保存 `$geo` 中一行原子坐标。

    Args:
        symbol: 原子符号，例如 `C` 或 `H`。
        x: X 坐标。
        y: Y 坐标。
        z: Z 坐标。
    """

    symbol: str
    x: float
    y: float
    z: float

    def to_dict(self) -> dict[str, str | float]:
        """将原子坐标转换为适合 JSON 输出的字典。

        Returns:
            包含原子符号和三维坐标的字典。
        """
        return {
            "symbol": self.symbol,
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }


@dataclass(slots=True)
class XmoStructureWeight:
    """保存一个 VB 结构的权重信息。

    Args:
        index: 结构序号。
        weight: 结构权重。
        structure_name: XMVb 输出中的结构描述文本。
        inactive_orbital_ranges: 结构描述中的非活性/闭壳层轨道范围。
        orbital_connections: 结构描述中的活性轨道连接。
        atom_connections: 活性轨道按 `$orb` 映射后的原子连接。
        flat_orbitals: 展平后的活性轨道编号序列。
        flat_atoms: 展平后的原子编号序列。
    """

    index: int
    weight: float
    structure_name: str
    inactive_orbital_ranges: list[tuple[int, int]] = field(default_factory=list)
    orbital_connections: list[tuple[int, int]] = field(default_factory=list)
    atom_connections: list[tuple[int, int]] = field(default_factory=list)
    flat_orbitals: list[int] = field(default_factory=list)
    flat_atoms: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """将权重信息转换为适合 JSON 输出的字典。

        Returns:
            包含结构序号、权重、结构名和结构化连接信息的字典。
        """
        return {
            "index": self.index,
            "weight": self.weight,
            "structure_name": self.structure_name,
            "inactive_orbital_ranges": self._pair_dicts(
                self.inactive_orbital_ranges
            ),
            "orbital_connections": self._pair_dicts(self.orbital_connections),
            "atom_connections": self._pair_dicts(self.atom_connections),
            "flat_orbitals": self.flat_orbitals,
            "flat_atoms": self.flat_atoms,
        }

    @staticmethod
    def _pair_dicts(pairs: list[tuple[int, int]]) -> list[dict[str, int]]:
        return [{"begin": begin, "end": end} for begin, end in pairs]


@dataclass(slots=True)
class XmoParsedData:
    """保存从 `.xmo` 文件中提取出的核心数据。

    Args:
        source_file: 被解析的 `.xmo` 文件路径。
        method: `$ctrl` 中的计算方法。
        basis: `$ctrl` 中的基组名称。
        ctrl_options: `$ctrl` 中解析出的控制选项。
        nae: `$ctrl` 中的活性电子数。
        nao: `$ctrl` 中的活性轨道数。
        orb: `$orb` 片段，已转换为 `list[list[int]]`，不包含首行数量说明。
        orbital_to_atom: 活性轨道编号到 `$geo` 原子编号的映射。
        geo: `$geo` 片段中的原子坐标列表。
        geo_text: `$geo` 片段中的纯文本坐标，不包含 `$geo` 和 `$end` 标记。
        cc_weights: `WEIGHTS OF STRUCTURES` 表中的 CC 权重。
        lowdin_weights: `Lowdin Weights` 表中的 Lowdin 权重。
        inverse_weights: `Inverse Weights` 表中的权重。
        renormalized_weights: `Renormalized Weights` 表中的权重。
    """

    source_file: Path
    method: str
    basis: str
    ctrl_options: dict[str, str | bool]
    nae: int
    nao: int
    orb: list[list[int]]
    orbital_to_atom: dict[int, int]
    geo: list[XmoGeometryAtom]
    geo_text: str
    cc_weights: list[XmoStructureWeight]
    lowdin_weights: list[XmoStructureWeight]
    inverse_weights: list[XmoStructureWeight]
    renormalized_weights: list[XmoStructureWeight]

    def to_dict(self) -> dict[str, Any]:
        """将完整解析结果转换为适合 JSON 输出的字典。

        Returns:
            包含文件路径、控制参数、轨道、坐标和权重表的字典。
        """
        return {
            "source_file": str(self.source_file),
            "method": self.method,
            "basis": self.basis,
            "ctrl_options": self.ctrl_options,
            "nae": self.nae,
            "nao": self.nao,
            "orb": self.orb,
            "orbital_to_atom": self.orbital_to_atom,
            "geo": [atom.to_dict() for atom in self.geo],
            "geo_text": self.geo_text,
            "cc_weights": [weight.to_dict() for weight in self.cc_weights],
            "lowdin_weights": [weight.to_dict() for weight in self.lowdin_weights],
            "inverse_weights": [weight.to_dict() for weight in self.inverse_weights],
            "renormalized_weights": [
                weight.to_dict() for weight in self.renormalized_weights
            ],
        }

    def save_geo_text(self, output_file: str | Path) -> None:
        """将 `$geo` 纯文本坐标写入文件。

        Args:
            output_file: 输出文件路径。
        """
        output_path = Path(output_file)
        text = self.geo_text
        if text and not text.endswith("\n"):
            text += "\n"
        output_path.write_text(text, encoding="utf-8")


class XmoParser:
    """解析 XMVb `.xmo` 输出文件中的输入片段和权重表。"""

    _SECTION_END = "$end"
    _WEIGHT_ROW_RE = re.compile(
        r"^\s*(?P<index>\d+)\s+"
        r"(?P<weight>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)\s+"
        r"\*+\s+"
        r"(?P<structure>.+?)\s*$"
    )
    _REPEAT_INT_RE = re.compile(r"^(?P<value>[-+]?\d+)\*(?P<count>\d+)$")

    def __init__(self, xmo_file: str | Path) -> None:
        """初始化解析器。

        Args:
            xmo_file: 需要解析的 `.xmo` 文件路径。
        """
        self.xmo_file = Path(xmo_file)

    def parse(self) -> XmoParsedData:
        """解析 `.xmo` 文件并返回结构化结果。

        Returns:
            包含 `nae`、`nao`、`$orb`、`$geo`、CC 权重和 Lowdin 权重的解析结果。

        Raises:
            FileNotFoundError: 输入文件不存在。
            ValueError: 文件中缺少必需片段或片段格式不符合预期。
        """
        lines = self._read_lines()
        ctrl_lines = self._section_lines(lines, "$ctrl")
        orb_lines = self._section_lines(lines, "$orb")
        geo_lines = self._section_lines(lines, "$geo")
        ctrl_options = self._parse_ctrl_options(ctrl_lines)
        nao = self._parse_required_int(ctrl_lines, "nao")
        orb = self._parse_orb(orb_lines)
        orbital_to_atom = self._active_orbital_to_atom(orb, nao)

        return XmoParsedData(
            source_file=self.xmo_file,
            method=self._parse_method(ctrl_lines),
            basis=str(ctrl_options.get("basis", "")),
            ctrl_options=ctrl_options,
            nae=self._parse_required_int(ctrl_lines, "nae"),
            nao=nao,
            orb=orb,
            orbital_to_atom=orbital_to_atom,
            geo=self._parse_geo(geo_lines),
            geo_text=self._section_plain_text(geo_lines),
            cc_weights=self._parse_weight_table(
                lines,
                "WEIGHTS OF STRUCTURES",
                orbital_to_atom,
            ),
            lowdin_weights=self._parse_weight_table(
                lines,
                "Lowdin Weights",
                orbital_to_atom,
            ),
            inverse_weights=self._parse_weight_table(
                lines,
                "Inverse Weights",
                orbital_to_atom,
            ),
            renormalized_weights=self._parse_weight_table(
                lines,
                "Renormalized Weights",
                orbital_to_atom,
            ),
        )

    def _read_lines(self) -> list[str]:
        """读取 `.xmo` 文件的全部文本行。

        Returns:
            文件的行列表。

        Raises:
            FileNotFoundError: 输入文件不存在。
        """
        return self.xmo_file.read_text(encoding="utf-8", errors="replace").splitlines()

    def _section_lines(self, lines: list[str], section_name: str) -> list[str]:
        """按 `$xxx` 和 `$end` 提取一个输入片段。

        Args:
            lines: 文件全部行。
            section_name: 需要提取的片段名，例如 `$orb`。

        Returns:
            片段内部的行，不包含片段起止标记。

        Raises:
            ValueError: 找不到片段开始或结束标记。
        """
        normalized_section = section_name.lower()
        start_index: int | None = None

        for index, line in enumerate(lines):
            if line.strip().lower() == normalized_section:
                start_index = index
                break

        if start_index is None:
            raise ValueError(f"Cannot find section {section_name!r} in {self.xmo_file}.")

        collected: list[str] = []
        for line in lines[start_index + 1 :]:
            if line.strip().lower() == self._SECTION_END:
                return collected
            collected.append(line)

        raise ValueError(f"Section {section_name!r} is missing {self._SECTION_END!r}.")

    def _parse_required_int(self, lines: list[str], key: str) -> int:
        """从 `$ctrl` 行中读取必需的整数参数。

        Args:
            lines: `$ctrl` 片段内部的行。
            key: 参数名，例如 `nae` 或 `nao`。

        Returns:
            读取到的整数值。

        Raises:
            ValueError: 找不到该参数。
        """
        for ctrl_key, value in self._iter_ctrl_items(lines):
            if ctrl_key == key.lower() and value is not None:
                return int(value)

        raise ValueError(f"Cannot find integer key {key!r} in $ctrl section.")

    def _parse_method(self, lines: list[str]) -> str:
        """从 `$ctrl` 中读取计算方法。

        Args:
            lines: `$ctrl` 片段内部的行。

        Returns:
            方法名，例如 `vbscf`。

        Raises:
            ValueError: 找不到方法行。
        """
        supported_methods = {method.lower() for method in SUPPORTED_METHODS}
        first_bare_token = ""

        for key, value in self._iter_ctrl_items(lines):
            if key in supported_methods:
                return f"{key}={value}" if value is not None else key
            if value is None and not first_bare_token:
                first_bare_token = key

        if first_bare_token:
            return first_bare_token

        raise ValueError("Cannot find method in $ctrl section.")

    def _parse_ctrl_options(self, lines: list[str]) -> dict[str, str | bool]:
        """解析 `$ctrl` 中的键值和开关型选项。

        Args:
            lines: `$ctrl` 片段内部的行。

        Returns:
            控制选项字典；无等号的选项保存为 `True`。
        """
        options: dict[str, str | bool] = {}
        method = self._parse_method(lines)
        method_key = method.split("=", 1)[0].lower()
        options["method"] = method

        for key, value in self._iter_ctrl_items(lines):
            if key == method_key:
                continue
            options[key] = value if value is not None else True

        return options

    def _iter_ctrl_items(self, lines: list[str]) -> Iterator[tuple[str, str | None]]:
        """按空白拆分 `$ctrl`，生成 `(key, value)` 项。"""
        for line in lines:
            tokens = self._strip_comment(line).split()
            index = 0
            while index < len(tokens):
                token = tokens[index]
                if token == "=":
                    index += 1
                    continue

                if "=" in token:
                    key, value = token.split("=", 1)
                    if value == "" and index + 1 < len(tokens):
                        index += 1
                        value = tokens[index]
                    if key:
                        yield key.lower(), value
                elif index + 2 < len(tokens) and tokens[index + 1] == "=":
                    yield token.lower(), tokens[index + 2]
                    index += 2
                else:
                    yield token.lower(), None

                index += 1

    def _parse_orb(self, lines: list[str]) -> list[list[int]]:
        """解析 `$orb` 片段为整数二维列表。

        首个有效数据行表示后续轨道信息的数量说明，不作为轨道数据返回。

        Args:
            lines: `$orb` 片段内部的行。

        Returns:
            每一行为一个整数列表的 `$orb` 数据；例如 `1*6` 会展开为六个 `1`。

        Raises:
            ValueError: 遇到无法转换为整数的 token。
        """
        orb_rows: list[list[int]] = []
        skipped_count_line = False

        for line_number, line in enumerate(lines, start=1):
            clean_line = self._strip_comment(line)
            if not clean_line:
                continue

            if not skipped_count_line:
                skipped_count_line = True
                continue

            row: list[int] = []
            for token in clean_line.split():
                repeat_match = self._REPEAT_INT_RE.match(token)
                if repeat_match:
                    value = int(repeat_match.group("value"))
                    count = int(repeat_match.group("count"))
                    row.extend([value] * count)
                    continue

                try:
                    row.append(int(token))
                except ValueError as exc:
                    raise ValueError(f"$orb line {line_number}: invalid integer token {token!r}.") from exc

            if row:
                orb_rows.append(row)

        return orb_rows

    def _active_orbital_to_atom(
        self,
        orb: list[list[int]],
        nao: int,
    ) -> dict[int, int]:
        """根据 `$orb` 最后 `nao` 行建立活性轨道到原子的映射。"""
        active_orb_rows = orb[-nao:]
        active_start_orbital = len(orb) - nao + 1
        return {
            active_start_orbital + offset: orb_row[0]
            for offset, orb_row in enumerate(active_orb_rows)
            if orb_row
        }

    def _parse_geo(self, lines: list[str]) -> list[XmoGeometryAtom]:
        """解析 `$geo` 片段为原子坐标列表。

        Args:
            lines: `$geo` 片段内部的行。

        Returns:
            原子坐标列表。

        Raises:
            ValueError: 坐标行字段不足，或坐标值不能转换为浮点数。
        """
        atoms: list[XmoGeometryAtom] = []

        for line_number, line in enumerate(lines, start=1):
            clean_line = self._strip_comment(line)
            if not clean_line:
                continue

            parts = clean_line.split()
            if len(parts) < 4:
                raise ValueError(f"$geo line {line_number}: expected symbol and 3 coordinates.")

            try:
                atoms.append(
                    XmoGeometryAtom(
                        symbol=parts[0],
                        x=float(parts[1]),
                        y=float(parts[2]),
                        z=float(parts[3]),
                    )
                )
            except ValueError as exc:
                raise ValueError(f"$geo line {line_number}: invalid coordinate value.") from exc

        return atoms

    def _section_plain_text(self, lines: list[str]) -> str:
        """将输入片段转换为去注释后的纯文本。

        Args:
            lines: 片段内部的原始行。

        Returns:
            去除空行和 `#` 注释后的纯文本，多行之间用换行符连接。
        """
        clean_lines: list[str] = []
        for line in lines:
            clean_line = self._strip_comment(line)
            if clean_line:
                clean_lines.append(clean_line)
        return "\n".join(clean_lines)

    def _parse_weight_table(
        self,
        lines: list[str],
        title: str,
        orbital_to_atom: dict[int, int],
    ) -> list[XmoStructureWeight]:
        """解析指定标题下方的结构权重表。

        Args:
            lines: 文件全部行。
            title: 权重表标题，例如 `WEIGHTS OF STRUCTURES` 或 `Lowdin Weights`。

        Returns:
            结构权重列表。

        如果找不到标题或标题下没有可解析行，会打印警告并返回空列表。
        """
        title_index = self._find_title_line(lines, title)
        if title_index is None:
            print(f"Warning: cannot find weight table {title!r} in {self.xmo_file}.")
            return []

        weights: list[XmoStructureWeight] = []

        for line in lines[title_index + 1 :]:
            match = self._WEIGHT_ROW_RE.match(line)
            if match:
                structure_name = match.group("structure").strip()
                orbital_connections = self._parse_structure_orbital_connections(
                    structure_name
                )
                atom_connections = self._map_orbital_connections(
                    orbital_connections,
                    orbital_to_atom,
                )
                weights.append(
                    XmoStructureWeight(
                        index=int(match.group("index")),
                        weight=float(match.group("weight")),
                        structure_name=structure_name,
                        inactive_orbital_ranges=self._parse_inactive_orbital_ranges(
                            structure_name
                        ),
                        orbital_connections=orbital_connections,
                        atom_connections=atom_connections,
                        flat_orbitals=self._flatten_pairs(orbital_connections),
                        flat_atoms=self._flatten_pairs(atom_connections),
                    )
                )
                continue

            if weights:
                break

        if not weights:
            print(
                f"Warning: cannot parse any rows under weight table {title!r} "
                f"in {self.xmo_file}."
            )

        return weights

    def _parse_inactive_orbital_ranges(
        self,
        structure_name: str,
    ) -> list[tuple[int, int]]:
        """解析 `1:30` 这类非活性/闭壳层轨道范围。"""
        ranges: list[tuple[int, int]] = []
        for token in structure_name.split():
            if ":" not in token:
                continue
            begin, end = token.split(":", 1)
            if begin.isdigit() and end.isdigit():
                ranges.append((int(begin), int(end)))
        return ranges

    def _parse_structure_orbital_connections(
        self,
        structure_name: str,
    ) -> list[tuple[int, int]]:
        """解析结构名中的活性轨道连接。"""
        connections: list[tuple[int, int]] = []
        pending_single_orbital: int | None = None

        for token in structure_name.split():
            if ":" in token:
                continue

            if "-" in token:
                if pending_single_orbital is not None:
                    raise ValueError(
                        f"Incomplete orbital self-pair before {token!r} "
                        f"in structure {structure_name!r}."
                    )
                begin, end = token.split("-", 1)
                if begin.isdigit() and end.isdigit():
                    connections.append((int(begin), int(end)))
                continue

            if token.isdigit():
                orbital_number = int(token)
                if pending_single_orbital is None:
                    pending_single_orbital = orbital_number
                else:
                    connections.append((pending_single_orbital, orbital_number))
                    pending_single_orbital = None

        if pending_single_orbital is not None:
            raise ValueError(
                f"Incomplete orbital self-pair at the end of structure "
                f"{structure_name!r}."
            )

        return connections

    def _map_orbital_connections(
        self,
        orbital_connections: list[tuple[int, int]],
        orbital_to_atom: dict[int, int],
    ) -> list[tuple[int, int]]:
        """把活性轨道连接映射为原子连接。"""
        return [
            (orbital_to_atom[begin], orbital_to_atom[end])
            for begin, end in orbital_connections
        ]

    @staticmethod
    def _flatten_pairs(pairs: list[tuple[int, int]]) -> list[int]:
        """把连接对展平为连续整数序列。"""
        return [value for pair in pairs for value in pair]

    def _find_title_line(self, lines: list[str], title: str) -> int | None:
        """查找权重表标题所在的行号。

        Args:
            lines: 文件全部行。
            title: 需要查找的标题文本。

        Returns:
            标题所在的零基行号；找不到时返回 `None`。
        """
        title_lower = title.lower()
        for index, line in enumerate(lines):
            if title_lower in line.lower():
                return index

        return None

    def _strip_comment(self, line: str) -> str:
        """移除一行中的 `#` 注释并去除首尾空白。

        Args:
            line: 原始文本行。

        Returns:
            去除注释和首尾空白后的文本。
        """
        return line.split("#", 1)[0].strip()
