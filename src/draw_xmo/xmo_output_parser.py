from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


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
    """

    index: int
    weight: float
    structure_name: str

    def to_dict(self) -> dict[str, int | float | str]:
        """将权重信息转换为适合 JSON 输出的字典。

        Returns:
            包含结构序号、权重和结构名的字典。
        """
        return {
            "index": self.index,
            "weight": self.weight,
            "structure_name": self.structure_name,
        }


@dataclass(slots=True)
class XmoParsedData:
    """保存从 `.xmo` 文件中提取出的核心数据。

    Args:
        source_file: 被解析的 `.xmo` 文件路径。
        nae: `$ctrl` 中的活性电子数。
        nao: `$ctrl` 中的活性轨道数。
        orb: `$orb` 片段，已转换为 `list[list[int]]`，不包含首行数量说明。
        geo: `$geo` 片段中的原子坐标列表。
        geo_text: `$geo` 片段中的纯文本坐标，不包含 `$geo` 和 `$end` 标记。
        cc_weights: `WEIGHTS OF STRUCTURES` 表中的 CC 权重。
        lowdin_weights: `Lowdin Weights` 表中的 Lowdin 权重。
    """

    source_file: Path
    nae: int
    nao: int
    orb: list[list[int]]
    geo: list[XmoGeometryAtom]
    geo_text: str
    cc_weights: list[XmoStructureWeight]
    lowdin_weights: list[XmoStructureWeight]

    def to_dict(self) -> dict[str, Any]:
        """将完整解析结果转换为适合 JSON 输出的字典。

        Returns:
            包含文件路径、控制参数、轨道、坐标和权重表的字典。
        """
        return {
            "source_file": str(self.source_file),
            "nae": self.nae,
            "nao": self.nao,
            "orb": self.orb,
            "geo": [atom.to_dict() for atom in self.geo],
            "geo_text": self.geo_text,
            "cc_weights": [weight.to_dict() for weight in self.cc_weights],
            "lowdin_weights": [weight.to_dict() for weight in self.lowdin_weights],
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

        return XmoParsedData(
            source_file=self.xmo_file,
            nae=self._parse_required_int(ctrl_lines, "nae"),
            nao=self._parse_required_int(ctrl_lines, "nao"),
            orb=self._parse_orb(orb_lines),
            geo=self._parse_geo(geo_lines),
            geo_text=self._section_plain_text(geo_lines),
            cc_weights=self._parse_weight_table(lines, "WEIGHTS OF STRUCTURES"),
            lowdin_weights=self._parse_weight_table(lines, "Lowdin Weights"),
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
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*(?P<value>\d+)\b", re.IGNORECASE)
        for line in lines:
            match = pattern.search(self._strip_comment(line))
            if match:
                return int(match.group("value"))

        raise ValueError(f"Cannot find integer key {key!r} in $ctrl section.")

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

    def _parse_weight_table(self, lines: list[str], title: str) -> list[XmoStructureWeight]:
        """解析指定标题下方的结构权重表。

        Args:
            lines: 文件全部行。
            title: 权重表标题，例如 `WEIGHTS OF STRUCTURES` 或 `Lowdin Weights`。

        Returns:
            结构权重列表。

        Raises:
            ValueError: 找不到标题，或标题下没有可解析的权重行。
        """
        title_index = self._find_title_line(lines, title)
        weights: list[XmoStructureWeight] = []

        for line in lines[title_index + 1 :]:
            match = self._WEIGHT_ROW_RE.match(line)
            if match:
                weights.append(
                    XmoStructureWeight(
                        index=int(match.group("index")),
                        weight=float(match.group("weight")),
                        structure_name=match.group("structure").strip(),
                    )
                )
                continue

            if weights:
                break

        if not weights:
            raise ValueError(f"Cannot parse any rows under weight table {title!r}.")

        return weights

    def _find_title_line(self, lines: list[str], title: str) -> int:
        """查找权重表标题所在的行号。

        Args:
            lines: 文件全部行。
            title: 需要查找的标题文本。

        Returns:
            标题所在的零基行号。

        Raises:
            ValueError: 找不到标题。
        """
        title_lower = title.lower()
        for index, line in enumerate(lines):
            if title_lower in line.lower():
                return index

        raise ValueError(f"Cannot find title {title!r} in {self.xmo_file}.")

    def _strip_comment(self, line: str) -> str:
        """移除一行中的 `#` 注释并去除首尾空白。

        Args:
            line: 原始文本行。

        Returns:
            去除注释和首尾空白后的文本。
        """
        return line.split("#", 1)[0].strip()
