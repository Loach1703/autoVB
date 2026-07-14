import argparse
from pathlib import Path
from types import SimpleNamespace

from ..io.writers import write_json_summary
from ..io.xmo_output_parser import XmoParser


def xmo2json_file(xmo_file: str | Path, output: str | Path | None = None) -> Path:
    """将指定的 XMVB 输出文件转换为 JSON 摘要。"""
    xmo_path = Path(xmo_file)
    parsed_data = XmoParser(xmo_path).parse()
    input_data = SimpleNamespace(
        filename=xmo_path.stem,
        title=xmo_path.stem,
        charge=int(parsed_data.ctrl_options.get("ncharge", 0)),
        spin=int(parsed_data.ctrl_options.get("nmul", 1)),
    )
    output_path = Path(output) if output else xmo_path.with_suffix(".json")
    return write_json_summary(input_data, parsed_data, output_path)


def xmo2json(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="xmo2json",
        description="Convert an XMVB .xmo output file to a JSON summary.",
    )
    parser.add_argument("xmo_file", type=Path, help="input .xmo file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output JSON path, default: <xmo_file>.json",
    )
    args = parser.parse_args(argv)

    output_path = xmo2json_file(args.xmo_file, args.output)
    print(f"JSON summary written to: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(xmo2json())
