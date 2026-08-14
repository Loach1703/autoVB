import argparse
from pathlib import Path

from ..io.readers import read_xmi_and_orb
from ..io.writers import write_xmi_with_orbital_guess
from ..utils.constants import SUPPORTED_METHODS


def xmo2xmi_file(
    xmi_file: str | Path,
    output: str | Path | None = None,
    method: str | None = None,
    iscf: int | None = None,
) -> Path:
    """用同名 ORB 文件替换 XMI 中的轨道初猜。"""
    xmi_path = Path(xmi_file)
    xmi_text, orb_text = read_xmi_and_orb(xmi_path)
    output_path = (
        Path(output)
        if output is not None
        else xmi_path.with_name(f"{xmi_path.stem}_new.xmi")
    )
    return write_xmi_with_orbital_guess(
        xmi_text,
        orb_text,
        output_path,
        method=method,
        iscf=iscf,
    )


def xmo2xmi(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="xmo2xmi",
        description="Replace an XMVB .xmi initial guess with its matching .orb file.",
    )
    parser.add_argument("xmi_file", type=Path, help="input .xmi file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output path, default: <xmi_file>_new.xmi",
    )
    parser.add_argument(
        "--method",
        choices=SUPPORTED_METHODS,
        default=None,
        help="replace the method in $ctrl, for example vbpt2 or bovb",
    )
    parser.add_argument(
        "--iscf",
        type=int,
        choices=(2, 5, 6),
        default=None,
        help="replace iscf in $ctrl",
    )
    args = parser.parse_args(argv)

    output_path = xmo2xmi_file(
        args.xmi_file,
        args.output,
        args.method,
        args.iscf,
    )
    print(f"XMI written to: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(xmo2xmi())
