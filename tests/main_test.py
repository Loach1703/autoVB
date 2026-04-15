import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples" / "unittest"
SKIP_PREFIXES = ("C4H6",)
VALID_SUFFIXES = {".autovb", ".xmi"}
SKIP_OUTPUT_SUFFIX = "_vb.xmi"


def discover_input_files() -> list[str]:
    if not EXAMPLES.exists():
        return []

    files: list[str] = []
    for p in sorted(EXAMPLES.iterdir(), key=lambda x: x.name):
        if not p.is_file():
            continue
        if p.name.startswith(SKIP_PREFIXES):
            continue
        if p.suffix.lower() not in VALID_SUFFIXES:
            continue
        # 过滤运行时生成的 xmi 产物，例如 xxx_vb.xmi / xxx_blw.xmi
        if p.suffix.lower() == ".xmi" and "_" in p.stem:
            continue
        if p.name.endswith(SKIP_OUTPUT_SUFFIX):
            continue
        files.append(p.name)
    return files


INPUT_FILES = discover_input_files()


def build_autovb_command(input_name: str) -> list[str]:
    autovb_bin = shutil.which("autovb")
    if autovb_bin:
        return [autovb_bin, input_name]
    return [sys.executable, "-m", "autoVB.cli", input_name]


@pytest.mark.parametrize("input_name", INPUT_FILES)
def test_main_cli_on_unittest_inputs(input_name: str):
    if not EXAMPLES.exists():
        pytest.skip(f"examples/unittest 目录不存在: {EXAMPLES}")
    if not INPUT_FILES:
        pytest.skip(f"examples/unittest 目录中没有可测试输入文件: {EXAMPLES}")

    proc = subprocess.run(
        build_autovb_command(input_name),
        cwd=str(EXAMPLES),
        capture_output=True,
        text=True,
        timeout=180,
    )

    if proc.stdout:
        print("=== SUBPROCESS STDOUT ===")
        print(proc.stdout)
    if proc.stderr:
        print("=== SUBPROCESS STDERR ===")
        print(proc.stderr)

    expect_error = input_name.startswith("error")
    if expect_error:
        assert proc.returncode != 0, (
            f"{input_name} expected failure, but return code is {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    else:
        assert proc.returncode == 0, (
            f"{input_name} expected success, but return code is {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
