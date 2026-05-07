import pytest
import sys
import subprocess
from pathlib import Path
from autoVB import cli

def test_autovb_nbo_help_exits_0():
    with pytest.raises(SystemExit) as exc:
        cli.autovb_nbo(["--help"])
    assert exc.value.code == 0

def test_autovb_xmi_help_exits_0():
    with pytest.raises(SystemExit) as exc:
        cli.autovb_xmi(["--help"])
    assert exc.value.code == 0

def test_draw_xmo_help_exits_0():
    with pytest.raises(SystemExit) as exc:
        cli.draw_xmo(["--help"])
    assert exc.value.code == 0

def test_draw_xmo_structures_per_row_parser():
    assert cli.parse_draw_xmo_structures_per_row("3") == 3

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples" / "C4H6"

# 只在 examples/C4H6 下运行的特定测试用例：
# (name, function_name, argv_list, required_files, expected_returncode)
CASES = [
    ("nbo_C4H6", "autovb_nbo", ["C4H6.xyz", "cc-pvdz"], ["C4H6.xyz"], 0),
    ("xmi_C4H6_t", "autovb_xmi", ["C4H6", "cc-pvdz", "-t", "1.95"], ["C4H6.fch"], 0),
    ("xmi_C4H6_nae_nao", "autovb_xmi", ["C4H6", "cc-pvdz", "-nae", "4", "-nao", "4"], ["C4H6.fch"], 0),
    ("xmi_C4H6_aoa", "autovb_xmi", ["C4H6", "cc-pvdz", "-aoa", "1", "4", "6", "8"], ["C4H6.fch"], 0),
    ("xmi_C4H6_aoa_nae_nao", "autovb_xmi", ["C4H6", "cc-pvdz", "-aoa", "1", "4", "6", "8", "-nae", "4", "-nao", "4"], ["C4H6.fch"], 0),
]

@pytest.mark.parametrize("case_name, func, argv, required_files, expected", CASES)
def test_examples_cli(case_name, func, argv, required_files, expected, tmp_path):
    if not EXAMPLES.exists():
        pytest.skip(f"examples/C4H6 目录不存在: {EXAMPLES}")

    # 若示例所需文件不存在则跳过该项
    for f in required_files:
        if not (EXAMPLES / f).exists():
            pytest.skip(f"示例文件缺失，跳过: {(EXAMPLES / f)}")

    # 在 examples/C4H6 目录下运行，每个测试用独立 Python 进程调用包内函数
    pycmd = (
        "import sys; "
        "from autoVB import cli; "
        f"sys.exit(cli.{func}({argv!r}))"
    )
    proc = subprocess.run([sys.executable, "-c", pycmd],
                          cwd=str(EXAMPLES),
                          capture_output=True,
                          text=True,
                          timeout=120)
    if proc.stdout:
        print("=== SUBPROCESS STDOUT ===")
        print(proc.stdout)
    if proc.stderr:
        print("=== SUBPROCESS STDERR ===")
        print(proc.stderr)

    assert proc.returncode == expected, (
        f"{case_name} failed: rc={proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
