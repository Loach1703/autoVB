import pytest
import sys
import subprocess
import logging
from pathlib import Path
from autoVB.cli import autovb, draw_xmo, nbo, xmi

def test_autovb_nbo_help_exits_0():
    with pytest.raises(SystemExit) as exc:
        nbo.autovb_nbo(["--help"])
    assert exc.value.code == 0

def test_autovb_xmi_help_exits_0():
    with pytest.raises(SystemExit) as exc:
        xmi.autovb_xmi(["--help"])
    assert exc.value.code == 0

def test_draw_xmo_help_exits_0():
    with pytest.raises(SystemExit) as exc:
        draw_xmo.draw_xmo(["--help"])
    assert exc.value.code == 0

def test_draw_xmo_structures_per_row_parser():
    assert draw_xmo.parse_draw_xmo_structures_per_row("3") == 3

def test_autovb_main_logs_usage_for_missing_input(caplog):
    caplog.set_level(logging.INFO)

    assert autovb.autovb_main([]) == 2

    assert "Welcome to autoVB!" in caplog.text
    assert "Usage: autovb <input-file>" in caplog.text

def test_autovb_main_parses_mem_nproc_and_debug(tmp_path, monkeypatch, caplog):
    input_file = tmp_path / "job.autovb"
    input_file.write_text("", encoding="utf-8")
    captured = {}

    class FakeInputParser:
        def __init__(self, path):
            captured["path"] = path
            self.input_data = type("InputData", (), {"mem": "2GB", "nproc": "2"})()

    class FakeMain:
        def __init__(self, input_data):
            captured["input_data"] = input_data

        def main(self):
            captured["ran"] = True

    monkeypatch.setattr(autovb, "autoVBInputParser", FakeInputParser)
    monkeypatch.setattr(autovb, "autoVBMain", FakeMain)
    caplog.set_level(logging.DEBUG)

    assert autovb.autovb_main([str(input_file), "--mem", "8G", "--nproc", "4", "--debug"]) is None

    assert captured["path"] == input_file
    assert captured["input_data"].mem == "8GB"
    assert captured["input_data"].nproc == "4"
    assert captured["ran"] is True
    assert "Using memory: 8GB, nproc: 4" in caplog.text

def test_normalize_memory():
    assert autovb.normalize_memory("8G") == "8GB"
    assert autovb.normalize_memory("512M") == "512MB"
    assert autovb.normalize_memory("4000") == "4000MB"

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples" / "C4H6"

# 只在 examples/C4H6 下运行的特定测试用例：
# (name, module_name, function_name, argv_list, required_files, expected_returncode)
CASES = [
    ("nbo_C4H6", "nbo", "autovb_nbo", ["C4H6.xyz", "cc-pvdz"], ["C4H6.xyz"], 0),
    ("xmi_C4H6_t", "xmi", "autovb_xmi", ["C4H6", "cc-pvdz", "-t", "1.95"], ["C4H6.fch"], 0),
    ("xmi_C4H6_nae_nao", "xmi", "autovb_xmi", ["C4H6", "cc-pvdz", "-nae", "4", "-nao", "4"], ["C4H6.fch"], 0),
    ("xmi_C4H6_aoa", "xmi", "autovb_xmi", ["C4H6", "cc-pvdz", "-aoa", "1", "4", "6", "8"], ["C4H6.fch"], 0),
    ("xmi_C4H6_aoa_nae_nao", "xmi", "autovb_xmi", ["C4H6", "cc-pvdz", "-aoa", "1", "4", "6", "8", "-nae", "4", "-nao", "4"], ["C4H6.fch"], 0),
]

@pytest.mark.parametrize("case_name, module_name, func, argv, required_files, expected", CASES)
def test_examples_cli(case_name, module_name, func, argv, required_files, expected, tmp_path):
    if not EXAMPLES.exists():
        pytest.skip(f"examples/C4H6 目录不存在: {EXAMPLES}")

    # 若示例所需文件不存在则跳过该项
    for f in required_files:
        if not (EXAMPLES / f).exists():
            pytest.skip(f"示例文件缺失，跳过: {(EXAMPLES / f)}")

    # 在 examples/C4H6 目录下运行，每个测试用独立 Python 进程调用包内函数
    pycmd = (
        "import sys; "
        f"from autoVB.cli import {module_name} as command_module; "
        f"sys.exit(command_module.{func}({argv!r}))"
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
