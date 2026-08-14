from autoVB.cli.xmo2xmi import xmo2xmi, xmo2xmi_file
from autoVB.io.readers import read_xmi_and_orb


XMI_TEXT = """test title
$ctrl
vbscf
str=full
basis=6-31g
$end

$orb
1*2
1
2
$end

$geo
H 0.0 0.0 0.0
H 0.0 0.0 0.7
$end

$gus
old orbital guess
$end

$str
1:2
$end
"""

ORB_TEXT = """2 2
# ORBITAL 1
 1.0000000000 1
# ORBITAL 2
 1.0000000000 2
"""


def test_read_xmi_and_matching_orb(tmp_path):
    xmi_path = tmp_path / "sample.xmi"
    orb_path = tmp_path / "sample.orb"
    xmi_path.write_text(XMI_TEXT, encoding="utf-8")
    orb_path.write_text(ORB_TEXT, encoding="utf-8")

    xmi_text, orb_text = read_xmi_and_orb(xmi_path)

    assert xmi_text == XMI_TEXT
    assert orb_text == ORB_TEXT


def test_xmo2xmi_replaces_only_guess_and_method(tmp_path):
    xmi_path = tmp_path / "sample.xmi"
    orb_path = tmp_path / "sample.orb"
    output_path = tmp_path / "converted.xmi"
    xmi_path.write_text(XMI_TEXT, encoding="utf-8")
    orb_path.write_text(ORB_TEXT, encoding="utf-8")

    result = xmo2xmi_file(xmi_path, output_path, method="vbpt2")

    expected = XMI_TEXT.replace("vbscf", "vbpt2", 1).replace(
        "old orbital guess\n",
        ORB_TEXT,
        1,
    )
    assert result == output_path
    assert output_path.read_text(encoding="utf-8") == expected


def test_xmo2xmi_cli_uses_new_xmi_default_name(tmp_path):
    xmi_path = tmp_path / "sample.xmi"
    orb_path = tmp_path / "sample.orb"
    xmi_path.write_text(XMI_TEXT, encoding="utf-8")
    orb_path.write_text(ORB_TEXT, encoding="utf-8")

    assert xmo2xmi([str(xmi_path)]) == 0

    output_path = tmp_path / "sample_new.xmi"
    assert output_path.exists()
    assert "vbscf" in output_path.read_text(encoding="utf-8")
    assert ORB_TEXT in output_path.read_text(encoding="utf-8")


def test_xmo2xmi_preserves_crlf_outside_replaced_content(tmp_path):
    xmi_path = tmp_path / "sample.xmi"
    orb_path = tmp_path / "sample.orb"
    output_path = tmp_path / "sample_new.xmi"
    xmi_text = XMI_TEXT.replace("\n", "\r\n")
    orb_text = ORB_TEXT.replace("\n", "\r\n")
    xmi_path.write_bytes(xmi_text.encode("utf-8"))
    orb_path.write_bytes(orb_text.encode("utf-8"))

    xmo2xmi_file(xmi_path, output_path, method="bovb")

    expected = xmi_text.replace("vbscf", "bovb", 1).replace(
        "old orbital guess\r\n",
        orb_text,
        1,
    )
    assert output_path.read_bytes().decode("utf-8") == expected
