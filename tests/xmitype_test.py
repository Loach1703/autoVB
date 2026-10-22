from types import SimpleNamespace

import pytest

from autoVB.gvb.gvb import XMVBGVB
from autoVB.io.readers import autoVBInputParser
from autoVB.io.writers import XMIData, write_xmi_file
from autoVB.main import VBSettings, XMIPassthrough, autoVBMain
from autoVB.nbo.nbo import XMVBNBO


def parse_options(options):
    parser = autoVBInputParser.__new__(autoVBInputParser)
    return parser.parse_autovb_options(f"autovb{{{options}}}")


def xmi_data(**options):
    return XMIData(
        molecule_name="H2", method="vbscf", stru_type="full", int_type="libcint",
        iscf=None, nae=2, nao=2, ncharge=0, nmul=1, basis_set="sto-3g",
        sort=False, orb_section="1*2\n1\n2", geo_section="H 0 0 0\nH 0 0 0.74",
        init_guess_section="1 1\n1.0 1\n1.0 2", **options,
    )


@pytest.mark.parametrize("xmitype,expected", [("4", 5), ("2026", 5), ("open", 8)])
def test_output_defaults_and_legacy_format(tmp_path, xmitype, expected):
    settings = parse_options(f"xmitype={xmitype}")
    assert settings.xmitype == xmitype
    assert settings.iscf == expected
    assert VBSettings(xmitype=xmitype).iscf == expected
    path = tmp_path / "h2.xmi"
    write_xmi_file(path, xmi_data(xmitype=xmitype))
    text = path.read_text()
    assert f"iscf={expected}\n" in text
    assert ("eigensolver=davidson\n" in text) == (xmitype == "open")
    assert "xmitype=" not in text  # This is an autoVB setting, not an XMVB keyword.
    if xmitype != "open":
        legacy = tmp_path / "legacy.xmi"
        write_xmi_file(legacy, xmi_data())
        assert text.splitlines()[1:] == legacy.read_text().splitlines()[1:]


@pytest.mark.parametrize("iscf", [7, 8, 9])
@pytest.mark.parametrize("options", ["xmitype=open,iscf={}", "iscf={},xmitype=open"])
def test_open_explicit_iscf_is_independent_of_parameter_order(iscf, options):
    assert parse_options(options.format(iscf)).iscf == iscf


@pytest.mark.parametrize("value", ["2", "5", "6", "0", "10", "8.5", "bad", "true"])
def test_open_invalid_iscf_raises_instead_of_using_default(value):
    with pytest.raises(ValueError, match="iscf"):
        parse_options(f"xmitype=open,iscf={value}")


@pytest.mark.parametrize("value", ["unknown", "2025", "true"])
def test_unknown_output_type_raises(value):
    with pytest.raises(ValueError, match="xmitype"):
        parse_options(f"xmitype={value}")
    with pytest.raises(ValueError, match="xmitype"):
        VBSettings(xmitype=4).validate()


@pytest.mark.parametrize("iscf", [None, 7, 8, 9, 5, "bad"])
def test_native_xmi_input_obeys_open_constraints(tmp_path, iscf):
    path = tmp_path / "input.xmi"
    setting = "" if iscf is None else f"iscf={iscf}\n"
    path.write_text("H2\n$ctrl\nvbscf\nbasis=sto-3g\nxmitype=open\n" + setting
                    + "$end\n$geo\nH 0 0 0\nH 0 0 0.74\n$end\n")
    if iscf in (5, "bad"):
        with pytest.raises(ValueError, match="iscf"):
            autoVBInputParser(path)
    else:
        data = autoVBInputParser(path).input_data
        assert data.vbsettings.iscf == (8 if iscf is None else iscf)
        assert data.vbsettings.xmitype == "open"


def test_writer_enforces_solver_and_rejects_invalid_iscf_before_writing(tmp_path):
    path = tmp_path / "open.xmi"
    data = xmi_data(xmitype="open")
    passthrough = XMIPassthrough(ctrl_extra_lines=[
        "EIGENSOLVER = lapack molden", "eigensolver=other", "iscf=8", "itmax=250",
    ])
    write_xmi_file(path, data, passthrough)
    text = path.read_text()
    assert text.lower().count("eigensolver=") == 1
    assert "eigensolver=davidson" in text
    assert text.count("iscf=") == 1
    assert "iscf=8" in text and "itmax=250" in text and "molden" in text
    original = path.read_bytes()
    for value in ("2", "8.5", "9"):
        with pytest.raises(ValueError, match="iscf"):
            write_xmi_file(path, data, XMIPassthrough(ctrl_extra_lines=[f"iscf={value}"]))
        assert path.read_bytes() == original
    data.iscf = 5
    with pytest.raises(ValueError, match="iscf"):
        write_xmi_file(path, data)
    assert path.read_bytes() == original


def orbital_model(kind, settings, method):
    model = kind.__new__(kind)
    model.input_data = SimpleNamespace(vbsettings=settings, method=method, charge=0, spin=1)
    model.active_electron = model.active_orbital = 2
    model.active_orb_atom = [[1], [2]]
    model.origin_filename = model.filename = "H2"
    model.basis_set = "sto-3g"
    model.geometry_text = "H 0 0 0\nH 0 0 0.74"
    model.get_active_orb_atom_order = lambda atoms: [1, 2]
    model.get_orb_section_total = lambda order: "1*2\n1\n2"
    model.get_init_guess_total = lambda order: "1 1\n1.0 1\n1.0 2"
    return model


@pytest.mark.parametrize("kind", [XMVBNBO, XMVBGVB])
@pytest.mark.parametrize("method", ["vbscf", "bovb"])
@pytest.mark.parametrize("xmitype,expected", [("4", 5), ("2026", 5), ("open", 8)])
def test_nbo_and_gvb_propagate_output_type(kind, method, xmitype, expected):
    data = orbital_model(kind, VBSettings(xmitype=xmitype), method).get_xmidata()
    assert data.xmitype == xmitype
    assert data.iscf == (2 if method == "bovb" and xmitype != "open" else expected)


@pytest.mark.parametrize("iscf", [7, 8, 9])
def test_stream_preparation_does_not_override_open_iscf(tmp_path, monkeypatch, iscf):
    monkeypatch.chdir(tmp_path)
    settings = VBSettings(xmitype="open", iscf=iscf, bovb_stream=True)
    model = orbital_model(XMVBNBO, settings, "bovb")
    model.get_aoi = lambda auto_set: (2, 2, [1, 2])
    model.split_inactive_active_orbitals = lambda indices: None
    workflow = autoVBMain.__new__(autoVBMain)
    workflow.input_data = SimpleNamespace(
        method="bovb", vbsettings=settings, xmi_passthrough=XMIPassthrough(),
    )
    workflow.wxp = model
    workflow.xmi_name = "h2_vb"
    workflow.generate_nbo_to_xmi()
    text = (tmp_path / "h2_vb.xmi").read_text()
    assert f"iscf={iscf}\n" in text and "eigensolver=davidson\n" in text


def test_blw_preparation_does_not_force_iscf5_in_open_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    workflow = autoVBMain.__new__(autoVBMain)
    workflow.wxp = SimpleNamespace(get_xmidata=lambda method_override: xmi_data(xmitype="open"))
    workflow.input_data = SimpleNamespace(xmi_passthrough=XMIPassthrough())
    workflow.blw_name = "h2_blw"
    workflow.generate_blw_to_xmi()
    assert "iscf=8\n" in (tmp_path / "h2_blw.xmi").read_text()


@pytest.mark.parametrize("xmitype,iscf,final_iscf", [("4", 5, 2), ("open", 7, 7), ("open", 8, 8), ("open", 9, 9)])
def test_bovb_stream_final_conversion_retains_open_settings(tmp_path, monkeypatch, xmitype, iscf, final_iscf):
    monkeypatch.chdir(tmp_path)
    initial = xmi_data(xmitype=xmitype)
    initial.iscf = iscf
    write_xmi_file(tmp_path / "h2_vb.xmi", initial)
    workflow = autoVBMain.__new__(autoVBMain)
    workflow.input_data = SimpleNamespace(vbsettings=VBSettings(
        xmitype=xmitype, iscf=iscf, nojson=True, bovb_stream=True,
    ))
    workflow.xmi_name = "h2_vb"
    workflow.bovb_name = "h2_bovb"
    workflow.run_xmvb = lambda name: (tmp_path / f"{name}.orb").write_text("1 1\n0.9 1\n0.9 2\n")
    workflow.parser_xmo = lambda *args: None
    workflow.run_bovb_stream()
    text = (tmp_path / "h2_bovb.xmi").read_text()
    assert "\nbovb\n" in text and f"iscf={final_iscf}\n" in text
    assert ("eigensolver=davidson\n" in text) == (xmitype == "open")
