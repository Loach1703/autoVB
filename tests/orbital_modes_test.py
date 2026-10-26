from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoVB.io.readers import autoVBInputParser
from autoVB.io.writers import XMIData, write_xmi_file
from autoVB.io.xmo_output_parser import XmoParser
from autoVB.main import VBSettings, XMIPassthrough, autoVBMain


def input_data(**options):
    # One inactive He orbital, followed by two active H orbitals.
    return XMIData(
        molecule_name="HHeH", method="vbscf", stru_type="full", int_type="libcint",
        iscf=5, nae=2, nao=2, ncharge=0, nmul=1, basis_set="sto-3g", sort=False,
        orb_section="1*3\n2\n1 # active\n3",
        geo_section="H 0 0 0\nHe 4 0 0\nH 0 0 0.74",
        init_guess_section="1 1 1\n1.0 2\n1.0 1\n1.0 3", **options,
    )


def section(text, name):
    return text.split(f"${name}\n", 1)[1].split("$end", 1)[0].strip()


@pytest.mark.parametrize("mode", ["hao", "mixed", "oeo", "oeo_direct"])
def test_orb_setting_is_parsed_and_normalized(mode):
    parser = autoVBInputParser.__new__(autoVBInputParser)
    assert parser.parse_autovb_options(f"autovb{{orb={mode.upper()}}}").orb == mode


def test_unknown_orb_setting_is_rejected():
    with pytest.raises(ValueError, match="orb"):
        VBSettings(orb="unknown").validate()


@pytest.mark.parametrize("mode,expected", [
    ("hao", "1*3\n2\n1 # active\n3"),
    ("mixed", "3 1 1\n1 2 3\n1 # active\n3"),
    ("oeo_direct", "3 3 3\n1 2 3\n1 2 3\n1 2 3"),
])
def test_orbital_domains_preserve_numbering_and_active_space(tmp_path, mode, expected):
    path = tmp_path / "direct.xmi"
    write_xmi_file(path, input_data(orb=mode), XMIPassthrough(
        ctrl_extra_lines=["orbtyp=hao frgtyp=atom itmax=30 custom=value"],
        str_section_text="1:1 2:3",
    ))
    text = path.read_text()
    assert section(text, "orb") == expected
    assert f"orbtyp={'oeo' if mode == 'oeo_direct' else 'hao'}" in text
    assert "nao=2\nnae=2" in text
    assert section(text, "str") == "1:1 2:3"
    assert "itmax=30 custom=value" in text
    assert section(text, "gus") == input_data().init_guess_section


def workflow(tmp_path, monkeypatch, mode, *, method="vbscf", guess="nbo", novb=False):
    monkeypatch.chdir(tmp_path)
    Path("existing.out").touch()
    settings = VBSettings(orb=mode, guess=guess, novb=novb, bovb_stream=True,
                          nbo_file=Path("existing.out"), nojson=True)
    settings.validate()
    job = autoVBMain.__new__(autoVBMain)
    job.input_data = SimpleNamespace(method=method, vbsettings=settings,
                                    xmi_passthrough=XMIPassthrough(str_section_text="1:1 2:3"))
    job.xmi_name, job.blw_name, job.bovb_name = "test_vb", "test_blw", "test_bovb"
    job.xmvb_exe = "fake-xmvb"
    calls = []
    selections = []
    initial_guess = [input_data().init_guess_section]

    def select(auto_set):
        selections.append(auto_set)
        return 2, 2, [2, 3]

    model = SimpleNamespace(
        get_aoi=select,
        split_inactive_active_orbitals=lambda indices: None,
        set_initial_guess_orbitals=lambda matrix: initial_guess.__setitem__(0, matrix),
        get_xmidata=lambda method_override=None: replace(
            input_data(), method=method_override or method,
            init_guess_section=initial_guess[0], orb="hao" if method_override == "blw" else mode,
        ),
    )
    job.read_nbo = lambda: model
    job.read_blw_orbitals = lambda: Path("test_blw.orb").read_text()

    def run(name=None):
        name = name or job.xmi_name
        calls.append(name)
        text = Path(f"{name}.xmi").read_text()
        Path(f"{name}.orb").write_text(f"1 1 1\n0.{len(calls)} 2\n1.0 1\n1.0 3\n")
        structure = "2" if "nmul=2\n" in text else "2-3"
        Path(f"{name}.xmo").write_text(
            text + "\nVBSCF converged in 1 iterations\nTotal Energy: -3.0\n"
            f"******  WEIGHTS OF STRUCTURES ******\n1 1.0 ****** {structure}\n"
        )

    job.run_xmvb = run
    return job, calls, selections


@pytest.mark.parametrize("mode,expected", [
    ("hao", ["test_vb"]),
    ("mixed", ["test_vb"]),
    ("oeo_direct", ["test_vb"]),
    ("oeo", ["test_vb_hao", "test_vb"]),
])
def test_only_oeo_adds_a_hao_stage(tmp_path, monkeypatch, mode, expected):
    job, calls, selections = workflow(tmp_path, monkeypatch, mode)
    job.main()
    assert calls == expected
    assert selections == [True]
    if mode == "oeo":
        assert section(Path("test_vb.xmi").read_text(), "gus") == Path("test_vb_hao.orb").read_text().strip()
        assert job.parsed_data.orbital_to_atom == {}
        assert job.parsed_data.cc_weights[0].orbital_connections == [(2, 3)]
        assert job.parsed_data.cc_weights[0].atom_connections == []


def test_oeo_composes_with_blw_guess_and_bovb_stream(tmp_path, monkeypatch):
    job, calls, selections = workflow(tmp_path, monkeypatch, "oeo", method="bovb", guess="blw")
    job.main()
    assert calls == ["test_blw", "test_vb_hao", "test_vb", "test_bovb"]
    assert selections == [True]
    for previous, following in zip(calls, calls[1:]):
        text = Path(f"{following}.xmi").read_text()
        assert section(text, "gus") == Path(f"{previous}.orb").read_text().strip()
        assert "nao=2\nnae=2\nncharge=0\nnmul=1" in text
        assert "basis=sto-3g" in text
        assert section(text, "str") == "1:1 2:3"
    assert "\nvbscf\n" in Path("test_vb_hao.xmi").read_text()
    assert "orbtyp=hao" in Path("test_vb_hao.xmi").read_text()
    assert "orbtyp=oeo" in Path("test_bovb.xmi").read_text()
    assert "\nbovb\n" in Path("test_bovb.xmi").read_text()


def test_novb_prepares_only_hao_precursor(tmp_path, monkeypatch):
    job, calls, _ = workflow(tmp_path, monkeypatch, "oeo", novb=True)
    job.xmvb_exe = None
    job.main()
    assert calls == []
    assert Path("test_vb_hao.xmi").exists()
    assert not Path("test_vb.xmi").exists()


def test_failed_hao_stops_before_oeo_input_is_written(tmp_path, monkeypatch):
    job, calls, _ = workflow(tmp_path, monkeypatch, "oeo")
    job.parser_xmo = lambda *args: SimpleNamespace(converged=False)
    with pytest.raises(RuntimeError, match="HAO calculation did not converge"):
        job.main()
    assert calls == ["test_vb_hao"]
    assert not Path("test_vb.xmi").exists()


def test_oeo_stream_retains_open_solver_charge_and_multiplicity(tmp_path, monkeypatch):
    job, calls, _ = workflow(tmp_path, monkeypatch, "oeo")
    job.input_data.xmi_passthrough.ctrl_extra_lines = ["ORBTYP=OEO FRGTYP=ATOM custom=keep"]
    job.input_data.xmi_passthrough.str_section_text = "1:1 2"
    target = replace(input_data(xmitype="open"), iscf=9, ncharge=1, nmul=2, nae=1)
    hao = job._write_vb_input(Path("test_vb.xmi"), target)
    assert "orbtyp=hao" in hao.read_text()
    job.run_oeo_preparation()
    for path in (hao, Path("test_vb.xmi")):
        text = path.read_text()
        assert "nao=2\nnae=1\nncharge=1\nnmul=2\niscf=9" in text
        assert "eigensolver=davidson" in text
        assert "custom=keep" in text
    assert calls == ["test_vb_hao"]


@pytest.mark.parametrize("xmitype", ["4", "2026"])
def test_oeo_output_without_orb_block_keeps_weights_and_geometry(tmp_path, xmitype):
    path = tmp_path / "oeo.xmo"
    write_xmi_file(path.with_suffix(".xmi"), input_data(orb="oeo_direct", xmitype=xmitype))
    text = path.with_suffix(".xmi").read_text()
    start, end = text.index("$orb"), text.index("$end", text.index("$orb")) + 4
    path.write_text(text[:start] + text[end:] +
                    "\n******  WEIGHTS OF STRUCTURES ******\n1 1.0 ****** 2-3\n")
    parsed = XmoParser(path).parse()
    assert len(parsed.geo) == 3
    assert parsed.nao == parsed.nae == 2
    assert parsed.orbital_to_atom == {}
    assert parsed.cc_weights[0].weight == 1.0
    assert parsed.cc_weights[0].orbital_connections == [(2, 3)]
    assert "h" not in parsed.ctrl_options  # Embedded GEO is not parsed as control keywords.
