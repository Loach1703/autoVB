import json
from types import SimpleNamespace

import pytest

from autoVB.io.writers import write_automr, write_json_summary
from autoVB.io.xmo_output_parser import XmoParser


@pytest.mark.parametrize("mem,nproc,enough", [
    ("512MB", 1, False), ("2047MB", 1, False), ("2048MB", 1, True),
    ("4095MB", 2, False), ("4096MB", 2, True), ("2GB", 1, True),
])
def test_automr_memory_requirement_respects_units(tmp_path, monkeypatch, mem, nproc, enough):
    from autoVB.utils import utils

    monkeypatch.setattr(utils, "pyscf_to_xyz", lambda mol: "H 0 0 0\nH 0 0 0.74")
    mol = SimpleNamespace(charge=0, spin=0, basis="sto-3g")
    filename = str(tmp_path / "h2")
    if not enough:
        with pytest.raises(ValueError, match="requires at least"):
            write_automr(mol, filename, mem=mem, nproc=nproc)
        assert not (tmp_path / "h2.gjf").exists()
    else:
        write_automr(mol, filename, mem=mem, nproc=nproc)
        assert f"%mem={mem}\n" in (tmp_path / "h2.gjf").read_text()


def test_write_json_summary_merges_structure_weights(tmp_path, monkeypatch):
    xmo_path = tmp_path / "sample.xmo"
    xmo_path.write_text(
        """
$ctrl
vbscf
nae=2
nao=2
ncharge=1
nmul=1
basis=cc-pVDZ
$end

$orb
1*2
1
2
$end

$geo
H 0.0 0.0 0.0
H 0.0 0.0 1.0
$end

VBSCF converged in 3 iterations
Total Energy: -1.10000000

****** WEIGHTS OF STRUCTURES ******
1 0.50 ****** 1-2

Lowdin Weights
1 0.40 ****** 1-2

Inverse Weights
1 0.30 ****** 1-2

Renormalized Weights
1 0.20 ****** 1-2
""".strip()
        + "\n",
        encoding="utf-8",
    )
    parsed_data = XmoParser(xmo_path).parse()
    input_data = SimpleNamespace(
        filename="H2",
        title="H2 test",
        charge=0,
        spin=3,
    )
    monkeypatch.chdir(tmp_path)

    output_path = write_json_summary(input_data, parsed_data)
    summary = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.resolve() == tmp_path / "H2.json"
    assert summary["molecule"] == {
        "name": "H2",
        "title": "H2 test",
        "charge": 1,
        "multiplicity": 1,
        "geometry": [
            {"symbol": "H", "x": 0.0, "y": 0.0, "z": 0.0},
            {"symbol": "H", "x": 0.0, "y": 0.0, "z": 1.0},
        ],
    }
    assert summary["orb"] == {
        "nae": 2,
        "nao": 2,
        "section": [[1], [2]],
        "orbital_to_atom": {"1": 1, "2": 2},
    }
    assert summary["calculation"] == {
        "method": "vbscf",
        "basis": "cc-pVDZ",
        "converged": True,
        "steps": 3,
        "energy": -1.1,
        "energy_terms": {"total_energy": -1.1},
    }
    assert summary["structures"] == [
        {
            "index": 1,
            "structure": "1-2",
            "inactive_orbital_ranges": [],
            "orbital_connections": [{"begin": 1, "end": 2}],
            "atom_connections": [{"begin": 1, "end": 2}],
            "unpaired_orbitals": [],
            "unpaired_atoms": [],
            "flat_orbitals": [1, 2],
            "flat_atoms": [1, 2],
            "weights": {
                "cc": 0.5,
                "lowdin": 0.4,
                "inverse": 0.3,
                "renormalized": 0.2,
            },
        }
    ]
