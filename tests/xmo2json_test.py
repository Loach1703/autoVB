import json

from autoVB.cli.xmo2json import xmo2json


def test_xmo2json_uses_xmo_stem_for_default_output(tmp_path):
    xmo_path = tmp_path / "sample.xmo"
    output_path = tmp_path / "sample.json"
    xmo_path.write_text(
        """
$ctrl
vbscf
nae=2
nao=2
ncharge=0
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
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert xmo2json([str(xmo_path)]) == 0

    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["molecule"]["name"] == "sample"
    assert summary["calculation"]["energy"] == -1.1
    assert summary["structures"][0]["weights"] == {"cc": 0.5}
