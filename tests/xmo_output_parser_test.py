from autoVB.draw_xmo import XmoParser


def write_xmo(tmp_path, text: str):
    xmo_path = tmp_path / "sample.xmo"
    xmo_path.write_text(text.strip() + "\n", encoding="utf-8")
    return xmo_path


def minimal_xmo(weight_tables: str) -> str:
    return f"""
$ctrl
vbscf
nae=2
nao=2
basis=cc-pVDZ
iscf=5
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

{weight_tables}
"""


def test_xmo_parser_reads_metadata_and_all_weight_tables(tmp_path):
    xmo_path = write_xmo(
        tmp_path,
        minimal_xmo(
            """
******  WEIGHTS OF STRUCTURES ******
1 0.50 ****** 1-2

Lowdin Weights
1 0.40 ****** 1-2

Inverse Weights
1 0.30 ****** 1-2

Renormalized Weights
1 0.20 ****** 1-2
"""
        ),
    )

    parsed = XmoParser(xmo_path).parse()

    assert parsed.method == "vbscf"
    assert parsed.basis == "cc-pVDZ"
    assert parsed.ctrl_options["iscf"] == "5"
    assert parsed.cc_weights[0].weight == 0.50
    assert parsed.cc_weights[0].orbital_connections == [(1, 2)]
    assert parsed.cc_weights[0].atom_connections == [(1, 2)]
    assert parsed.cc_weights[0].flat_orbitals == [1, 2]
    assert parsed.cc_weights[0].flat_atoms == [1, 2]
    assert parsed.lowdin_weights[0].weight == 0.40
    assert parsed.inverse_weights[0].weight == 0.30
    assert parsed.renormalized_weights[0].weight == 0.20
    assert parsed.to_dict()["basis"] == "cc-pVDZ"
    assert parsed.to_dict()["cc_weights"][0]["atom_connections"] == [
        {"begin": 1, "end": 2}
    ]
    assert parsed.to_dict()["cc_weights"][0]["flat_orbitals"] == [1, 2]


def test_xmo_parser_maps_structure_orbitals_to_orb_atoms(tmp_path):
    xmo_path = write_xmo(
        tmp_path,
        """
$ctrl
vbscf
nae=3
nao=3
basis=cc-pVDZ
$end

$orb
1*5
8
9
1
3
5
$end

$geo
C 0.0 0.0 0.0
C 0.0 0.0 1.0
C 0.0 1.0 0.0
C 1.0 0.0 0.0
C 1.0 1.0 0.0
$end

******  WEIGHTS OF STRUCTURES ******
1 0.50 ****** 1:2 3-4 4-5
""",
    )

    parsed = XmoParser(xmo_path).parse()
    weight = parsed.cc_weights[0]

    assert parsed.orbital_to_atom == {3: 1, 4: 3, 5: 5}
    assert weight.inactive_orbital_ranges == [(1, 2)]
    assert weight.orbital_connections == [(3, 4), (4, 5)]
    assert weight.atom_connections == [(1, 3), (3, 5)]
    assert weight.flat_orbitals == [3, 4, 4, 5]
    assert weight.flat_atoms == [1, 3, 3, 5]


def test_xmo_parser_reads_space_separated_self_pair_structure(tmp_path):
    orb_rows = "\n".join(["9"] * 30 + [str(i) for i in range(1, 11)])
    geo_rows = "\n".join(f"C {i}.0 0.0 0.0" for i in range(10))
    xmo_path = write_xmo(
        tmp_path,
        f"""
$ctrl
vbscf
nae=10
nao=10
basis=cc-pVDZ
$end

$orb
1*40
{orb_rows}
$end

$geo
{geo_rows}
$end

******  WEIGHTS OF STRUCTURES ******
1 0.50 ****** 1:30 32 32 33-34 36-37 35-38 39-40
""",
    )

    parsed = XmoParser(xmo_path).parse()
    weight = parsed.cc_weights[0]

    assert parsed.orbital_to_atom == {
        31: 1,
        32: 2,
        33: 3,
        34: 4,
        35: 5,
        36: 6,
        37: 7,
        38: 8,
        39: 9,
        40: 10,
    }
    assert weight.inactive_orbital_ranges == [(1, 30)]
    assert weight.orbital_connections == [
        (32, 32),
        (33, 34),
        (36, 37),
        (35, 38),
        (39, 40),
    ]
    assert weight.atom_connections == [
        (2, 2),
        (3, 4),
        (6, 7),
        (5, 8),
        (9, 10),
    ]
    assert weight.flat_orbitals == [32, 32, 33, 34, 36, 37, 35, 38, 39, 40]
    assert weight.flat_atoms == [2, 2, 3, 4, 6, 7, 5, 8, 9, 10]


def test_xmo_parser_reads_space_separated_ctrl_options(tmp_path):
    xmo_path = write_xmo(
        tmp_path,
        """
$ctrl
vbscf nae=2 nao = 2
basis=cc-pVDZ iprint = 3
molden output=aim
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

******  WEIGHTS OF STRUCTURES ******
1 0.50 ****** 1-2
""",
    )

    parsed = XmoParser(xmo_path).parse()

    assert parsed.method == "vbscf"
    assert parsed.nae == 2
    assert parsed.nao == 2
    assert parsed.ctrl_options["basis"] == "cc-pVDZ"
    assert parsed.ctrl_options["iprint"] == "3"
    assert parsed.ctrl_options["molden"] is True
    assert parsed.ctrl_options["output"] == "aim"


def test_xmo_parser_warns_when_optional_weight_tables_are_missing(tmp_path, capsys):
    xmo_path = write_xmo(
        tmp_path,
        minimal_xmo(
            """
******  WEIGHTS OF STRUCTURES ******
1 0.50 ****** 1-2
"""
        ),
    )

    parsed = XmoParser(xmo_path).parse()
    captured = capsys.readouterr()

    assert parsed.cc_weights
    assert parsed.lowdin_weights == []
    assert parsed.inverse_weights == []
    assert parsed.renormalized_weights == []
    assert "Lowdin Weights" in captured.out
    assert "Inverse Weights" in captured.out
    assert "Renormalized Weights" in captured.out
