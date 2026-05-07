from autoVB.constants import to_gaussian_basis_name


def test_to_gaussian_basis_name_maps_common_aliases():
    assert to_gaussian_basis_name("def2-svp") == "def2SVP"
    assert to_gaussian_basis_name("def2_tzvppd") == "def2TZVPPD"
    assert to_gaussian_basis_name("cc-pvdz") == "cc-pVDZ"


def test_to_gaussian_basis_name_keeps_unknown_names():
    assert to_gaussian_basis_name("custom-basis") == "custom-basis"
