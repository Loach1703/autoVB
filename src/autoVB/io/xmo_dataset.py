"""Extract final VB orbitals without loading large structure tables into memory."""

from pathlib import Path
import re

import numpy as np

from .xmo_output_parser import XmoParser


class XmoDatasetReader(XmoParser):
    """Reuse input-section parsing and stream the final AO coefficient table."""

    def read(self) -> tuple[dict, dict[str, np.ndarray]]:
        header = []
        in_input = False
        in_table = False
        columns = []
        entries = {}
        ao_labels = {}
        coordinates = []
        in_coordinates = False
        basis_lines = []
        in_basis = False
        n_basis = None
        n_electrons = None
        converged = False
        finished = False
        energy = None
        row_pattern = re.compile(r"^\s*(\d+)\s+([A-Za-z]+)\s+(\d+)\s+(\S+)\s+(.+)$")

        with self.xmo_file.open() as stream:
            for line in stream:
                stripped = line.strip()
                if stripped == "---------------Input File---------------":
                    in_input = True
                    continue
                if stripped == "---------------End of Input--------------":
                    in_input = False
                if in_input:
                    header.append(line)
                    continue

                if "COORDINATES (BOHR)" in line:
                    in_coordinates = True
                    coordinates = []
                    continue
                if in_coordinates:
                    parts = stripped.split()
                    if len(parts) == 5:
                        coordinates.append([float(x) for x in parts[1:]])
                    elif coordinates:
                        in_coordinates = False

                if stripped == "ATOMIC BASIS SET":
                    in_basis = True
                if "TOTAL NUMBER OF BASIS SET SHELLS" in line:
                    in_basis = False
                if in_basis:
                    basis_lines.append(line)
                if "NUMBER OF CARTESIAN GAUSSIAN BASIS FUNCTIONS" in line:
                    n_basis = int(line.split("=")[-1])
                if "NUMBER OF ELECTRONS" in line and "=" in line:
                    n_electrons = int(line.split("=")[-1])
                if "VBSCF converged in" in line:
                    converged = True
                if stripped.startswith("Total Energy:"):
                    energy = float(stripped.split(":", 1)[1])
                if "Job finished at" in line:
                    finished = True

                if "ORBITALS IN PRIMITIVE BASIS FUNCTIONS" in line:
                    in_table = True
                    columns = []
                    entries = {}
                    ao_labels = {}
                    continue
                if not in_table:
                    continue
                if "*****" in line:
                    in_table = False
                    continue
                if not stripped:
                    continue
                parts = stripped.split()
                if all(part.isdigit() for part in parts):
                    columns = [int(part) - 1 for part in parts]
                    continue
                match = row_pattern.match(line)
                if not match:
                    raise ValueError(f"Unrecognized final orbital row: {stripped}")
                ao, symbol, atom, component, values = match.groups()
                ao_index = int(ao) - 1
                label = {"symbol": symbol, "atom_index": int(atom) - 1, "component": component}
                if ao_index in ao_labels and ao_labels[ao_index] != label:
                    raise ValueError("AO labels change between orbital column blocks")
                ao_labels[ao_index] = label
                numbers = [float(value.replace("D", "E")) for value in values.split()]
                if len(numbers) != len(columns) or not columns:
                    raise ValueError("Orbital coefficient count does not match column header")
                for orbital, coefficient in zip(columns, numbers):
                    key = (orbital, ao_index)
                    if key in entries:
                        raise ValueError(f"Duplicate coefficient at {key}")
                    entries[key] = coefficient

        ctrl = self._section_lines(header, "$ctrl")
        options = self._parse_ctrl_options(ctrl)
        if self._parse_method(ctrl).lower() != "vbscf":
            raise ValueError("Dataset export currently supports VBSCF only")
        if not converged or not finished:
            raise ValueError("Dataset export requires converged and normally finished VBSCF")
        orb = self._parse_orb(self._section_lines(header, "$orb"))
        nae = self._parse_required_int(ctrl, "nae")
        nao = self._parse_required_int(ctrl, "nao")
        if n_basis is None or n_electrons is None or not coordinates:
            raise ValueError("Missing Cartesian basis dimension, electron count or output coordinates")
        if not 0 <= nao <= len(orb) or 2 * (len(orb) - nao) + nae != n_electrons:
            raise ValueError("Orbital roles and electron count are inconsistent")
        expected = {(i, j) for i in range(len(orb)) for j in range(n_basis)}
        if entries.keys() != expected:
            raise ValueError("Final orbital table is missing, incomplete or has unexpected dimensions")
        matrix = np.array([[entries[i, j] for j in range(n_basis)] for i in range(len(orb))])
        labels = [ao_labels[j] for j in range(n_basis)]
        atom_indices = np.array([label["atom_index"] for label in labels], dtype=np.int64)
        coordinates = np.asarray(coordinates)
        if any(atom < 1 or atom > len(coordinates) for row in orb for atom in row):
            raise ValueError("Orbital atom label outside geometry")
        if np.any(atom_indices < 0) or np.any(atom_indices >= len(coordinates)):
            raise ValueError("AO atom label outside geometry")
        if not np.isfinite(matrix).all() or not np.isfinite(coordinates).all():
            raise ValueError("Nonfinite coefficients or coordinates")
        active_mask = np.arange(len(orb)) >= len(orb) - nao
        metadata = {
            "schema_version": 1,
            "source_file": str(self.xmo_file.resolve()),
            "sample_id": self.xmo_file.stem,
            "method": "vbscf",
            "ctrl_options": options,
            "input_sections": {
                name: "".join(self._section_lines(header, name))
                for name in ("$ctrl", "$orb", "$geo", "$str", "$frag", "$basis")
                if any(line.strip().lower() == name for line in header)
            },
            "basis": options.get("basis"),
            "charge": self._parse_optional_int(ctrl, "ncharge", 0),
            "multiplicity": self._parse_optional_int(ctrl, "nmul", 1),
            "nae": nae, "nao": nao, "n_electrons": n_electrons,
            "n_orbitals": len(orb), "n_basis": n_basis,
            "orbital_atoms": orb,
            "orbital_atoms_index_base": 1,
            "array_atom_index_base": 0,
            "orbital_roles": ["active" if active else "inactive" for active in active_mask],
            "ao_labels": labels,
            "positions_unit": "bohr",
            "coefficient_layout": "orbital_by_ao",
            "coefficient_source": "ORBITALS IN PRIMITIVE BASIS FUNCTIONS",
            "coefficient_convention": "raw XMVB printed values; no reordering or renormalization",
            "basis_output_text": "".join(basis_lines),
            "converged": converged, "finished": finished, "energy_hartree": energy,
            "ao_overlap_available": False,
            "limitations": ["Printed coefficient precision is retained; higher precision is not recovered.",
                            "orbital_atom_mask encodes input labels, not verified strict localization."],
        }
        arrays = {
            "positions": coordinates[:, 1:],
            "atomic_numbers": coordinates[:, 0].astype(np.int64),
            "coefficients": matrix,
            "active_mask": active_mask,
            "ao_atom_indices": atom_indices,
            "orbital_atom_mask": np.array([np.isin(atom_indices + 1, row) for row in orb]),
        }
        return metadata, arrays
