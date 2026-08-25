# autoVB

`autoVB` is a Python workflow tool for preparing and running XMVB valence-bond calculations.
It can select an active space, prepare initial guesses, generate XMVB input files, run calculations, and produce result summaries and SVG plots.

The full documentation is available in [UserGuide.md](UserGuide.md) and [UserGuide_en.md](UserGuide_en.md).

## Prerequisites

- Python >= 3.11
- Gaussian and `formchk` for the default NBO workflow
- XMVB (`xmvb`) for VB calculations
- MOKIT (`automr`) and GAMESS (`rungms`) when using `guess=gvb`

Example environment configuration:

```bash
export GAUSS_EXE=/share/apps/gaussian/g16/g16
export PATH=/share/apps/gaussian/16/formchk:$PATH
export PATH=/share/apps/xmvb/latest/bin:$PATH
export PATH=/share/apps/mokit/bin:$PATH
export GMS=/share/apps/gamess/rungms
```

Adjust these paths for your local installation. Add them to `~/.bashrc` or the corresponding shell startup file if they should persist between sessions.

## Installation

Using Conda for the scientific dependencies is recommended:

```bash
conda create -n autovb python=3.11 -y
conda activate autovb
conda install -c conda-forge numpy pyscf rdkit -y
conda install -c mokit -c conda-forge mokit -y
pip install pyssian
```

For a source checkout, install the package normally after preparing the dependencies above:

```bash
pip install --no-deps .
```

If a wheel has already been built, install it instead:

```bash
pip install --no-deps dist/autovb-*.whl
```

For developers working directly from a source checkout, editable installation is optional:

```bash
pip install --no-deps -e .
```

Check the installation with:

```bash
python -c "import autoVB, mokit, numpy, pyscf, rdkit, pyssian; print('Python dependencies are ready')"
autovb --help
```

The external programs are checked separately by autoVB when a workflow is run. Make sure `formchk`, `xmvb`, `automr`, and `rungms` are available when the corresponding workflow requires them.

## Command-Line Tools

Run `<command> --help` to see all options.

| Command | Purpose |
| --- | --- |
| `autovb` | Run the complete autoVB workflow from an `.autovb`, `.gjf`, or `.xmi` input. |
| `xyz2nbo` | Generate a Gaussian NBO `.gjf` input from an `.xyz` file. |
| `nbo2xmi` | Generate an XMVB `.xmi` input from an NBO `.fch`/`.chk` result. |
| `fch2vb` / `fch2xmi` | Convert `.fch` orbitals into a minimal XMVB `.xmi` initial guess. |
| `xmo2svg` / `draw_xmo` | Convert an XMVB `.xmo` output into SVG valence-bond structures. |
| `xmo2json` | Convert an XMVB `.xmo` output into a JSON summary. |
| `xmo2xmi` | Replace an `.xmi` initial guess with the matching `.orb` file. |

Typical examples:

```bash
autovb C6H6.autovb
xyz2nbo C6H6.xyz 6-31g*
nbo2xmi C6H6.fch 6-31g*
fch2xmi C6H6.fch -o C6H6_guess.xmi
xmo2svg C6H6_vb.xmo --weight both
xmo2json C6H6_vb.xmo
```

See [UserGuide.md](UserGuide.md) for input syntax, active-space selection, GVB/BOVB workflows, plotting options, and detailed examples.
