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
| `xmoview` / `xmo2svg` / `draw_xmo` | View XMO structures as 2D SVG, 3D PNG, or interactive HTML (`-f svg/png/html`). |
| `xmo2png` | Alias for `xmoview` with PNG as the default output format. |
| `xmo2html` | Alias for `xmoview` with interactive HTML as the default output format. |
| `xmo2json` | Convert an XMVB `.xmo` output into a JSON summary. |
| `xmo2xmi` | Replace an `.xmi` initial guess with the matching `.orb` file. |

Typical examples:

```bash
autovb C6H6.autovb
xyz2nbo C6H6.xyz 6-31g*
nbo2xmi C6H6.fch 6-31g*
fch2xmi C6H6.fch -o C6H6_guess.xmi
xmoview C6H6_vb.xmo
xmoview C6H6_vb.xmo -f png
xmo2html C6H6_vb.xmo
xmo2json C6H6_vb.xmo
```

See [UserGuide.md](UserGuide.md) for input syntax, active-space selection, GVB/BOVB workflows, plotting options, and detailed examples.

For the optional PyVista renderer, install `pip install -e '.[visualization]'`, then run
`xmoview C6H5NH2_nbo_vb.xmo -f png`.
The output is `C6H5NH2_nbo_vb.png`; a working EGL/OSMesa or desktop OpenGL backend is required even for off-screen rendering.

For an offline interactive HTML, install `pip install -e '.[visualization-html]'` and use
`xmoview C6H5NH2_nbo_vb.xmo -f html --view optimized3d -m 6`.

All drawing aliases share defaults: top 6 structures, CC/Lowdin weights displayed (ranked by Lowdin), 3 panels per row, and hidden atom/connection labels. SVG hides ordinary hydrogens by default; PNG/HTML show all hydrogens by default. Use `--show-hydrogens` or `--hide-hydrogens` to override; active hydrogens remain visible. Use `-o figure.svg/png/html` for a custom output name; its extension determines the format. `--layout` controls SVG layout and `--view` controls the PNG/HTML camera (default: `optimized3d`); `--projection` remains a compatibility alias. HTML uses Rank tabs and rejects grid/size/transparent-background options.

PNG/HTML retain the original element colors by default (`--color-scheme element`). Use `--color-scheme highlight` for a light-gray framework and bright-red active electron-pair lines, dots and charge-change labels; atom spheres retain their element colors. For example: `xmo2png molecule.xmo --color-scheme highlight` or `xmo2html molecule.xmo --color-scheme highlight`.

PNG includes one shared atom-color legend below the grid; HTML shows the same visible-element colors in its upper-right corner. Only visible elements are listed, using the atom sphere colors for either color scheme. Use `--hide-legend` to hide this key; structure weights are retained. SVG has no atom-color legend.

Charge-change labels are larger and bold in SVG/PNG/HTML and use the full minus sign `−`. Add `--circle-charges` to draw an outline circle around each complete charge label; circles are off by default. `--hide-charges` hides both text and circles.
