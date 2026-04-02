# autoVB

autoVB is a Python package for generating and running XMVB workflows from Gaussian/NBO based inputs.

## 1. Prerequisites

You need the following external programs available in your runtime environment:

- Gaussian executable (`g16`/`g09`)
- `formchk`
- `xmvb`

Typical environment setup (example paths):

```bash
export GAUSS_EXE=/share/apps/g16_avx2/g16/g16
export PATH=/share/apps/g16_avx2/g16:$PATH
export PATH=/share/apps/xmvb/latest/bin:$PATH
```

## 2. User Installation (Recommended, avoid gcc build issues)

To avoid local C/C++/Fortran compilation (gcc/gfortran), install heavy scientific dependencies from prebuilt binaries first (Conda), then install `autoVB` wheel.

```bash
conda create -n autovb python=3.11 -y
conda activate autovb

# Prebuilt binary dependencies
conda install -c conda-forge numpy pyscf -y
conda install -c mokit -c conda-forge mokit -y
pip install pyssian

# Install autoVB from built wheel (recommended)
pip install --no-deps autovb-0.1.0-py3-none-any.whl
```

Why `--no-deps`: dependencies are already installed from Conda binaries, so pip will not try source builds.

## 3. Developer Installation

```bash
conda create -n autovb-dev python=3.11 -y
conda activate autovb-dev
conda install -c conda-forge numpy pyscf -y
conda install -c mokit -c conda-forge mokit -y
pip install pyssian
pip install -e .
```

## 4. Build Package (for maintainers)

Run at project root:

```bash
python -m pip install -U pip build twine
python -m build
twine check dist/*
```

Artifacts will be generated in `dist/`:

- `autovb-0.1.0-py3-none-any.whl`
- `autovb-0.1.0.tar.gz`

## 5. Install From Local Dist Files

```bash
pip install --no-deps dist/autovb-0.1.0-py3-none-any.whl
```

If your machine is fully pip-based and has suitable binary wheels for dependencies, you may also use:

```bash
pip install dist/autovb-0.1.0-py3-none-any.whl
```

If pip tries to compile dependencies (gcc/gfortran errors), switch back to the Conda-first workflow in Section 2.

## 6. Verify Installation

```bash
autovb --help
```