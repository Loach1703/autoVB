#!/bin/bash
#SBATCH -J autovb
#SBATCH -o autovb.log-%j
#SBATCH -e autovb.err-%j
#SBATCH --export=NONE

CONDA_ROOT=/share/apps/anaconda3
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
else
  eval "$("$CONDA_ROOT/bin/conda" shell.bash hook)"
fi
# conda activate mokit-py311

MOKIT_BIN="/pool1/home/chentz/.conda/envs/mokit-py311/bin"
export PATH="$MOKIT_BIN:$PATH"
export PATH="/share/apps/xmvb/latest/bin:$PATH"
export g16root=/share/apps/g16_avx2
source /share/apps/g16_avx2/g16/bsd/g16.profile
module load mpich/4.0.2-gcc13
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

JOB_MEM=${SLURM_MEM_PER_NODE:-4G}
JOB_CPUS=${SLURM_CPUS_PER_TASK:-1}

ulimit -c 0
ulimit -s unlimited
ulimit -v unlimited

inputname=$1
SUBMIT_DIR=$(pwd)
JOB_ID=${SLURM_JOB_ID:-$$}
JOB_NAME=${SLURM_JOB_NAME:-automr}

RUNDIR=/job_dir/${JOB_NAME}_${JOB_ID}
mkdir -p "$RUNDIR"
chmod 700 "$RUNDIR"
GAUSS_SCRDIR=$RUNDIR/scr
export GAUSS_SCRDIR
mkdir ${GAUSS_SCRDIR}
cp "$inputname" "$RUNDIR"/
basename_input=$(basename "$inputname")
outname="${basename_input%.*}.out"

cd "$RUNDIR" || exit 1

# run in the run dir so intermediate files live on the node-local FS
autovb "$basename_input" "$JOB_MEM" "$JOB_CPUS" > "$outname" 2>&1
rc=$?

# copy back results (avoid overwriting original input unless desired)
for f in $(find . -maxdepth 1 -type f); do
  [ "$(basename "$f")" = "$basename_input" ] && continue
  cp -p "$f" "$SUBMIT_DIR"/
done

# cleanup
cd "$SUBMIT_DIR"
rm -rf "$RUNDIR"

exit $rc