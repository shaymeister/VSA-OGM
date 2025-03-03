#!/bin/bash
#SBATCH --job-name=vsa-ogm-test-mapper
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1g.10gb:1                # up to 8; only request what you need

#SBATCH --output=logs/slurm/vsa-ogm-test-mapper-%j.out
#SBATCH --error=logs/slurm/vsa-ogm-test-mapper-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=32G
#SBATCH --time=2-0:00:00

# User specific aliases and functions
export CONDA_PKGS_DIRS=/projects/mparsa/ssnyde9/.conda/cache/
export CONDA_ENVS_DIRS=/projects/mparsa/ssnyde9/.conda/envs/

# Initialize Conda
__conda_setup="$('/opt/sw/spack/apps/linux-rhel8-x86_64_v2/gcc-10.3.0/miniconda3-22.11.1-gy/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/sw/spack/apps/linux-rhel8-x86_64_v2/gcc-10.3.0/miniconda3-22.11.1-gy/etc/profile.d/conda.sh" ]; then
        . "/opt/sw/spack/apps/linux-rhel8-x86_64_v2/gcc-10.3.0/miniconda3-22.11.1-gy/etc/profile.d/conda.sh"
    else
        export PATH="/opt/sw/spack/apps/linux-rhel8-x86_64_v2/gcc-10.3.0/miniconda3-22.11.1-gy/bin:$PATH"
    fi
fi
unset __conda_setup

# Start Job
conda activate VSA-OGM
nvidia-smi
which python