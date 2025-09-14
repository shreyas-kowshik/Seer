#!/bin/bash
#SBATCH --job-name=seer_eval_v1
#SBATCH --partition=general          # partition name
#SBATCH --qos=normal            # qos
#SBATCH --gres=gpu:A6000:8               # request 1 GPU
#SBATCH --cpus-per-task=12          # number of CPUs
#SBATCH --time=48:00:00          # time (1 day)
#SBATCH --mem-per-cpu=4G
#SBATCH --output=outcome_%j.out      # Standard output file (%j is job ID)
#SBATCH --error=outcome_%j.err       # Standard error file (%j is job ID)

# Run inside Apptainer container
cd /home/skowshik/work/Seer/
apptainer exec -B $HOME/.bashrc:/root/.bashrc --nv --fakeroot --writable-tmpfs /home/skowshik/work/wmsrl/seer_v4.sif bash scripts/CALVIN_ABC_D/Seer-Large/eval_D_D_large.sh
