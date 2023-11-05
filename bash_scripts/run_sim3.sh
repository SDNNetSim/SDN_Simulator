#!/bin/bash

#SBATCH -p cpu-long
#SBATCH -c 28
#SBATCH --mem=48000
#SBATCH -t 3-12
#SBATCH -o slurm-%j.out

# shellcheck disable=SC2164
cd
# shellcheck disable=SC2164
cd /home/arash_rezaee_student_uml_edu/Git/SDN_Simulator/



python run_sim.py --sim_type arash --route_method xt_aware --xt_type without_length
