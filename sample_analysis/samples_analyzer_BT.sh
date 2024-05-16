#!/bin/bash
# Bash script to submit job to slurm

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=samples_analyzer
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=5000m 
#SBATCH --time=10:00:00
#SBATCH --account=mejn0
#SBATCH --partition=standard

/bin/hostname 
python samples_analyzer_BT.py
