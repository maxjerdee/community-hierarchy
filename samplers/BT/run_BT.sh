#!/bin/bash
# Bash script to submit job to slurm

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=additive_fixed_groups_get_samples
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=5000m 
#SBATCH --time=10:00:00
#SBATCH --account=mejn0
#SBATCH --partition=standard
#SBATCH --array=1-84 

/bin/hostname 
./BT.exe -i ../../data/gml_files/AddHealth/highschool_$SLURM_ARRAY_TASK_ID.gml -o ../../data/samples/AddHealth/BT/highschool_$SLURM_ARRAY_TASK_ID.csv 

# ./BT.exe -i ../../data/gml_files/villages/anyinteractions$SLURM_ARRAY_TASK_ID.gml -o ../../data/samples/villages/BT/anyinteractions$SLURM_ARRAY_TASK_ID.csv #SBATCH --array=1-77 
