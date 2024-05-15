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
#SBATCH --array=1-77 

/bin/hostname 
# ./SBM-BT -i ../../data/gml_files/AddHealth/highschool_$SLURM_ARRAY_TASK_ID.gml -o ../../data/samples/AddHealth/SBM-BT/highschool_$SLURM_ARRAY_TASK_ID.csv #SBATCH --array=1-84 

./SBM-BT.exe -i ../../data/gml_files/villages/anyinteractions$SLURM_ARRAY_TASK_ID.gml -o ../../data/samples/villages/SBM-BT/anyinteractions$SLURM_ARRAY_TASK_ID.csv 
# ./SBM-BT.exe -i ../../data/gml_files/villages/anyinteractions1.gml -o ../../data/samples/villages/SBM-BT/anyinteractions1.csv 

# ./SBM -i ../../data/gml_files/misc/test.gml -o ../../data/samples/misc/SBM/test.csv
# ./SBM-BT -i ../../data/gml_files/AddHealth/highschool_9.gml -o ../../data/samples/AddHealth/SBM-BT/highschool_9.csv
# ./SBM-BT -i ../../data/gml_files/AddHealth/highschool_83.gml -o ../../data/samples/AddHealth/SBM-BT/highschool_83.csv
# ./SBM-BT -i ../../data/gml_files/AddHealth/highschool_54.gml -o ../../data/samples/AddHealth/SBM-BT/highschool_54.csv
# ./SBM-BT.exe -i ../../data/gml_files/AddHealth/highschool_52.gml -o ../../data/samples/AddHealth/SBM-BT/highschool_52.csv
# ./SBM-BT -i ../../data/gml_files/AddHealth/highschool_27.gml -o ../../data/samples/AddHealth/SBM-BT/highschool_27.csv

# ./SBM-BT.exe -i ../../data/gml_files/AddHealth/highschool_81.gml -o ../../data/samples/AddHealth/SBM-BT/highschool_81.csv
# ./SBM-BT.exe -i ../../data/gml_files/AddHealth/highschool_75.gml -o ../../data/samples/AddHealth/SBM-BT/highschool_75.csv
# ./SBM-BT.exe -i ../../data/gml_files/AddHealth/highschool_49.gml -o ../../data/samples/AddHealth/SBM-BT/highschool_49.csv
# ./SBM-BT.exe -i ../../data/gml_files/AddHealth/highschool_43.gml -o ../../data/samples/AddHealth/SBM-BT/highschool_43.csv

# ./SBM-BT.exe -i ../../data/gml_files/AddHealth/highschool_38.gml -o ../../data/samples/AddHealth/SBM-BT/highschool_38.csv

# ./SBM-BT.exe -i ../../data/gml_files/AddHealth/highschool_84.gml -o ../../data/samples/AddHealth/SBM-BT/highschool_84.csv