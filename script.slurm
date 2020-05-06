#!/bin/bash

# options for sbatch
#SBATCH --job-name=FoodOntology # Job name
#SBATCH --nodes=1 # should never be anything other than 1
#SBATCH --ntasks=1 # number of cpus to use
#SBATCH --time=12:00:00 # Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".
#SBATCH --mem=7000 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --partition=production # cluster partition
#SBATCH --output=./output/log/stdout_%j.out # File to which STDOUT will be written
#SBATCH --error=./output/log/stderr_%j.err # File to which STDERR will be written

# for calculating the amount of time the job takes and echo the hostname
begin=`date +%s`
echo $HOSTNAME

# load modules
module load anaconda3

# run
python3 populate_foodon.py --config_file=./config/populate_foodon.ini

# getting end time to calculate time elapsed
end=`date +%s`
elapsed=`expr $end - $begin`
echo Time taken: $elapsed
