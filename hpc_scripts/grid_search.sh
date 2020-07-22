#!/bin/bash

# parse argument
if [ -z "$1" ]; then
    echo "Setting mode to local."
    mode="local"
else
    mode=$1
fi


if [ "$mode" = "local" ]; then
    echo "Running grid search locally..."
elif [ "$mode" = "hpc" ]; then
    echo "Running grid search on HPC..."
else # invalid option
    echo "Invalid input '$mode'!"
    exit
fi

temp='./config/populate_foodon.ini'

for alpha in '0.0' '0.1' '0.2' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8' '0.9' '1.0' ;
do
    for random in $(seq 1 100);
    do
        # update populate_foodon.ini file
        populate_configfile="./config/populate_foodon_alpha"$alpha"_"$random".ini"
        cp './config/populate_foodon.ini' $populate_configfile

        logfile="populate_foodon_alpha"$alpha"_"$random".log"
        sed -i 's|populate_foodon.log|'$logfile'|g' $populate_configfile

        scoring="scoring_alpha"$alpha"_"$random".ini"
        sed -i 's|scoring.ini|'$scoring'|g' $populate_configfile

        foodon_parse="foodon_parse_alpha"$alpha"_"$random".ini"
        sed -i 's|foodon_parse.ini|'$foodon_parse'|g' $populate_configfile

        # update scoring.ini file
        scoring_configfile="./config/scoring_alpha"$alpha"_"$random".ini"
        cp './config/scoring.ini' $scoring_configfile

        sed -i 's|alpha = alpha_replace|alpha = '$alpha'|g' $scoring_configfile

        pairs="pairs_alpha"$alpha"_"$random".pkl"
        sed -i 's|pairs.pkl|'$pairs'|g' $scoring_configfile

        populated="populated_alpha"$alpha"_"$random".pkl"
        sed -i 's|populated.pkl|'$populated'|g' $scoring_configfile

        parents="parents_scores_"$random".csv"
        sed -i 's|parents_scores.csv|'$parents'|g' $scoring_configfile

        siblings="siblings_scores_"$random".csv"
        sed -i 's|siblings_scores.csv|'$siblings'|g' $scoring_configfile

        # update foodon_parse.ini file
        foodon_parse_configfile="./config/foodon_parse_alpha"$alpha"_"$random".ini"
        cp './config/foodon_parse.ini' $foodon_parse_configfile

        skeleton="random_seeds/2/skeleton_candidate_classes_dict_"$random".pkl"
        sed -i 's|skeleton_candidate_classes_dict.pkl|'$skeleton'|g' $foodon_parse_configfile

        # run code
        if [ "$mode" = "local" ]; then
            python3 populate_foodon.py --config_file=$populate_configfile
        elif [ "$mode" = "hpc" ]; then
            sed -i 's|'$temp'|'$populate_configfile'|g' ./script.slurm
            sleep 0.1s
            sbatch script.slurm
            sleep 0.1s
        fi

        temp=$populate_configfile
    done
done

# revert to original script.slurm
if [ "$mode" = "hpc" ]; then
    sed -i 's|'$temp'|populate_foodon.ini|g' ./script.slurm
fi
