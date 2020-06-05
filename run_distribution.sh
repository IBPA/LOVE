#!/bin/bash



# temp='./config/populate_foodon.ini'

# for i in $(seq 1 5);
# do
#     for j in $(seq 1 100);
#     do
#         # update populate_foodon.ini file
#         populate_configfile="./config/populate_foodon_"$i"_"$j".ini"
#         cp './config/populate_foodon.ini' $populate_configfile
#         foodon_parse="foodon_parse_"$i"_"$j".ini"
#         sed -i 's|foodon_parse.ini|'$foodon_parse'|g' $populate_configfile
#         scoring="scoring_"$i"_"$j".ini"
#         sed -i 's|scoring.ini|'$scoring'|g' $populate_configfile

#         # update foodon_parse.ini file
#         foodon_parse_configfile="./config/foodon_parse_"$i"_"$j".ini"
#         cp './config/foodon_parse.ini' $foodon_parse_configfile
#         skeleton="FoodOn/random_seeds/"$i"/skeleton_candidate_classes_dict_"$j".pkl"
#         sed -i 's|FoodOn/skeleton_candidate_classes_dict.pkl|'$skeleton'|g' $foodon_parse_configfile

#         # update scoring.ini file
#         scoring_configfile="./config/scoring_"$i"_"$j".ini"
#         cp './config/scoring.ini' $scoring_configfile
#         pairs="wiki/random_"$i"/pairs_"$j".pkl"
#         sed -i 's|pairs.pkl|'$pairs'|g' $scoring_configfile
#         populated="wiki/random_"$i"/populated_"$j".pkl"
#         sed -i 's|populated.pkl|'$populated'|g' $scoring_configfile
#         siblings="wiki/random_"$i"/siblings_scores_"$j".csv"
#         sed -i 's|siblings_scores.csv|'$siblings'|g' $scoring_configfile
#         parents="wiki/random_"$i"/parents_scores_"$j".csv"
#         sed -i 's|parents_scores.csv|'$parents'|g' $scoring_configfile

#         sed -i 's|'$temp'|'$populate_configfile'|g' ./script.slurm
#         sleep 0.1s
#         sbatch script.slurm
#         sleep 0.1s

#         temp=$populate_configfile
#     done
# done


temp='./config/populate_foodon.ini'

for i in $(seq 1 100);
do
    # update populate_foodon.ini file
    populate_configfile="./config/populate_foodon_"$i".ini"
    cp './config/populate_foodon.ini' $populate_configfile
    foodon_parse="foodon_parse_"$i".ini"
    sed -i 's|foodon_parse.ini|'$foodon_parse'|g' $populate_configfile
    scoring="scoring_"$i".ini"
    sed -i 's|scoring.ini|'$scoring'|g' $populate_configfile

    # update foodon_parse.ini file
    foodon_parse_configfile="./config/foodon_parse_"$i".ini"
    cp './config/foodon_parse.ini' $foodon_parse_configfile
    skeleton="FoodOn/random_seeds/2/skeleton_candidate_classes_dict_"$i".pkl"
    sed -i 's|FoodOn/skeleton_candidate_classes_dict.pkl|'$skeleton'|g' $foodon_parse_configfile

    # update scoring.ini file
    scoring_configfile="./config/scoring_"$i".ini"
    cp './config/scoring.ini' $scoring_configfile
    pairs="pairs_"$i".pkl"
    sed -i 's|pairs.pkl|'$pairs'|g' $scoring_configfile
    populated="populated_"$i".pkl"
    sed -i 's|populated.pkl|'$populated'|g' $scoring_configfile

    parents="parents_scores_"$i".csv"
    sed -i 's|parents_scores.csv|'$parents'|g' $scoring_configfile
    siblings="siblings_scores_"$i".csv"
    sed -i 's|siblings_scores.csv|'$siblings'|g' $scoring_configfile

    # python3 populate_foodon.py --config_file=$populate_configfile
    # rm ./data/scores/siblings_scores*.csv
    # mv ./data/scores/*.pkl ./data/scores/glove_wiki

    sed -i 's|'$temp'|'$populate_configfile'|g' ./script.slurm
    sleep 0.2s
    sbatch script.slurm
    sleep 0.2s

    temp=$populate_configfile
done
