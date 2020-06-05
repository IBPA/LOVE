#!/bin/bash

# mkdir ./data/FoodOn/multiple

# for i in $(seq 100 $END);
# do
# 	python3 populate_foodon.py
# 	mv ./data/FoodOn/skeleton_candidate_classes_dict.pkl ./data/FoodOn/multiple/skeleton_candidate_classes_dict_$i.pkl
# done


for i in $(seq 6 10);
do
	foodon_parse_configfile="./config/foodon_parse_"$i".ini"
	cp './config/foodon_parse.ini' $foodon_parse_configfile
	sed -i 's|num_seeds_replace|'$i'|g' $foodon_parse_configfile

	populate_foodon_configfile="./config/populate_foodon_"$i".ini"
	cp './config/populate_foodon.ini' $populate_foodon_configfile
	sed -i 's|foodon_parse.ini|foodon_parse_'$i'.ini|g' $populate_foodon_configfile

	for j in $(seq 100 $END);
	do
		python3 populate_foodon.py --config_file=$populate_foodon_configfile
		mv ./data/FoodOn/skeleton_candidate_classes_dict.pkl ./data/FoodOn/random_seeds/$i/skeleton_candidate_classes_dict_$j.pkl
	done
done


# mkdir ./data/scores/random/

# for i in $(seq 100 $END);
# do
# 	python3 populate_foodon.py
# 	mv ./data/scores/pairs.pkl ./data/scores/random/pairs_$i.pkl
# 	mv ./data/scores/populated.pkl ./data/scores/random/populated_$i.pkl
# 	rm ./data/scores/*csv
# done
