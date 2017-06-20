#!/usr/bin/env/bash
declare -a arr=("lstm" "gru")
cd ..
for i in "${arr[@]}";
do
	for j in `seq 2 4`;
	do
		python keras_models.py -fold $j -rnn $i 
	done
done
