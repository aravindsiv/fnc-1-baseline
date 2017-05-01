declare -a arr=("no" "lstm" "gru")

for i in `seq 0 4`;
do
	for j in "${arr[@]}";
	do
		python keras_models.py -fold $i -rnn $j
	done
done