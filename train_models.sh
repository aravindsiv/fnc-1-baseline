declare -a arr=("logistic" "svm")

for i in `seq 0 4`;
do
	for j in "${arr[@]}";
	do
		python keras_models.py -fold $i -rnn $j
	done
done
