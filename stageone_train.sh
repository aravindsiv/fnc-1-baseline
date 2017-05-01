declare -a arr=("logistic" "svm")

for i in `seq 0 4`;
do
	for j in "${arr[@]}";
	do
		python kfold_stagone.py -fold $i -model $j
	done
done
