a=2
max_a=4

path="SDE"
if [ -d "./data/${path}/CRN" ] 
then
	rm -rf "./data/${path}/CRN"
	mkdir -p "./data/${path}/CRN"

	rm -rf "./data/CPU/${path}/CRN"
	mkdir -p "./data/CPU/${path}/CRN"
else
	mkdir -p "./data/${path}/CRN"
	mkdir -p "./data/CPU/${path}/CRN"
fi

while [ $a -le $max_a ]
do
    	# Print the values
    	echo $a
		julia --threads=16 --project="./GPU_ODE_Julia/" ./GPU_ODE_Julia/sde_examples/bench_crn_model.jl $a
    	# increment the value
    	a=$((a*2))
done
