a=8
max_a=$1

path="CPU"
if [ -d "./data/${path}/SDE" ] 
then
	rm -f "./data/${path}/SDE"/* || true
	mkdir -p "./data/${path}/SDE"
else
	mkdir -p "./data/${path}/SDE"
fi

while [ $a -le $max_a ]
do
    	# Print the values
    	echo $a
		julia --threads=16 --project="./GPU_ODE_Julia/" ./GPU_ODE_Julia/sde_examples/bench_cpu.jl $a
    	# increment the value
    	a=$((a*4))
done
