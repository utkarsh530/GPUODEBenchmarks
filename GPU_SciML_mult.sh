a=8
max_a=$((2**24))
while [ $a -lt $max_a ]
do
    	# Print the values
    	echo $a
	julia --project="./GPU_ODE_SciML/" ./GPU_ODE_SciML/benchmark_multi_device.jl $a "oneAPI"
    	# increment the value
    	a=$((a*4))
done
