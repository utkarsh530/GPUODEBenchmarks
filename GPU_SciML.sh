a=10
max_a=$((10**7))
while [ $a -lt $max_a ]
do
    	# Print the values
    	echo $a
	julia --project="./GPU_ODE_SciML/" ./GPU_ODE_SciML/benchmark.jl $a
    	# increment the value
    	a=$((a*10))
done
