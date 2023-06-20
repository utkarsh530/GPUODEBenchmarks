a=8
max_a=$1
XLA_PYTHON_CLIENT_PREALLOCATE=false
while [ $a -le $max_a ]
do
    	# Print the values
    	echo "No. of trajectories = $a"
		python3 ./GPU_ODE_JAX/benchmark_stiff_ode_diffrax.py $a	
    	# increment the value
    	a=$((a*4))
done
