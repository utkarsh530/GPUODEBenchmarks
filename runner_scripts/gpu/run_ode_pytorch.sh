a=8
max_a=$1
while [ $a -le $max_a ]
do
    	# Print the values
    	echo "No. of trajectories = $a"
		python3 ./GPU_ODE_PyTorch/bench_torchdiffeq.py $a	
    	# increment the value
    	a=$((a*4))
done
