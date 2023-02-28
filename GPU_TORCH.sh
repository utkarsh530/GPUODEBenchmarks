a=8
max_a=$((2**24))
while [ $a -lt $max_a ]
do
    	# Print the values
    	echo $a
		python3 GPU_TORCH/torchdiffeq_bench.py $a	
    	# increment the value
    	a=$((a*4))
done
