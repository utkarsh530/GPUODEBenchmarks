a=8
max_a=$((2**24))
while [ $a -lt $max_a ]
do
    	# Print the values
    	echo $a
	sed -i "17d" ./GPU_MPGOS/Lorenz.cu
	sed -i "17 i const int NT = $a;" ./GPU_MPGOS/Lorenz.cu
	make clean --directory=./GPU_MPGOS/
	make --directory=./GPU_MPGOS/
	./GPU_MPGOS/Lorenz.exe $a
    	# increment the value
    	a=$((a*4))
done
