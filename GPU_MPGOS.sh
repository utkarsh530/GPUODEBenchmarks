a=10
max_a=$((10**7))
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
    	a=$((a*10))
done
