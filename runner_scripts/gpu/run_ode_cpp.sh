a=8
# max_a=$((2**24))
max_a=$1
while [ $a -le $max_a ]
do
    echo $a
	sed -i "15d" ./GPU_ODE_MPGOS/Lorenz.cu
	sed -i "15 i #define SOLVER RK4" ./GPU_ODE_MPGOS/Lorenz.cu
	sed -i "17d" ./GPU_ODE_MPGOS/Lorenz.cu
	sed -i "17 i const int NT = $a;" ./GPU_ODE_MPGOS/Lorenz.cu

	make clean --directory=./GPU_ODE_MPGOS/
	make --directory=./GPU_ODE_MPGOS/
	./GPU_ODE_MPGOS/Lorenz.exe $a

	sed -i "15d" ./GPU_ODE_MPGOS/Lorenz.cu
	sed -i "15 i #define SOLVER RKCK45" ./GPU_ODE_MPGOS/Lorenz.cu

	make clean --directory=./GPU_ODE_MPGOS/
	make --directory=./GPU_ODE_MPGOS/
	./GPU_ODE_MPGOS/Lorenz.exe $a
	# increment the value
	a=$((a*4))
done
