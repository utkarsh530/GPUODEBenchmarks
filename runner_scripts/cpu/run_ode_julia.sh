a=8
max_a=$1

path="CPU"
if [ -d "./data/${path}" ] 
then
	rm -rf "./data/${path}/stiff"
	mkdir -p "./data/${path}/stiff"
else
	mkdir -p "./data/${path}/stiff"
fi

while [ $a -le $max_a ]
do
    	# Print the values
    	echo $a
		julia --threads=16 --project="./GPU_ODE_Julia/" ./GPU_ODE_Julia/stiff_odes/benchmark_cpu.jl $a
    	# increment the value
    	a=$((a*4))
done
