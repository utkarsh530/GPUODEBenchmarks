a=8
max_a=$1
backend=$2
if [ -d "./data/devices/${backend}" ] 
then
	rm -rf "./data/devices/${backend}"
	mkdir -p "./data/devices/${backend}"
else
	mkdir -p "./data/devices/${backend}"
fi

while [ $a -le $max_a ]
do
    	# Print the values
    	echo $a
		julia --project="./GPU_ODE_Julia/" ./GPU_ODE_Julia/bench_multi_device.jl $a $backend
    	# increment the value
    	a=$((a*4))
done
