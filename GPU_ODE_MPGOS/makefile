ROOT_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
INCL_DIR = -I$(ROOT_DIR)/SourceCodes
CMPL_OPT = -O3 -std=c++11 --ptxas-options=-v --gpu-architecture=sm_70 -lineinfo -maxrregcount=128
SOURCE   = Lorenz.cu

all: Lorenz.exe

Lorenz.exe: $(SOURCE)
	nvcc -o	Lorenz.exe $(SOURCE) $(INCL_DIR) $(CMPL_OPT)

clean:
	rm -f Lorenz.exe
