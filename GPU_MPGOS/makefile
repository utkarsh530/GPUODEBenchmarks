INCL_DIR = -I$(HOME)/ode_solver_tests/Lorenz_RK4/GPU_MPGOS/SourceCodes
CMPL_OPT = -O3 -std=c++11 --ptxas-options=-v --gpu-architecture=sm_70 -lineinfo -maxrregcount=64
SOURCE   = Lorenz.cu

all: Lorenz.exe

Lorenz.exe: $(SOURCE)
	nvcc -o	Lorenz.exe $(SOURCE) $(INCL_DIR) $(CMPL_OPT)

clean:
	rm -f Lorenz.exe