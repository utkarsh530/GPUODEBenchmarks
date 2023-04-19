#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>

#include "Lorenz_SystemDefinition.cuh"
#include "SingleSystem_PerThread_Interface.cuh"

#define PI 3.14159265358979323846

using namespace std;

// Solver Configuration
#define SOLVER RKCK45
#define PRECISION float  // float, double
const int NT = 8;
const int SD   = 3;     // SystemDimension
const int NCP  = 1;     // NumberOfControlParameters
const int NSP  = 0;     // NumberOfSharedParameters
const int NISP = 0;     // NumberOfIntegerSharedParameters
const int NE   = 0;     // NumberOfEvents
const int NA   = 0;     // NumberOfAccessories
const int NIA  = 0;     // NumberOfIntegerAccessories
const int NDO  = 0;     // NumberOfPointsOfDenseOutput

void Linspace(vector<PRECISION>&, PRECISION, PRECISION, int);
void FillSolverObject(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>&, const vector<PRECISION>&, int);
void SaveData(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>&, int);

int main(int argc, char *argv[])
{
	int NumberOfProblems = NT;
	int BlockSize        = 1024;
	
	ListCUDADevices();
	
	int MajorRevision  = 3;
	int MinorRevision  = 5;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);
	
	PrintPropertiesOfSpecificDevice(SelectedDevice);
	
	
	int NumberOfParameters_R = NumberOfProblems;
	PRECISION R_RangeLower = 0.0;
    PRECISION R_RangeUpper = 21.0;
		vector<PRECISION> Parameters_R_Values(NumberOfParameters_R,0);
		Linspace(Parameters_R_Values, R_RangeLower, R_RangeUpper, NumberOfParameters_R);
	
	
	ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION> ScanLorenz(SelectedDevice);
	
	ScanLorenz.SolverOption(ThreadsPerBlock, BlockSize);
	ScanLorenz.SolverOption(InitialTimeStep, 1.0e-3);
	
	
	clock_t SimulationStart;
	clock_t SimulationEnd;
	
	FillSolverObject(ScanLorenz, Parameters_R_Values, NT);
	
	ScanLorenz.SynchroniseFromHostToDevice(All);
	
	SimulationStart = clock();
	ScanLorenz.Solve();
	ScanLorenz.InsertSynchronisationPoint();
	ScanLorenz.SynchroniseSolver();
	SimulationEnd = clock();
	
	ScanLorenz.SynchroniseFromDeviceToHost(All);
	
	cout << "Total simulation time:           " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC << "ms" << endl;
	cout << "Simulation time / 1000 RK4 step: " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC << "ms" << endl;
	cout << "Ensemble size:                   " << NT << endl << endl;
		
	
	ofstream datafile;
	if (SOLVER == RK4){
		datafile.open ("./data/cpp/MPGOS_times_unadaptive.txt",ios::app);
		datafile << NT << "\t"<< 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC <<"\n";
		datafile.close();
	}else{
		
		datafile.open ("./data/cpp/MPGOS_times_adaptive.txt",ios::app);
		datafile << NT << "\t"<< 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC <<"\n";
		datafile.close();
	}
	
	//SaveData(ScanLorenz, NT);
	
	cout << "Test finished!" << endl;
}

// AUXILIARY FUNCTION -----------------------------------------------------------------------------

void Linspace(vector<PRECISION>& x, PRECISION B, PRECISION E, int N)
{
    PRECISION Increment;
	
	x[0]   = B;
	
	if ( N>1 )
	{
		x[N-1] = E;
		Increment = (E-B)/(N-1);
		
		for (int i=1; i<N-1; i++)
		{
			x[i] = B + i*Increment;
		}
	}
}

void FillSolverObject(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>& Solver, const vector<PRECISION>& R_Values, int NumberOfThreads)
{
	int ProblemNumber = 0;
	for (int k=0; k<NumberOfThreads; k++)
	{
		Solver.SetHost(ProblemNumber, TimeDomain,  0, 0 );
		Solver.SetHost(ProblemNumber, TimeDomain,  1, 0.001*1000.0 );
		
		Solver.SetHost(ProblemNumber, ActualState, 0, 1.0 );
		Solver.SetHost(ProblemNumber, ActualState, 1, 0.0 );
		Solver.SetHost(ProblemNumber, ActualState, 2, 0.0 );
		
		Solver.SetHost(ProblemNumber, ControlParameters, 0, R_Values[k] );
		
		ProblemNumber++;
	}
}

void SaveData(ProblemSolver<NT,SD,NCP,NSP,NISP,NE,NA,NIA,NDO,SOLVER,PRECISION>& Solver, int NumberOfThreads)
{
	ofstream DataFile;
	DataFile.open ( "Lorenz.txt" );
	
	int Width = 18;
	DataFile.precision(10);
	DataFile.flags(ios::scientific);
	
	for (int tid=0; tid<NumberOfThreads; tid++)
	{
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ControlParameters, 0) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ActualState, 0) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ActualState, 1) << ',';
		DataFile.width(Width); DataFile << Solver.GetHost<PRECISION>(tid, ActualState, 2);
		DataFile << '\n';
	}
	
	DataFile.close();
}
