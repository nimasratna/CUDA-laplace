
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

const int NX = 10;      // X size
const int NY = 10;      // y size

const int MAX_ITER = 1000;  

__global__ void Laplace(float *T_old, float *T_new)
{
	float klamda = 0.1;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	 
	int point = i + j*NX;           
	int top = i + (j + 1)*NX;       
	int down = i + (j - 1)*NX;       
	int right = (i + 1) + j*NX;       
	int left = (i - 1) + j*NX;       
								  
	if (i>0 && i<NX-1  && j>0 && j<NY-1) {
		T_new[point] = klamda*(T_old[right] + T_old[left] + T_old[top] + T_old[down]);
	}
	
		
	
}

// initialization

void Initialize(float *TEMPERATURE)
{
	for (int i = 0; i<NX; i++) {
		for (int j = 0; j<NY; j++) {
			int index = i + j*NX;
			TEMPERATURE[index] = 0.0;
		}
	}

	// set boundary condition

	for (int j = 0; j<NY; j++) {
		int index = j*NX;
		TEMPERATURE[index] = 1000;
	}
	for (int j = 0; j<NY; j++) {
		int index = j;
		TEMPERATURE[index] = 2000;
	}
	//9,14,19
	for (int j = (NY*2)-1; j<NY*NX; j+=NY) {
		int index = j;
		TEMPERATURE[index] = 1000;
	}
	for (int j = (NY *(NX-1) ) ; j<NY*NX; j++) {
		int index = j;
		TEMPERATURE[index] = 1000;
	}
}

int main(int argc, char **argv)
{
	        
	float *_T1, *_T2;  
					   
	float *T = new float[NX*NY];
	std::cout << " start \n";
	Initialize(T);
	for (int j = NY - 1; j >= 0; j--) {
		for (int i = 0; i<NX; i++) {
			int index = i + j*NX;
			std::cout << T[index] << " ";
		}
		std::cout << std::endl;
	}


	cudaMalloc((void **)&_T1, NX*NY * sizeof(float));
	cudaMalloc((void **)&_T2, NX*NY * sizeof(float));

	cudaMemcpy(_T1, T, NX*NY * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(_T2, T, NX*NY * sizeof(float), cudaMemcpyHostToDevice);

	int ThreadsPerBlock = 16;
	dim3 dimBlock(ThreadsPerBlock, ThreadsPerBlock);
	dim3 dimGrid(ceil(float(NX) / float(dimBlock.x)), ceil(float(NY) / float(dimBlock.y)), 1);

	for (size_t i = 0; i < MAX_ITER/2; i++)
	{
		Laplace << <dimGrid, dimBlock >> >(_T1, _T2);
		Laplace << <dimGrid, dimBlock >> >(_T2, _T1);
	}
		
	
	cudaMemcpy(T, _T2, NX*NY * sizeof(float), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	std::cout << " result \n";
	// print the results to screen
	for (int j=NY-1;j>=0;j--) {
	for (int i=0;i<NX;i++) {
	int index = i + j*NX;
	std::cout << T[index] << " ";
	}
	std::cout << std::endl;
	}
	
	delete T;

	cudaFree(_T1);
	cudaFree(_T2);

	return 0;
}
