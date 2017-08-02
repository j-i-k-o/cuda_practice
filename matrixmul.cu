#include <iostream>
#include <cstdlib>

//matrix A[i][j]

template<int N>
__device__ int l(int a, int b){
	return a*N + b;
}

template<int N>
__global__ void multiply(double *A, double *B, double *C){
// C[i][k] = \sum_{j} A[i][j] + B[j][k]
	int i = blockIdx.y*blockDim.y+threadIdx.y;
	int k = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<N && k<N){
		double result = 0;
		for(int j=0; j<N; j++){
			result += A[l<N>(i, j)] * B[l<N>(j, k)];
		}
		C[l<N>(i, k)] = result;
	}
}

void random(double* x, int size){
	for(int i=0; i<size; i++){
		x[i] = (double)(rand()%100)/100;
	}
}

template<int N>
void dispMat(const double *A){
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			std::cout << A[i*N+j] << ",";
		}
		std::cout << std::endl;
	}
}

void checkError(){
	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
}

int main(void){
	const int N = 1000;
	const int THREADS_PER_BLOCK = 16;
	
	double *A;
	double *B;
	double *C;

	double *d_A;
	double *d_B;
	double *d_C;

	int memsize = N*N*sizeof(double);

	A = (double *)malloc(memsize);
	B = (double *)malloc(memsize);
	C = (double *)malloc(memsize);

	random(A, N*N);
	random(B, N*N);

	std::cout << "A=" << std::endl;
	dispMat<N>(A);
	std::cout << "B=" << std::endl;
	dispMat<N>(B);

	cudaMalloc((void**)&d_A, memsize);
	checkError();
	cudaMalloc((void**)&d_B, memsize);
	checkError();
	cudaMalloc((void**)&d_C, memsize);
	checkError();

	cudaMemcpy(d_A, A, memsize, cudaMemcpyHostToDevice);
	checkError();
	cudaMemcpy(d_B, B, memsize, cudaMemcpyHostToDevice);
	checkError();

	dim3 blocks((N/THREADS_PER_BLOCK)+1, (N/THREADS_PER_BLOCK)+1, 1);
	dim3 thPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);

	multiply<N><<<blocks,thPerBlock>>>(d_A, d_B, d_C);
	checkError();

	cudaMemcpy(C, d_C, memsize, cudaMemcpyDeviceToHost);
	checkError();

	std::cout << "C=" << C[10*N+10] << std::endl;
	dispMat<N>(C);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(A);
	free(B);
	free(C);

}
