#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <curand_kernel.h>

#include <iostream>
#include <iomanip>
#include <numeric>
#include <time.h>


__global__ void mc_pi(float *d, int seed, int n_try){
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	curandState s;
	curand_init(seed, index, 0, &s);

	int inside = 0;
	for(int i=0; i<n_try; i++){
		float x = curand_uniform(&s);
		float y = curand_uniform(&s);
		if(x*x + y*y < 1){
			inside++;
		}
	}

	d[index] = inside / (float)n_try;
}


int main(void){
	const int THREADS = 512;

	std::cout << "input N: ";
	int N;
	std::cin >> N;
	std::cout << "input n_try: ";
	int n_try;
	std::cin >> n_try;

	thrust::device_vector<float> D(N);
	float* d = thrust::raw_pointer_cast(D.data());
	mc_pi<<<(N/THREADS)+1, THREADS>>>(d, time(NULL), n_try);

	thrust::host_vector<float> H = D;

	std::cout << "PI: ";
	std::cout << std::setprecision(16);
	std::cout << 4.0 * std::accumulate(H.begin(), H.end(), 0.0)/N << std::endl;
	return 0;
}
