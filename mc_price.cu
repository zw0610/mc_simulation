#include <iostream>
#include <math.h>
#include <time.h>

#include <stdio.h>
#include <stdlib.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "param.hpp"


#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \ printf("Error at %s:%d\n",__FILE__,__LINE__); \ return EXIT_FAILURE;}} while(0)



/* Device functions */

__global__ void mc_simulation(
	float * d_r,
	float * d_s,
	const float T,
    const float K,
    const float B,
    const float S0,
    const float sigma,
    const float mu,
    const float r,
    const float dt,
 	const float sqrdt,
	const unsigned int num_round,
	const unsigned int num_last_round) {

	/* Initialize Shared Memory */
	/* Create double shared memory space */
	__shared__ float rr[(num_rand_thread<<1)];

	// Get the thread index. If blockDim.x-1, Assemble; if not-the-last, reading/loading random //
	const unsigned int role_idx = threadIdx.x;
	const unsigned int path_idx = blockIdx.x;

	bool loaded = 0;

	float mudt = mu*dt;

	float curr_s = S0;

	if ( role_idx != num_rand_thread ) {
		/* If the role_idx shows it's not the last thread, then loading */
		/* random number from Global Memory to Shared Memory */
		rr[role_idx] = d_r[ path_idx*num_step + role_idx ];
	}
	__syncthreads();
	/* Keep all threads synchronized so assemble thread will not proceed */
	/* before the first segment is fully loaded */


	for (size_t i = 0; i < (num_round-1); i++) {

		if ( role_idx == num_rand_thread ) {
			for (size_t j = 0; j < num_rand_thread; j++) {
				curr_s = curr_s + mudt*curr_s + sigma*curr_s*rr[ loaded*num_rand_thread + j ];
			}
		} else {
			rr[ role_idx + (!loaded)*num_rand_thread ] = d_r[ path_idx*num_step + i*num_rand_thread + role_idx ];
		}

		loaded = !loaded;
		__syncthreads();
	}

	if ( role_idx == num_rand_thread ) {
		for (size_t j = 0; j < num_last_round; j++) {
			curr_s = curr_s + mudt*curr_s + sigma*curr_s*rr[ loaded*num_rand_thread + j ];
		}
		d_s[path_idx] = 1.23f;
		//printf("%i %i %f %f\n", role_idx, path_idx, S0, curr_s);
	}

}



/* Host functions */
int main () {

	float *d_s = NULL;
	cudaMalloc((void**) &d_s, num_path * sizeof(float));

	float *d_r = NULL;
	cudaMalloc((void**) &d_r, num_path * num_step * sizeof(float));

	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateNormal(gen, d_r, (num_path*num_step), 0.0f, sqrdt);
	curandDestroyGenerator(gen);

	std::cout << "Random number generation is finished." << std::endl;

	unsigned int N_ROUND = (num_step/num_rand_thread);
	unsigned int N_LASTR = num_step - N_ROUND*num_rand_thread;

	mc_simulation<<<num_path, (num_rand_thread+1)>>>( d_r, d_s, T, K, B, S0, sigma, mu, r, dt, sqrdt, N_ROUND, N_LASTR );

	float h_s[num_path];
	cudaDeviceSynchronize();
	cudaMemcpy(&h_s, d_s, num_path*sizeof(float),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	

	cudaFree(d_s);
	cudaFree(d_r);
	cudaDeviceSynchronize();

	return 0;
}
