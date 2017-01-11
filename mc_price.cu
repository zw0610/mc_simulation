#include <iostream>
#include <math.h>
#include <time.h>

#include <stdio.h>
#include <stdlib.h>

#include "statistics.hpp"

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "param.hpp"


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
		d_s[path_idx] = curr_s;
		//printf("%i %i %f %f\n", role_idx, path_idx, S0, curr_s);
	}

}



/* Host functions */
int main () {

	thrust::device_vector<float> d_x(num_path, 1.0f);
	float *d_s = thrust::raw_pointer_cast(d_x.data());
	//cudaMalloc((void**) &d_s, num_path * sizeof(float));

	float *d_r = NULL;
	cudaMalloc((void**) &d_r, num_path * num_step * sizeof(float));


	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateNormal(gen, d_r, (num_path*num_step), 0.0f, sqrdt);
	curandDestroyGenerator(gen);

	unsigned int N_ROUND = (num_step/num_rand_thread);
	unsigned int N_LASTR = num_step - N_ROUND*num_rand_thread;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	mc_simulation<<<num_path, (num_rand_thread+1)>>>( d_r, d_s, T, K, B, S0, sigma, mu, r, dt, sqrdt, N_ROUND, N_LASTR );
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Elapsed Time is " << milliseconds << " milliseconds." << std::endl;

	//float h_s[num_path];
	//cudaDeviceSynchronize();
	//cudaMemcpy(&h_s, d_s, num_path*sizeof(float),cudaMemcpyDeviceToHost);

	typedef float T;
	summary_stats_unary_op<T>  unary_op;
    summary_stats_binary_op<T> binary_op;
    summary_stats_data<T>      init;
	init.initialize();

	cudaDeviceSynchronize();
	// compute summary statistics
    summary_stats_data<T> result = thrust::transform_reduce(d_x.begin(), d_x.end(), unary_op, init, binary_op);

    std::cout <<"******Summary Statistics Example*****"<<std::endl;
    //print_range("The data", d_x.begin(), d_x.end());

    std::cout <<"Count              : "<< result.n << std::endl;
    std::cout <<"Minimum            : "<< result.min <<std::endl;
    std::cout <<"Maximum            : "<< result.max <<std::endl;
    std::cout <<"Mean               : "<< result.mean << std::endl;
    std::cout <<"Variance           : "<< result.variance() << std::endl;
    std::cout <<"Standard Deviation : "<< std::sqrt(result.variance_n()) << std::endl;

	cudaFree(d_r);
	cudaDeviceSynchronize();

	return 0;
}
