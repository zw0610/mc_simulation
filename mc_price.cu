#include <iostream>
#include <math.h>
#include <time.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <curand.h>
#include <curand_kernel.h>

#include "param.hpp"


typedef struct {

	bool avail;
	float x;

	__device__ void update( curandState_t &state, const float sqrdt ) {

		x = sqrdt*curand_normal(&state);
		avail = true;

	}

	__device__ void get( float &out ) {

		out = x;
		avail = false;

	}

} rand_real;

/* Device functions */

__global__ void mc_init(unsigned int seed, curandState_t* states) {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	curand_init(seed, idx, 0, &states[idx]);
}

__global__ void mc_simulation(
	curandState_t * states,
	float * d_s,
	const float T,
    const float K,
    const float B,
    const float S0,
    const float sigma,
    const float mu,
    const float r,
    const float dt,
 	const float sqrdt ) {

	/* Initialize Shared Memory */
	__shared__ rand_real rr[num_rand_thread];

	// Get the thread index. If 0, Assemble; if non-zero, generate random //
	const unsigned int local_idx = threadIdx.x;
	const unsigned int path_idx = blockIdx.x;

	unsigned int count = 0;

	float curr_s = 0.0f;

	if ( local_idx==num_rand_thread ) {
		/* Principle Thread, Assemble Role */
		for ( count = 0; count < num_step; count++ ) {
			float curr_rand_real = 0.0f;
			bool yet_new_rand = true;
			while ( yet_new_rand ) {
				if ( rr[path_idx*num_step + count].avail ) {
					rr[path_idx*num_step + count].get(curr_rand_real);
					yet_new_rand = false;
				}
			}

			curr_s = curr_s + mu*curr_s*dt + sigma*curr_s*curr_rand_real;

		}
		d_s[path_idx] = curr_s;
	} else {
		/* Supporting Thread, Generating Random Numbers */
		rr[local_idx].avail = false;
		rr[local_idx].x = 0.0f;

		while ( !(rr[local_idx].avail) ) {
			rr[local_idx].update( states[path_idx*num_step + count*num_rand_thread + local_idx], sqrdt );
			count++;
		}

	}


}



/* Host functions */
int main () {

	thrust::device_vector<float> d_s(num_path, 0.0f);
	float * d_s_ptr = thrust::raw_pointer_cast(d_s.data());

	curandState_t *state_list;
	cudaMalloc((void**) &state_list, (num_path*num_step) * sizeof(curandState_t));
	time_t timer;
	mc_init<<<(num_path*num_step),1024>>>( time(&timer), state_list);

	mc_simulation<<<num_path, (num_rand_thread+1)>>>( state_list, d_s_ptr, T, K, B, S0, sigma, mu, r, dt, sqrdt );

	thrust::host_vector<float> h_s = d_s;

	cudaFree(state_list);

	return 0;
}
