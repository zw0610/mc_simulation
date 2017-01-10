#ifndef param_hpp
#define param_hpp

#define num_path 10
#define num_step 365
#define num_rand_thread 4

const float T = 1.0f;
const float K = 100.0f;
const float B = 95.0f;
const float S0 = 100.0f;
const float sigma = 0.2f;
const float mu = 0.1f;
const float r = 0.05f;
const float dt = T/(1.0f*num_step);
const float sqrdt = sqrt(dt);

#endif
