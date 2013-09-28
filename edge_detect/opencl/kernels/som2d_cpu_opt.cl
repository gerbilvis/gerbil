#define DEBUG

#ifdef DEBUG
#define assert(x) \
            if (! (x)) \
            { \
                printf((__constant char*)"Assert(%s) failed in line: %d\n", \
                       (__constant char*)#x, __LINE__); \
            }
#else
        #define assert(X)
#endif

float sum(float4 in)
{
    return dot(in, (float4)(1.0f, 1.0f, 1.0f, 1.0f));
}

__kernel void calculate_distances(__global const float4* som_data,
                                  __constant float4* input_vector,
                                  __global float* output_distances,
                                  const int neuron_vec_size)
{
    int global_id = get_global_id(0);
    const int global_vector_idx = global_id * neuron_vec_size;

    float4 distance = 0;

    for(int i = 0; i < neuron_vec_size; ++i)
    {
        float4 diff = som_data[global_vector_idx + i] - input_vector[i];
        distance += diff * diff;
    }

    output_distances[global_id] = sum(distance);
}

__kernel void find_global_min(__global const float* values,
                              __global float* min_values,
                              __global int* min_indexes,
                              const int chunk_size)
{
    int global_id_x = get_global_id(0);
    int offset = chunk_size * global_id_x;

    __global const float* ptr = values + chunk_size * global_id_x;

    float min_val = MAXFLOAT;
    int min_idx = offset;

    for(int i = 0; i < chunk_size; ++i)
    {
        float val = ptr[i];

        if(val < min_val)
        {
            min_val = val;
            min_idx = i + offset;
        }
    }

    min_values[global_id_x] = min_val;
    min_indexes[global_id_x] = min_idx;
}


__kernel void update_network(__global float4* som_data,              /* 0 */
                             __constant float4* winner_vector,       /* 1 */
                             const int som_size_x,                  /* 2 */
                             const int neuron_vec_size,                 /* 4 */
                             const int2 winner,
                             const int2 offset,
                             const float sigma_square,              /* 11 */
                             const float learning_rate)             /* 12 */
{
    int global_id_x = get_global_id(0);
    int global_id_y = get_global_id(1);

    //int neuron_vec_size = neuron_size >> 2;

    int global_translated_x = global_id_x + offset.x;
    int global_translated_y = global_id_y + offset.y;

    int2 pos = (int2)(global_translated_x, global_translated_y);

    //printf("get_global_size(0): %d\n", get_global_size(0));

    int global_vector_idx = (global_translated_y * som_size_x
                            + global_translated_x) * neuron_vec_size;


    float2 diff = convert_float2(pos - winner);
    float2 tmp = diff * diff;
    float distance_squared = dot(tmp, (float2)(1.0f, 1.0f));
    float fake_gauss = native_exp(-(distance_squared)/ (2.f * sigma_square));

    float weight = learning_rate * fake_gauss;

    if(weight >= 0.01f)
    {
        for(int i = 0; i < neuron_vec_size; ++i)
        {
            som_data[global_vector_idx + i] += (winner_vector[i]
                               - som_data[global_vector_idx + i]) * weight;
        }
    }
}
