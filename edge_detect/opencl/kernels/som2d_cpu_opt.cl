//#define DEBUG

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
                                  __global float* output_distances)
{
    const int global_id = get_global_id(0);
    const int global_vector_idx = mul24(global_id, VEC_SIZE);

    float4 distance = 0;

    for(int i = 0; i < VEC_SIZE; ++i)
    {
        const float4 diff = som_data[global_vector_idx + i] - input_vector[i];
        distance += diff * diff;
    }

    output_distances[global_id] = sum(distance);
}

__kernel void find_global_min(__global const float* values,
                              __global float* min_values,
                              __global int* min_indexes,
                              const int chunk_size)
{
    const int global_id_x = get_global_id(0);
    const int offset = chunk_size * global_id_x;

    __global const float* ptr = values + chunk_size * global_id_x;

    float min_val = MAXFLOAT;
    int min_idx = offset;

    for(int i = 0; i < chunk_size; ++i)
    {
        const float val = ptr[i];

        if(val < min_val)
        {
            min_val = val;
            min_idx = i + offset;
        }
    }

    min_values[global_id_x] = min_val;
    min_indexes[global_id_x] = min_idx;
}


__kernel void update_network(__global float4* som_data,
                             __constant float4* winner_vector,
                             const int som_size_x,
                             const int2 winner,
                             const int2 offset,
                             const float sigma_square,
                             const float learning_rate)
{
    const int global_id_x = get_global_id(0);
    const int global_id_y = get_global_id(1);

    const int global_translated_x = global_id_x + offset.x;
    const int global_translated_y = global_id_y + offset.y;

    int2 pos = (int2)(global_translated_x, global_translated_y);


    const int global_vector_idx = (mul24(global_translated_y, som_size_x)
                                  + global_translated_x) * VEC_SIZE;

    const float2 diff = convert_float2(pos - winner);
    const float2 tmp = diff * diff;
    const float distance_squared = dot(tmp, (float2)(1.0f, 1.0f));
    const float fake_gauss = native_exp(-(distance_squared)/ (2.f * sigma_square));

    const float weight = learning_rate * fake_gauss;

    if(weight >= 0.01f)
    {
        for(int i = 0; i < VEC_SIZE; ++i)
        {
            const int idx = global_vector_idx + i;
            som_data[idx] = mix(som_data[idx], winner_vector[i], weight);
        }
    }
}
