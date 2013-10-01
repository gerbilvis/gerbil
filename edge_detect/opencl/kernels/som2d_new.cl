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

__kernel void calculate_distances(__global const float* som_data,
                                  __constant float* input_vector,
                                  __global float* output_distances,
                                  const int som_size_x,
                                  const int som_size_y,
                                  const int neuron_size,
                                  __local float* reduction_buff)
{
    int local_id_x = get_local_id(0);
    int local_id_y = get_local_id(1);

    int global_id_y = get_global_id(1);
    int group_size_x = get_local_size(0);

    int group_id_x = get_group_id(0);

    const int local_vector_idx = local_id_y * group_size_x + local_id_x;
    const int global_vector_idx = (global_id_y * som_size_x + group_id_x) * neuron_size;

    __local float* vec_reduction_buff = reduction_buff + local_id_y * group_size_x;

    if((local_id_x < neuron_size) && (global_id_y < som_size_y))
    {
        int global_idx = global_vector_idx + local_id_x;

        float diff = som_data[global_idx] - input_vector[local_id_x];
       // float diff = som_data[global_vector_idx] - input_vector[global_vector_idx % neuron_size];

        vec_reduction_buff[local_id_x] = diff * diff;
    }
    else
    {
        vec_reduction_buff[local_id_x] = 0.f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);


    for (unsigned int s = group_size_x/2; s > 0; s >>= 1)
    {
        if (local_id_x < s)
        {
            vec_reduction_buff[local_id_x] += vec_reduction_buff[local_id_x + s];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(local_id_x == 0 && global_id_y < som_size_y)
    {
        int global_idx = global_id_y  * som_size_x + group_id_x;
        output_distances[global_idx] = vec_reduction_buff[0];
    }
}

__kernel void find_global_min(__global const float* values,
                              __global float* min_values,
                              __global int* min_indexes,
                              __local float* reduction_buff,
                              const int vector_size)
{
    int local_id_x = get_local_id(0);
    int group_size_x = get_local_size(0);

    int global_id_x = get_global_id(0);
    int group_id_x = get_group_id(0);

    if(global_id_x < vector_size)
        reduction_buff[local_id_x] = values[global_id_x];
    else
        reduction_buff[local_id_x] = 0xFFFF; // max float should be used

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = group_size_x/2; s > 0; s >>= 1)
    {
        if (local_id_x < s)
        {
            reduction_buff[local_id_x] = fmin(reduction_buff[local_id_x],
                                              reduction_buff[local_id_x + s]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(reduction_buff[0] == values[global_id_x])
    {
        min_values[group_id_x] = reduction_buff[0];
        min_indexes[group_id_x] = global_id_x;
    }
    //}
}


__kernel void update_network(__global float* som_data,              /* 0 */
                             __constant float* winner_vector,       /* 1 */
                             const int som_size_x,                  /* 2 */
                             const int som_size_y,                  /* 3 */
                             const int neuron_size,                 /* 4 */
                             const int winner_x,                    /* 5 */
                             const int winner_y,                    /* 6 */
                             const int offset_x,                    /* 7 */
                             const int offset_y,                    /* 8 */
                             const int width,                       /* 9 */
                             const int height,                      /* 10 */
                             const float sigma_square,              /* 11 */
                             const float learning_rate,             /* 12 */
                             __local float* weights)                /* 13 */
{

    int local_id_x = get_local_id(0);
    int local_id_y = get_local_id(1);

    int global_id_x = get_global_id(0);
    int global_id_y = get_global_id(1);

    int group_id_x = get_group_id(0);
    int group_id_y = get_group_id(1);

    int global_translated_x = group_id_x + offset_x;
    int global_translated_y = global_id_y + offset_y;

    int global_vector_idx = global_translated_y * som_size_x * neuron_size
                            + global_translated_x * neuron_size;

    if(local_id_x == 0)
    {
        int diff_x = global_translated_x - winner_x;
        int diff_y = global_translated_y - winner_y;

        float distance_squared = diff_x * diff_x + diff_y * diff_y;
        float fake_gauss = exp(-(distance_squared)/ (2.f * sigma_square));

        weights[local_id_y] = learning_rate * fake_gauss;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float weight = weights[local_id_y];

    if(local_id_x < neuron_size && global_id_y < height && weight >= 0.01f)
    {
        int global_idx = global_vector_idx + local_id_x;
        som_data[global_idx] += (winner_vector[local_id_x] - som_data[global_idx]) * weight;
    }
}
