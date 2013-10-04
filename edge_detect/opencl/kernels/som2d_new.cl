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

#define X_DIM 32
#define SOM_SIZE 1024

__kernel void calculate_distances(__global const float* som_data,
                                  __global const float* input_vector,
                                  __global float* output_distances,
                                  const int som_size_x,
                                  const int som_size_y,
                                  const int neuron_size,
                                  __local float* reduction_buff)
{
    size_t local_id_x = get_local_id(0);
    size_t local_id_y = get_local_id(1);

    int global_id_y = get_global_id(1);
    int group_size_x = get_local_size(0);

    int group_id_x = get_group_id(0);

    const int global_vector_idx = (global_id_y * som_size_x + group_id_x) * neuron_size;

    __local volatile float* vec_reduction_buff = reduction_buff + local_id_y * X_DIM * 2;

//    if(local_id_x < neuron_size)
//    {
        int global_idx = global_vector_idx + local_id_x;
        float diff = som_data[global_idx] - input_vector[local_id_x];
        vec_reduction_buff[local_id_x] = diff * diff;

        diff = som_data[global_idx + X_DIM] - input_vector[local_id_x + X_DIM];
        vec_reduction_buff[local_id_x] += diff * diff;

//    }
//    else
//    {
//        vec_reduction_buff[local_id_x] = 0.f;
//    }

    barrier(CLK_LOCAL_MEM_FENCE);


    if(X_DIM >= 512)
    {
        if (local_id_x < 256)
        {
            vec_reduction_buff[local_id_x] += vec_reduction_buff[local_id_x + 256];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(X_DIM >= 256)
    {
        if (local_id_x < 128)
        {
            vec_reduction_buff[local_id_x] += vec_reduction_buff[local_id_x + 128];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(X_DIM >= 128)
    {
        if (local_id_x < 64)
        {
            vec_reduction_buff[local_id_x] += vec_reduction_buff[local_id_x + 64];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }


    if (local_id_x < 32)
    {
        if(X_DIM >= 64)
            vec_reduction_buff[local_id_x] += vec_reduction_buff[local_id_x + 32];
        if(X_DIM >= 32)
            vec_reduction_buff[local_id_x] += vec_reduction_buff[local_id_x + 16];
        if(X_DIM >= 16)
            vec_reduction_buff[local_id_x] += vec_reduction_buff[local_id_x + 8];
        if(X_DIM >= 8)
            vec_reduction_buff[local_id_x] += vec_reduction_buff[local_id_x + 4];
        if(X_DIM >= 4)
            vec_reduction_buff[local_id_x] += vec_reduction_buff[local_id_x + 2];
        if(X_DIM >= 2)
            vec_reduction_buff[local_id_x] += vec_reduction_buff[local_id_x + 1];
    }

    if(local_id_x == 0)
    {
        int global_idx = global_id_y  * som_size_x + group_id_x;
        output_distances[global_idx] = vec_reduction_buff[0];
    }
}

__kernel void find_global_first_pass(__global const float* values,
                                     __global float* min_values,
                                     __global uint* min_indexes,
                                     __local volatile float* rbuff,
                                     const uint vector_size)
{
    const size_t local_id = get_local_id(0);
    const size_t group_size = get_local_size(0);

    const size_t global_id = get_global_id(0);
    const size_t group_id = get_group_id(0);

    if(global_id < vector_size)
        rbuff[local_id] = values[global_id];
    else
        rbuff[local_id] = MAXFLOAT;

    barrier(CLK_LOCAL_MEM_FENCE);

    if(group_size >= 512)
    {
        if(local_id < 256)
        {
            rbuff[local_id] = fmin(rbuff[local_id], rbuff[local_id + 256]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(group_size >= 256)
    {
        if(local_id < 128)
        {
            rbuff[local_id] = fmin(rbuff[local_id], rbuff[local_id + 128]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(group_size >= 128)
    {
        if(local_id < 64)
        {
            rbuff[local_id] = fmin(rbuff[local_id], rbuff[local_id + 64]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }


    if(local_id < 32)
    {
        rbuff[local_id] = fmin(rbuff[local_id], rbuff[local_id + 32]);
        rbuff[local_id] = fmin(rbuff[local_id], rbuff[local_id + 16]);
        rbuff[local_id] = fmin(rbuff[local_id], rbuff[local_id + 8]);
        rbuff[local_id] = fmin(rbuff[local_id], rbuff[local_id + 4]);
        rbuff[local_id] = fmin(rbuff[local_id], rbuff[local_id + 2]);
        rbuff[local_id] = fmin(rbuff[local_id], rbuff[local_id + 1]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(global_id < vector_size)
    {
        float val = rbuff[0];

        if(val == values[global_id])
        {
            min_values[group_id] = val;
            min_indexes[group_id] = global_id;
        }
    }
}


__kernel void find_global_min(__global float* min_values,
                              __global uint* min_indexes,
                              const uint som_width,
                              const uint vector_size,
                              __local volatile float* reduction_buff)
{
    size_t local_id = get_local_id(0);
    size_t global_id = get_global_id(0);
    size_t group_id = get_group_id(0);

    if(global_id < vector_size)
        reduction_buff[local_id] = min_values[global_id];
    else
        reduction_buff[local_id] = MAXFLOAT;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t s = get_local_size(0) / 2; s > 0; s >>= 1)
    {
        if (local_id < s)
        {
            reduction_buff[local_id] = fmin(reduction_buff[local_id],
                                            reduction_buff[local_id + s]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float min_val = reduction_buff[0];

    barrier(CLK_LOCAL_MEM_FENCE);

    if(global_id < vector_size)
    {
        float val = min_values[global_id];

        if(val == min_val)
        {
            if(get_num_groups(0) == 1)
            {
                uint idx = min_indexes[global_id];
                min_indexes[0] = idx % som_width;
                min_indexes[1] = idx / som_width;
            }
            else
            {
                min_values[group_id] = min_val;
                min_indexes[group_id] = min_indexes[global_id];
            }
        }
    }
}





__kernel void update_network(__global float* som_data,              /* 0 */
                             __global const float* input_vector,       /* 1 */
                             __global const uint* winner_idx,
                             const int som_size_x,                  /* 2 */
                             const int som_size_y,                  /* 3 */
                             const int neuron_size,                 /* 4 */
                       //      const int winner_x,                    /* 5 */
                         //    const int winner_y,                    /* 6 */
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

    uint winner_x = winner_idx[0];
    uint winner_y = winner_idx[1];

    int offset_x = max(0, winner_x - radius);
    int offset_y = max(0, winner_y - radius);

    int width = min(offset_x + 2 * radius + 1, som_size_x) - ofset_x;
    int height = min(offset_y + 2 * radius + 1, som_size_y) - ofset_y;

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
        som_data[global_idx] += (input_vector[local_id_x] - som_data[global_idx]) * weight;
    }
}
