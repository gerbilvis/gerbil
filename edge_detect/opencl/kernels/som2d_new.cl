#define DEBUG
#define CPU

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

#ifdef CPU
    #define FORCE_BARRIER_ON_CPU (barrier(CLK_LOCAL_MEM_FENCE));
#elif
    #define FORCE_BARRIER_ON_CPU
#endif


//#define X_DIM 32 <-- should be passed as a compile-time parameter

#define NEURON_SIZE_ROUNDED (X_DIM * 2)


__kernel void calculate_distances(__global const float* som_data,
                                  __global const float* input_vector,
                                  __global float* output_distances,
                                  const int som_size_x,
                                  const int som_size_y,
                                  __local float* rbuff)
{
    const size_t local_id_x = get_local_id(0);
    const size_t local_id_y = get_local_id(1);

    const size_t global_id_y = get_global_id(1);
    const size_t group_id_x = get_group_id(0);

    const int global_vector_idx = (global_id_y * som_size_x
                                    + group_id_x) * NEURON_SIZE_ROUNDED;

    __local volatile float* rbuff_local = rbuff
                                            + local_id_y * NEURON_SIZE_ROUNDED;

    if(global_id_y < som_size_y)
    {
        const size_t global_idx = global_vector_idx + local_id_x;

        /* loading data to local memory in two phases
         * with first step of reduction */
        float diff = som_data[global_idx] - input_vector[local_id_x];
        rbuff_local[local_id_x] = diff * diff;

        diff = som_data[global_idx + X_DIM] - input_vector[local_id_x + X_DIM];
        rbuff_local[local_id_x] += diff * diff;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef CPU
    for (uint s = X_DIM / 2; s > 0; s >>= 1)
    {
        if (local_id_x < s)
        {
            rbuff_local[local_id_x] += rbuff_local[local_id_x + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#elif
    if(X_DIM >= 512)
    {
        if (local_id_x < 256)
        {
            rbuff_local[local_id_x] += rbuff_local[local_id_x + 256];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(X_DIM >= 256)
    {
        if (local_id_x < 128)
        {
            rbuff_local[local_id_x] += rbuff_local[local_id_x + 128];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(X_DIM >= 128)
    {
        if (local_id_x < 64)
        {
            rbuff_local[local_id_x] += rbuff_local[local_id_x + 64];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }


    if (local_id_x < 32)
    {
        if(X_DIM >= 64)
            rbuff_local[local_id_x] += rbuff_local[local_id_x + 32];
        if(X_DIM >= 32)
            rbuff_local[local_id_x] += rbuff_local[local_id_x + 16];
        if(X_DIM >= 16)
            rbuff_local[local_id_x] += rbuff_local[local_id_x + 8];
        if(X_DIM >= 8)
            rbuff_local[local_id_x] += rbuff_local[local_id_x + 4];
        if(X_DIM >= 4)
            rbuff_local[local_id_x] += rbuff_local[local_id_x + 2];
        if(X_DIM >= 2)
            rbuff_local[local_id_x] += rbuff_local[local_id_x + 1];
    }
#endif
    if(local_id_x == 0 && global_id_y < som_size_y)
    {
        int global_idx = global_id_y  * som_size_x + group_id_x;
        output_distances[global_idx] = rbuff_local[0];
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

#ifdef CPU
    for (uint s = group_size / 2; s > 0; s >>= 1)
    {
        if (local_id < s)
        {
            rbuff[local_id] = fmin(rbuff[local_id],
                                   rbuff[local_id + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#elif
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
#endif

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


__kernel void update_network(__global float* som_data,
                             __global const float* input_vector,
                             __global const uint* winner_idx,
                             const uint som_size_x,
                             const uint som_size_y,
                             const uint radius,
                             const float sigma_square,
                             const float learning_rate,
                             __local float* weights)
{
    uint local_id_x = get_local_id(0);
    uint local_id_y = get_local_id(1);

    uint global_id_x = get_global_id(0);
    uint global_id_y = get_global_id(1);

    uint group_id_x = get_group_id(0);

    uint winner_x = winner_idx[0];
    uint winner_y = winner_idx[1];

    uint offset_x = (uint)max(0, (int)winner_x - (int)radius);
    uint offset_y = (uint)max(0, (int)winner_y - (int)radius);

    uint width = min(offset_x + 2 * radius + 1, som_size_x) - offset_x;
    uint height = min(offset_y + 2 * radius + 1, som_size_y) - offset_y;

    int global_translated_x = group_id_x + offset_x;
    int global_translated_y = global_id_y + offset_y;

    int global_vector_idx = global_translated_y * som_size_x * NEURON_SIZE_ROUNDED
                            + global_translated_x * NEURON_SIZE_ROUNDED;


    if(local_id_x == 0 && group_id_x < width && global_id_y < height)
    {
        int diff_x = global_translated_x - winner_x;
        int diff_y = global_translated_y - winner_y;

        float distance_squared = diff_x * diff_x + diff_y * diff_y;
        float fake_gauss = native_exp(-(distance_squared)
                                      / (2.f * sigma_square));

        weights[local_id_y] = learning_rate * fake_gauss;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float weight = weights[local_id_y];

    if(group_id_x < width && global_id_y < height
        && local_id_x < (X_DIM * 2) && weight >= 0.01f)
    {
        int global_idx = global_vector_idx + local_id_x;
        som_data[global_idx] += (input_vector[local_id_x]
                                - som_data[global_idx]) * weight;
    }
}
