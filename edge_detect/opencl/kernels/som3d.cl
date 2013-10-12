//#define DEBUG
//#define CPU

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

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

//#define X_DIM 32 <-- should be passed as a compile-time parameter

#define NEURON_SIZE_ROUNDED (X_DIM * 2)


__kernel void calculate_distances(__global const float* som_data,
                                  __global const float* input_vector,
                                  __global float* output_distances,
                                  const int som_size_x,
                                  const int som_size_y,
                                  __local float* rbuff)
{
    const int local_id_x = get_local_id(0);
    const int local_id_y = get_local_id(1);
    //const int local_id_z = get_local_id(2);

    const int global_id_y = get_global_id(1);
    const int global_id_z = get_global_id(2);
    const int group_id_x = get_group_id(0);

//    const int global_vector_idx = (global_id_y * som_size_x
//                                    + group_id_x) * NEURON_SIZE_ROUNDED;

    const int global_vector_idx = (som_size_x * som_size_y * global_id_z
                                    + global_id_y * som_size_x
                                    + group_id_x) * NEURON_SIZE_ROUNDED;

    __local volatile float* rbuff_local = rbuff + local_id_y * X_DIM;

    if(global_id_y < som_size_y)
    {
        const int global_idx = global_vector_idx + local_id_x;

        /* loading data to local memory in two phases
         * with first step of reduction */
        float diff = som_data[global_idx] - input_vector[local_id_x];
        rbuff_local[local_id_x] = diff * diff;

        diff = som_data[global_idx + X_DIM] - input_vector[local_id_x + X_DIM];
        rbuff_local[local_id_x] += diff * diff;
    }

    if(X_DIM > 32)
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
#else
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


    if(X_DIM >= 64)
        if(local_id_x < 32)
            rbuff_local[local_id_x] += rbuff_local[local_id_x + 32];
    if(X_DIM >= 32)
        if(local_id_x < 16)
            rbuff_local[local_id_x] += rbuff_local[local_id_x + 16];
    if(X_DIM >= 16)
        if(local_id_x < 8)
            rbuff_local[local_id_x] += rbuff_local[local_id_x + 8];
    if(X_DIM >= 8)
        if(local_id_x < 4)
            rbuff_local[local_id_x] += rbuff_local[local_id_x + 4];
    if(X_DIM >= 4)
        if(local_id_x < 2)
            rbuff_local[local_id_x] += rbuff_local[local_id_x + 2];
    if(X_DIM >= 2)
        if(local_id_x < 1)
            rbuff_local[local_id_x] += rbuff_local[local_id_x + 1];
#endif

  //  barrier(CLK_LOCAL_MEM_FENCE);

    if(local_id_x == 0 && global_id_y < som_size_y)
    {
        //int global_idx = global_id_y  * som_size_x + group_id_x;

        int global_idx = som_size_x * som_size_y * global_id_z
                         + global_id_y  * som_size_x + group_id_x;

        output_distances[global_idx] = rbuff_local[0];
    }
}

__kernel void find_global_first_pass(__global const float* values,
                                     __global float* min_values,
                                     __global int* min_indexes,
                                     __local volatile float* rbuff,
                                     const int vector_size)
{
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);

    const int global_id = get_global_id(0);
    const int group_id = get_group_id(0);

    if(global_id < vector_size)
        rbuff[local_id] = values[global_id];
    else
        rbuff[local_id] = MAXFLOAT;

    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef CPU
    for (int s = group_size / 2; s > 0; s >>= 1)
    {
        if (local_id < s)
        {
            rbuff[local_id] = fmin(rbuff[local_id],
                                   rbuff[local_id + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#else
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
/*
            ulong partial = *((uint*)(&val));
            ulong coords = (((global_id % 32) << 16) | (global_id / 32));
            partial |= coords << 32;

            ulong old_val, old_val_2, prev;
            bool update = false;

            do
            {
                old_val = *((volatile __global ulong*)min_indexes);

                uint to_float = old_val & 0xFFFFFFFF;
                float f_val = *((float*)(&to_float));

                if(val < f_val)
                    update = true;

                old_val_2 = *((volatile __global ulong*)min_indexes);

                if(old_val == old_val_2 && !update)
                    break;

                prev = atom_cmpxchg((volatile __global ulong*)min_indexes,
                                            old_val, partial);

                update = false;
            }
            while(old_val != prev); */
        }
    }
}

__kernel void find_global_min(__global float* min_values,
                              __global int* min_indexes,
                              const int som_size_x,
                              const int som_size_y,
                              const int vector_size,
                              __local volatile float* reduction_buff)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);

    if(global_id < vector_size)
        reduction_buff[local_id] = min_values[global_id];
    else
        reduction_buff[local_id] = MAXFLOAT;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = get_local_size(0) / 2; s > 0; s >>= 1)
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
                int idx = min_indexes[global_id];

                int slice = som_size_x * som_size_y;

                int z = idx / slice;
                idx = idx % slice;

                min_indexes[0] = idx % som_size_x;
                min_indexes[1] = idx / som_size_x;
                min_indexes[2] = z;
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
                             __global const int* winner_idx,
                             const int som_size_x,
                             const int som_size_y,
                             const int som_size_z,
                             const int radius,
                             const float sigma_square,
                             const float learning_rate,
                             __local float* weights)
{
    const int local_id_x = get_local_id(0);
    const int local_id_y = get_local_id(1);

    const int global_id_y = get_global_id(1);
    const int global_id_z = get_global_id(2);

    const int group_id_x = get_group_id(0);

/*    const int winner_x = winner_idx[1] >> 16;
    const int winner_y = winner_idx[1] & 0xFFFF;*/
    const int winner_x = winner_idx[0];
    const int winner_y = winner_idx[1];
    const int winner_z = winner_idx[2];

    const int offset_x = max(0, winner_x - radius);
    const int offset_y = max(0, winner_y - radius);
    const int offset_z = max(0, winner_z - radius);

    const int width = min(offset_x + 2 * radius + 1, som_size_x) - offset_x;
    const int height = min(offset_y + 2 * radius + 1, som_size_y) - offset_y;
    const int depth = min(offset_z + 2 * radius + 1, som_size_z) - offset_z;

    const int global_translated_x = group_id_x + offset_x;
    const int global_translated_y = global_id_y + offset_y;
    const int global_translated_z = global_id_z + offset_z;

//    const int global_vector_idx = (global_translated_y * som_size_x
//                                   + global_translated_x) * NEURON_SIZE_ROUNDED;

    const int global_vector_idx = (som_size_x * som_size_y * global_translated_z
                                   + global_translated_y * som_size_x
                                   + global_translated_x) * NEURON_SIZE_ROUNDED;

//    const int global_vector_idx = (som_size_x * som_size_y * global_id_z
//                                    + global_id_y * som_size_x
//                                    + group_id_x) * NEURON_SIZE_ROUNDED;

    if(get_local_id(0) == 0 && group_id_x < width
                            && global_id_y < height
                            && global_id_z < depth)
    {
        float diff_x = global_translated_x - winner_x;
        float diff_y = global_translated_y - winner_y;
        float diff_z = global_translated_z - winner_z;

        float distance_squared = diff_x * diff_x
                                 + diff_y * diff_y
                                 + diff_z * diff_z;

        float fake_gauss = native_exp(-(distance_squared)
                                      / (2.f * sigma_square));

      // if(distance_squared == 0.f)

       // if((diff_x == 0.f || diff_y == 0.f) && diff_z == 0.f)
            weights[local_id_y] = learning_rate * fake_gauss;
      //  else
       //     weights[local_id_y] = 0;//learning_rate * fake_gauss;
    }

  //  if(X_DIM > 32)
        barrier(CLK_LOCAL_MEM_FENCE);

    float weight = weights[local_id_y];

    if(group_id_x < width && global_id_y < height
        && global_id_z < depth && weight >= 0.01f)
    {
        int global_idx = global_vector_idx + local_id_x;
        som_data[global_idx] += (input_vector[local_id_x]
                                - som_data[global_idx]) * weight;

        global_idx += X_DIM;
        som_data[global_idx] += (input_vector[local_id_x + X_DIM]
                                - som_data[global_idx]) * weight;
    }
}
