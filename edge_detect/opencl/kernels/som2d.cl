__kernel void find_nearest_neuron(__global const float* som_data,
                                  __global const float* input_vectors,
                                  __global int* output_min_indexes,
                                  __global float* output_min_values,
                                  int som_size_x,
                                  int som_size_y,
                                  int som_size_z,
                                  int input_vector_idx,
                                  int input_vector_size,
                                  __local float* som_part,
                                  __local int* reduction_buff)
{
    int local_id_x = get_local_id(0);
    int local_id_y = get_local_id(1);
    int local_id_z = get_local_id(2);

    int global_id_x = get_global_id(0);
    int global_id_y = get_global_id(1);
    int global_id_z = get_global_id(2);

    int group_size_x = get_local_size(0);
    int group_size_y = get_local_size(1);

    int group_id_x = get_group_id(0);
    int group_id_y = get_group_id(1);

    int group_id_linear = get_num_groups(0) * group_id_y + group_id_x;

    int global_idx = som_size_x * som_size_y * global_id_z
                     + som_size_x * global_id_y + global_id_x;

    int local_idx = group_size_x * group_size_y * local_id_z
                     + group_size_x * local_id_y + local_id_x;


    /* loading input vector element to shared memory */

    __global const float* current_in_vec =
                        input_vectors + input_vector_size * input_vector_idx;

    // TO DO: this access memory pattern may be not efficient,
    //        for further consideration
    float vec_elem = current_in_vec[local_id_z];


    /* loading som data to shared memory */
    if(global_id_x < som_size_x && global_id_y < som_size_y)
    {
        float val = som_data[global_idx] - vec_elem;
        som_part[local_idx] = val * val;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    /* calculating sum by reduction */

    int idx2d = group_size_x * local_id_y + local_id_x;
    int step = group_size_x * group_size_y;

    for (unsigned int s = input_vector_size/2; s > 0; s >>= 1)
    {
        if (local_id_z < s)
        {
            som_part[idx2d] += som_part[idx2d + s * step];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(local_id_z == 0)
    {
        reduction_buff[group_size_x * local_id_y + local_id_x]
                                = ((global_id_x << 16) | global_id_y);
                                //= som_size_x * global_id_y + global_id_x;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int limit = group_size_x * group_size_y;

    for (unsigned int s = limit / 2; s>0; s >>= 1) {

        int tid = limit * local_id_z + group_size_x * local_id_y + local_id_x;

        if (tid < s)
        {
            float old = som_part[tid];
            som_part[tid] = fmin(old, som_part[tid + s]);

            if(som_part[tid] != old)
            {
                reduction_buff[tid] = reduction_buff[tid + s];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(local_id_x == 0 && local_id_y == 0 && local_id_z == 0)
    {
//        printf("group id linear: %d, rb: %d, %d\n", group_id_linear, reduction_buff[0],
//                                                som_size_x * global_id_y + global_id_x);
//        printf("som_size_x: %d, global_id_y: %d, global_id_x: %d\n",
//                                                som_size_x, global_id_y, global_id_x);
        output_min_indexes[group_id_linear] = reduction_buff[0];
    }

    if(local_id_x == 0 && local_id_y == 0 && local_id_z == 1)
    {
        output_min_values[group_id_linear] = som_part[0];
    }
}

// poor implementation...
__kernel void find_global_min(__global const int* min_indexes,
                                  __global const float* min_values,
                                  __global int* global_min_index,
                                  __local float* reduction_buff,
                                  int vector_size)
{
    int global_id_x = get_global_id(0);

    if(global_id_x < vector_size)
        reduction_buff[global_id_x] = min_values[global_id_x];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int s = vector_size/2; s > 0; s >>= 1)
    {
        if (global_id_x < s)
        {
            reduction_buff[global_id_x] = fmin(reduction_buff[global_id_x],
                                               reduction_buff[global_id_x + s]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float val = reduction_buff[0];

    if(global_id_x < vector_size)
    {
        if(val == min_values[global_id_x])
            *global_min_index = min_indexes[global_id_x];
    }
}

#define TEST_OUTPUT

__kernel void generic_update(__global float* som_data,
                             __global const float* input_vectors,
                             __global const int* winner_idx,
                             #ifdef TEST_OUTPUT
                             __global float* test_output, // to check neighbourhood detection
                             #endif
                             int som_size_x,
                             int som_size_y,
                             int som_size_z,
                             int input_vector_idx,
                             int input_vector_size,
                             float sigma_square,
                             float learning_rate,
                             __local uchar* neighbourhood,
                             __local float* input_vector)
{
    int local_id_x = get_local_id(0);
    int local_id_y = get_local_id(1);
    int local_id_z = get_local_id(2);

    int global_id_x = get_global_id(0);
    int global_id_y = get_global_id(1);
    int global_id_z = get_global_id(2);

    int group_size_x = get_local_size(0);

    int group_id_x = get_group_id(0);
    int group_id_y = get_group_id(1);

    int winner_x = *winner_idx;
    int winner_y = winner_x & 0xFFFF;
    winner_x >>= 16;

    int local_idx = local_id_y * group_size_x + local_id_x;

    float weight;

    if(global_id_z == 0)
    {
        int diff_x = global_id_x - winner_x;
        int diff_y = global_id_y - winner_y;

        float distance_squared = diff_x * diff_x + diff_y * diff_y;
        float fake_gauss = exp(-(distance_squared)/ (2.f * sigma_square));
        weight = learning_rate * fake_gauss;

        neighbourhood[local_idx] = (weight >= 0.01f);

        #ifdef TEST_OUTPUT
        int global_idx = som_size_x * global_id_y + global_id_x;
        test_output[global_idx] = neighbourhood[local_idx] ? weight : 0;
        #endif
    }

    if(global_id_z == 1 && local_idx < input_vector_size)
    {
        input_vector[local_idx] = input_vectors[input_vector_idx
                                        * input_vector_size + local_idx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uchar do_update = neighbourhood[local_idx];

    if(do_update)
    {
        int global_idx = som_size_x * som_size_y * global_id_z
                         + som_size_x * global_id_y + global_id_x;

        som_data[global_idx] += (som_data[global_idx]
                                    - input_vector[local_id_x]) * weight;
    }
}
