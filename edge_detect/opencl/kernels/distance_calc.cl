#define BLOCK_SIZE 16
//#define VECTOR_SIZE

__kernel void calculate_all_distances(__global const float* input_1,
                                      __global const float* input_2,
                                      int input_length_1,
                                      int input_length_2,
                                      __global float* distances)
{
    int block_size_x = get_local_size(0);

    int local_id_x = get_local_id(0);
    int local_id_y = get_local_id(1);

    int global_id_x = get_global_id(0);
    int global_id_y = get_global_id(1);

    int group_id_x = get_group_id(0);
    int group_id_y = get_group_id(1);

    __local float s_input_1[BLOCK_SIZE][BLOCK_SIZE];
    __local float s_input_2[BLOCK_SIZE][BLOCK_SIZE];

    float distance = 0.f;


    // VECTOR_SIZE must be multiply of BLOCK_SIZE !
    for(int i = local_id_x; i < VECTOR_SIZE; i += BLOCK_SIZE)
    {
        s_input_1[local_id_x][local_id_y]
            = input_1[global_id_y * VECTOR_SIZE + i];

        s_input_2[local_id_x][local_id_y]
            = input_2[(group_id_x * BLOCK_SIZE + local_id_y) * VECTOR_SIZE + i];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int j = 0; j < BLOCK_SIZE; ++j)
        {
            float diff = s_input_1[j][local_id_y] - s_input_2[j][local_id_x];
            distance += diff * diff;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    distances[global_id_y * input_length_2 + global_id_x] = distance;
}
