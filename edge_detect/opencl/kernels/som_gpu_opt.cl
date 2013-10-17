//#define DEBUG
//#define CPU

//#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

//#ifdef DEBUG
//#define assert(x) \
//            if (! (x)) \
//            { \
//                printf((__constant char*)"Assert(%s) failed in line: %d\n", \
//                       (__constant char*)#x, __LINE__); \
//            }
//#else
//        #define assert(X)
//#endif

//#define X_DIM 32 <-- should be passed as a compile-time parameter
//#define SOM_SIZE_X 32
//#define SOM_SIZE_Y 32
//#define SOM_SIZE_Z 32

#define NEURON_SIZE_ROUNDED (X_DIM * 2)


__kernel void calculate_distances(__global const float* som_data,
                                  __global const float* input_vectors,
                                  __global float* output_distances,
                                  const int input_vector_idx,
                                  __local float* rbuff)
{
    const int local_id_x = get_local_id(0);
    const int local_id_y = get_local_id(1);

    const int global_id_y = get_global_id(1);
#ifdef SOM_3D
    const int global_id_z = get_global_id(2);
#endif
    const int group_id_x = get_group_id(0);

#ifdef SOM_3D
    const int global_vector_idx = (SOM_SIZE_X * SOM_SIZE_Y * global_id_z
                                    + global_id_y * SOM_SIZE_X
                                    + group_id_x) * NEURON_SIZE_ROUNDED;
#else
    const int global_vector_idx = (global_id_y * SOM_SIZE_X
                                    + group_id_x) * NEURON_SIZE_ROUNDED;
#endif

    __local volatile float* rbuff_local = rbuff + local_id_y * X_DIM;

    __global const float* input_vector = input_vectors
                                      + NEURON_SIZE_ROUNDED * input_vector_idx;

    if(global_id_y < SOM_SIZE_Y)
    {
        const int global_idx = global_vector_idx + local_id_x;

        /* loading data to local memory in two phases
         * with first step of reduction */
        float diff = som_data[global_idx] - input_vector[local_id_x];
        rbuff_local[local_id_x] = diff * diff;

        diff = som_data[global_idx + X_DIM] - input_vector[local_id_x + X_DIM];
        rbuff_local[local_id_x] += diff * diff;
    }

#ifdef CPU

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint s = X_DIM / 2; s > 0; s >>= 1)
    {
        if (local_id_x < s)
        {
            rbuff_local[local_id_x] += rbuff_local[local_id_x + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#else

    if(X_DIM > 32)
        barrier(CLK_LOCAL_MEM_FENCE);

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

    if(local_id_x == 0 && global_id_y < SOM_SIZE_Y)
    {
#ifdef SOM_3D
        int global_idx = SOM_SIZE_X * SOM_SIZE_Y * global_id_z
                         + global_id_y  * SOM_SIZE_X + group_id_x;
#else
        int global_idx = global_id_y  * SOM_SIZE_X + group_id_x;
#endif
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
            ulong coords = (((global_id % SOM_SIZE_X) << 16)
                           | (global_id / SOM_SIZE_X));
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
#ifdef SOM_3D
                int slice = SOM_SIZE_X * SOM_SIZE_Y;
                int z = idx / slice;
                idx = idx % slice;

                min_indexes[0] = idx % SOM_SIZE_X;
                min_indexes[1] = idx / SOM_SIZE_X;
                min_indexes[2] = z;
#else
                min_indexes[0] = idx % SOM_SIZE_X;
                min_indexes[1] = idx / SOM_SIZE_X;
#endif

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
                             __global const float* input_vectors,
                             __global const int* winner_idx,
                             int input_vector_idx,
                             const int radius,
                             const float sigma_square,
                             const float learning_rate,
                             __local float* weights)
{
    const int local_id_x = get_local_id(0);
    const int local_id_y = get_local_id(1);

    const int global_id_y = get_global_id(1);
#ifdef SOM_3D
    const int global_id_z = get_global_id(2);
#endif

    const int group_id_x = get_group_id(0);

/*    const int winner_x = winner_idx[1] >> 16;
    const int winner_y = winner_idx[1] & 0xFFFF;*/
    const int winner_x = winner_idx[0];
    const int winner_y = winner_idx[1];
#ifdef SOM_3D
    const int winner_z = winner_idx[2];
#endif

    const int offset_x = max(0, winner_x - radius);
    const int offset_y = max(0, winner_y - radius);
#ifdef SOM_3D
    const int offset_z = max(0, winner_z - radius);
#endif

    const int width = min(offset_x + 2 * radius + 1, SOM_SIZE_X) - offset_x;
    const int height = min(offset_y + 2 * radius + 1, SOM_SIZE_Y) - offset_y;
#ifdef SOM_3D
    const int depth = min(offset_z + 2 * radius + 1, SOM_SIZE_Z) - offset_z;
#endif

    const int global_translated_x = group_id_x + offset_x;
    const int global_translated_y = global_id_y + offset_y;
#ifdef SOM_3D
    const int global_translated_z = global_id_z + offset_z;
#endif

#ifdef SOM_3D
    const int global_vector_idx = (SOM_SIZE_X * SOM_SIZE_Y * global_translated_z
                                  + global_translated_y * SOM_SIZE_X
                                  + global_translated_x) * NEURON_SIZE_ROUNDED;
#else
    const int global_vector_idx = (global_translated_y * SOM_SIZE_X
                                   + global_translated_x) * NEURON_SIZE_ROUNDED;
#endif

    __global const float* input_vector = input_vectors
                                      + NEURON_SIZE_ROUNDED * input_vector_idx;


#ifdef SOM_3D
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

        weights[local_id_y] = learning_rate * fake_gauss;
    }
#else
    if(get_local_id(0) == 0 && group_id_x < width && global_id_y < height)
    {
        float diff_x = global_translated_x - winner_x;
        float diff_y = global_translated_y - winner_y;

        float distance_squared = diff_x * diff_x + diff_y * diff_y;
        float fake_gauss = native_exp(-(distance_squared)
                                      / (2.f * sigma_square));

        weights[local_id_y] = learning_rate * fake_gauss;
    }
#endif

#ifdef CPU
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    if(X_DIM > 32)
        barrier(CLK_LOCAL_MEM_FENCE);
#endif

    float weight = weights[local_id_y];

#ifdef SOM_3D
    if(group_id_x < width && global_id_y < height
        && global_id_z < depth && weight >= 0.01f)
#else
    if(group_id_x < width && global_id_y < height && weight >= 0.01f)
#endif
    {
        int global_idx = global_vector_idx + local_id_x;
        som_data[global_idx] += (input_vector[local_id_x]
                                - som_data[global_idx]) * weight;

        global_idx += X_DIM;
        som_data[global_idx] += (input_vector[local_id_x + X_DIM]
                                - som_data[global_idx]) * weight;
    }
}

#define BLOCK_SIZE 16
#define VECTOR_SIZE NEURON_SIZE_ROUNDED

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


////Computes all pairs of distances between Q and X.
//__global__ void dist1Kernel(const matrix Q, unint qStart, const matrix X, unint xStart, matrix D){
//  unint c, i, j;

//  unint qB = blockIdx.y*BLOCK_SIZE + qStart;
//  unint q  = threadIdx.y;
//  unint xB = blockIdx.x*BLOCK_SIZE + xStart;
//  unint x = threadIdx.x;

//  real ans=0;

//  //This thread is responsible for computing the dist between Q[qB+q] and X[xB+x]

//  __shared__ real Qs[BLOCK_SIZE][BLOCK_SIZE];
//  __shared__ real Xs[BLOCK_SIZE][BLOCK_SIZE];


//  for(i=0 ; i<Q.pc/BLOCK_SIZE ; i++){
//    c=i*BLOCK_SIZE; //current col block

//    Qs[x][q] = Q.mat[ IDX(qB+q, c+x, Q.ld) ];
//    Xs[x][q] = X.mat[ IDX(xB+q, c+x, X.ld) ];

//    __syncthreads();

//    for(j=0 ; j<BLOCK_SIZE ; j++)
//      ans += DIST( Qs[j][q], Xs[j][x] );

//    __syncthreads();
//  }

//  D.mat[ IDX( qB+q, xB+x, D.ld ) ] = ans;

//}
