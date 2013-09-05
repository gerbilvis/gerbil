#include "ocl_som2d.h"

#include "iostream"

#include <CL/cl.hpp>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "ocl_utils.h"
som_data::som_data(int x, int y, int z)
    : x(x), y(y), z(z)
{
    if(x == 0 || y == 0 || z == 0)
    {
        size = 0;
        data = 0;
    }
    else
    {
        size = x * y * z;
        data = new float[size];
    }
}

som_data::~som_data()
{
    if(data)
        delete[] data;
}


void init_som(som_data& som)
{
    for(int z = 0; z < som.z; ++z)
    {
        for(int y = 0; y < som.y; ++y)
        {
            for(int x = 0; x < som.x; ++x)
            {
                int idx = som.x * som.y * z + som.x * y + x;

                som.data[idx] = 1;//x + y;
            }
        }
    }
}


void ocl_som2d_test_impl()
{


    std::cout << "ocl som2d hello world!" << std::endl;
    std::string source = read_source("kernels/som2d.cl");

//    std::cout << source << std::endl;

    try {

        cl::Context context;
        cl::CommandQueue queue;

        init_opencl(context, queue);
        cl::Program program = build_cl_program(context, source);

        // Make kernel
        cl::Kernel kernel(program, "find_nearest_neuron");

        const int neuron_size = 8;
        const int som_size_x = 32;
        const int som_size_y = 32;
        som_data som(som_size_x, som_size_y, neuron_size);

        init_som(som);

        som.data[32*32-1] = 0.6;
//        som.data[128] = 4;
//        som.data[133] = 0.9;

        int som_size = som.size;

        const int input_vectors_size = neuron_size * 100;
        float input_vectors_host[input_vectors_size];

        for(int i = 0; i < input_vectors_size; ++i){
            input_vectors_host[i] = 0.f;
        }

        // Create memory buffers
        cl::Buffer som_data_buff(context, CL_MEM_READ_WRITE,
                            som_size * sizeof(float));

        cl::Buffer input_vectors(context, CL_MEM_READ_WRITE,
                                 input_vectors_size * sizeof(float));

        cl::Buffer out_min_indexes(context, CL_MEM_READ_WRITE, 16 * sizeof(int));
        cl::Buffer out_min_values(context, CL_MEM_READ_WRITE, 16 * sizeof(float));

        // Copy lists A and B to the memory buffers
        queue.enqueueWriteBuffer(som_data_buff, CL_TRUE, 0,
                                 som_size * sizeof(float), som.data);
        queue.enqueueWriteBuffer(input_vectors, CL_TRUE, 0,
                                 input_vectors_size * sizeof(float),
                                 input_vectors_host);

        const int kernel_size_x = 8;
        const int kernel_size_y = 8;

        cl::LocalSpaceArg local_mem_subsom = cl::__local(sizeof(float)
                                                             * kernel_size_x
                                                             * kernel_size_y
                                                             * neuron_size);

        cl::LocalSpaceArg local_mem_reduction = cl::__local(sizeof(int)
                                                             * kernel_size_x
                                                             * kernel_size_y);

        // Run the kernel on specific ND range
        cl::NDRange global(som_size_x, som_size_y, neuron_size);
        cl::NDRange local(8, 8, neuron_size);

        // Set arguments to kernel
        kernel.setArg(0, som_data_buff);
        kernel.setArg(1, input_vectors);
        kernel.setArg(2, out_min_indexes);
        kernel.setArg(3, out_min_values);
        kernel.setArg(4, som.x);
        kernel.setArg(5, som.y);
        kernel.setArg(6, som.z);
        kernel.setArg(7, 0);
        kernel.setArg(8, neuron_size);
        kernel.setArg(9, local_mem_subsom);
        kernel.setArg(10, local_mem_reduction);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

        int host_out_min_indexes[16];
        float host_out_min_values[16];

        queue.enqueueReadBuffer(out_min_indexes, CL_TRUE, 0,
                                16 * sizeof(int), host_out_min_indexes);

        queue.enqueueReadBuffer(out_min_values, CL_TRUE, 0,
                                16 * sizeof(float), host_out_min_values);

        for(int i = 0; i < 16; ++i)
        {
            int x = host_out_min_indexes[i] >> 16;
            int y = host_out_min_indexes[i] & 0xFFFF;

            int idx = y * 32 + x;

            std::cout << i << ": (" << x << ", " << y << ")"
                      << " = " << host_out_min_values[i] << std::endl;
        }

        // FINIDNG GLOBAL MINIMUM

        cl::Kernel find_global_min_kernel(program, "find_global_min");

        cl::Buffer global_min_idx(context, CL_MEM_READ_WRITE, sizeof(int));

        local_mem_reduction = cl::__local(sizeof(int) * 16);

        find_global_min_kernel.setArg(0, out_min_indexes);
        find_global_min_kernel.setArg(1, out_min_values);
        find_global_min_kernel.setArg(2, global_min_idx);
        find_global_min_kernel.setArg(3, local_mem_reduction);
        find_global_min_kernel.setArg(4, 16);

        global = cl::NDRange(16);
        local = cl::NDRange(16);

        queue.enqueueNDRangeKernel(find_global_min_kernel,
                                   cl::NullRange,global, local);

        int global_min;

        queue.enqueueReadBuffer(global_min_idx, CL_TRUE, 0,
                                sizeof(int), &global_min);

        int global_min_x = global_min >> 16;
        int global_min_y = global_min & 0xFFFF;

        std::cout << "global min index: (" << global_min_x
                  << ", " << global_min_y << ")" << std::endl;

        // UPDATING SOM
        cl::Kernel update_kernel(program, "generic_update");

        float sigma_square = 2.f;
        float learning_rate = 1.f;

        cl::LocalSpaceArg local_mem_neighbourhood = cl::__local(sizeof(unsigned char)
                                                             * kernel_size_x
                                                             * kernel_size_y);
        cl::LocalSpaceArg local_mem_input_vec = cl::__local(sizeof(float)
                                                             * neuron_size);

        cl::Buffer neighbourhood_verify(context, CL_MEM_READ_WRITE,
                                        sizeof(unsigned char) * som_size_x * som_size_y);

        update_kernel.setArg(0, som_data_buff);
        update_kernel.setArg(1, input_vectors);
        update_kernel.setArg(2, global_min_idx);
        update_kernel.setArg(3, neighbourhood_verify);
        update_kernel.setArg(4, som.x);
        update_kernel.setArg(5, som.y);
        update_kernel.setArg(6, som.z);
        update_kernel.setArg(7, 0);
        update_kernel.setArg(8, neuron_size);
        update_kernel.setArg(9, sigma_square);
        update_kernel.setArg(10, learning_rate);
        update_kernel.setArg(11, local_mem_neighbourhood);
        update_kernel.setArg(12, local_mem_input_vec);

        global = cl::NDRange(som_size_x, som_size_y, neuron_size);
        local = cl::NDRange(8, 8, neuron_size);

        queue.enqueueNDRangeKernel(update_kernel, cl::NullRange, global, local);

        unsigned char nighbour_verify_host[som_size_x * som_size_y];

        queue.enqueueReadBuffer(neighbourhood_verify, CL_TRUE, 0,
                                sizeof(unsigned char) * som_size_x * som_size_y,
                                nighbour_verify_host);

        for(int j = 0; j < som_size_y; ++j)
        {
            unsigned char* line_ptr = nighbour_verify_host + som_size_x * j;

            for(int i = 0; i < som_size_x; ++i)
            {
                std::cout << (int)line_ptr[i];
            }

            std::cout << std::endl;
        }


    }
    catch(cl::Error error)
    {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    }
}

void ocl_som2d_test()
{
    for(int i = 0; i < 1; ++i)
        ocl_som2d_test_impl();
}

