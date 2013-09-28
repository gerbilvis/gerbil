#include "ocl_som2d.h"

#include "iostream"

#include <CL/cl.hpp>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>

#include "stopwatch.h"

//#define TIME_MEASURE

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


OCL_SOM2d::OCL_SOM2d(const vole::EdgeDetectionConfig &conf,
                     const multi_img &data,
                     std::vector<multi_img_base::BandDesc> meta)
    : SOM2d(conf, data, meta),
      d_data(conf.sidelength,
             conf.type == vole::SOM_SQUARE ? conf.sidelength : 1, data.size())
{    
}

OCL_SOM2d::OCL_SOM2d(const vole::EdgeDetectionConfig &conf,
                     int dimension,
                     std::vector<multi_img_base::BandDesc> meta)
    : SOM2d(conf, dimension, meta),
      d_data(conf.sidelength,
             conf.type == vole::SOM_SQUARE ? conf.sidelength : 1, dimension)
{
    initOpenCL();
    initParamters();
    initLocalMemDims();
    initRanges();
    initDeviceBuffers();
    //uploadDataToDevice();
    setKernelParams();
}

void OCL_SOM2d::initOpenCL()
{
    init_opencl(d_context, d_queue);

    std::cout << "ocl som2d hello world!" << std::endl;
    std::string source = read_source("kernels/som2d.cl");

#ifdef DEBUG_MODE
    program = build_cl_program(d_context, source, "-DDEBUG_MODE");
#else
    program = build_cl_program(d_context, source);
#endif

    // Make kernels
    find_nearest_kernel = cl::Kernel(program, "find_nearest_neuron");
    global_min_kernel = cl::Kernel(program, "find_global_min");
    update_kernel = cl::Kernel(program, "generic_update");
}

void OCL_SOM2d::initParamters()
{
    int som_size_x = get2dWidth();
    int som_size_y = get2dHeight();

    preferred_max_block_size = 512;

    const int possible_xy_sizes[] = {1, 2, 4, 8, 16};

    int kernel_xy_dim = 1;

    for(int i = 0; i < 5; ++i)
    {
        int xy_squared = possible_xy_sizes[i] * possible_xy_sizes[i];

        if(preferred_max_block_size >= xy_squared * dim)
        {
            kernel_xy_dim = possible_xy_sizes[i];
        }
        else
        {
            break;
        }
    }
    // TODO: ONLY SQUARE SOM AT THIS MOMENT (NO SUPPORT FOR 1D)
    kernel_size_x = kernel_xy_dim;
    kernel_size_y = kernel_xy_dim;

    reduction_kernel_global_x = (som_size_x + kernel_size_x - 1)
                                    / kernel_size_x;
    reduction_kernel_global_y = (som_size_y + kernel_size_y - 1)
                                    / kernel_size_y;

    reduction_kernel_total = reduction_kernel_global_x
                                 * reduction_kernel_global_y;

    kernel_global_x = reduction_kernel_global_x * kernel_size_x;
    kernel_global_y = reduction_kernel_global_y * kernel_size_y;

//#ifdef DEBUG_MODE

    std::cout << "kernel_xy_dim: " << kernel_xy_dim << std::endl;
    std::cout << "reduction_kernel_total: " << reduction_kernel_total
              << std::endl;
    std::cout << "kernel_global_x: " << kernel_global_x << std::endl;
    std::cout << "kernel_global_y: " << kernel_global_y << std::endl;
    std::cout << "neuron_size: " << dim << std::endl;
//#endif
}


void OCL_SOM2d::initLocalMemDims()
{
    // for find_nearest_kernel
    local_mem_subsom = cl::__local(sizeof(float)
                                   * kernel_size_x
                                   * kernel_size_y
                                   * dim);

    local_mem_reduction = cl::__local(sizeof(int)
                                      * kernel_size_x
                                      * kernel_size_y);

    // for global_min_kernel
    local_mem_reduction_global = cl::__local(sizeof(int)
                                             * reduction_kernel_total);

    // for update kernel
    local_mem_neighbourhood = cl::__local(sizeof(float)
                                          * kernel_size_x
                                          * kernel_size_y);

    local_mem_input_vec = cl::__local(sizeof(float) * dim);
}

void OCL_SOM2d::initRanges()
{
    find_nearest_global = cl::NDRange(kernel_global_x, kernel_global_y, dim);
    find_nearest_local = cl::NDRange(kernel_size_x, kernel_size_y, dim);

    // assumption that reduction_kernel_total is not very big...
    global_min_global = cl::NDRange(reduction_kernel_total);
    global_min_local = cl::NDRange(reduction_kernel_total);

    update_global = cl::NDRange(kernel_global_x, kernel_global_y, dim);
    update_local = cl::NDRange(kernel_size_x, kernel_size_y, dim);
}

void OCL_SOM2d::initDeviceBuffers()
{
    int som_size_x = get2dWidth();
    int som_size_y = get2dHeight();

    d_som = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                       d_data.size * sizeof(float));


    // DRAWBACK of current implementation - only 1 input vect allocated at once
    input_vectors = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                               dim * sizeof(float));

    out_min_indexes = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                                 reduction_kernel_total * sizeof(int));
    out_min_values = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                                reduction_kernel_total * sizeof(float));

    global_min_idx = cl::Buffer(d_context, CL_MEM_READ_WRITE, sizeof(int));

#ifdef DEBUG_MODE
    neighbourhood_verify = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                                      sizeof(float) * som_size_x * som_size_y);
#endif
}

void OCL_SOM2d::setKernelParams()
{
    // Set arguments to kernel
    find_nearest_kernel.setArg(0, d_som);
    find_nearest_kernel.setArg(1, input_vectors);
    find_nearest_kernel.setArg(2, out_min_indexes);
    find_nearest_kernel.setArg(3, out_min_values);
    find_nearest_kernel.setArg(4, get2dWidth());
    find_nearest_kernel.setArg(5, get2dHeight());
    find_nearest_kernel.setArg(6, dim);
    find_nearest_kernel.setArg(7, 0);
    find_nearest_kernel.setArg(8, dim);
    find_nearest_kernel.setArg(9, local_mem_subsom);
    find_nearest_kernel.setArg(10, local_mem_reduction);

    global_min_kernel.setArg(0, out_min_indexes);
    global_min_kernel.setArg(1, out_min_values);
    global_min_kernel.setArg(2, global_min_idx);
    global_min_kernel.setArg(3, local_mem_reduction_global);
    global_min_kernel.setArg(4, reduction_kernel_total);

    update_kernel.setArg(0, d_som);
    update_kernel.setArg(1, input_vectors);
    update_kernel.setArg(2, global_min_idx);
#ifdef DEBUG_MODE
    update_kernel.setArg(3, neighbourhood_verify);
    update_kernel.setArg(4, get2dWidth());
    update_kernel.setArg(5, get2dHeight());
    update_kernel.setArg(6, dim);
    update_kernel.setArg(7, 0);
    update_kernel.setArg(8, dim);
    //update_kernel.setArg(9, sigma_square);
    //update_kernel.setArg(10, learning_rate);
    update_kernel.setArg(11, local_mem_neighbourhood);
    update_kernel.setArg(12, local_mem_input_vec);
#else
    update_kernel.setArg(3, get2dWidth());
    update_kernel.setArg(4, get2dHeight());
    update_kernel.setArg(5, dim);
    update_kernel.setArg(6, 0);
    update_kernel.setArg(7, dim);
    //update_kernel.setArg(8, sigma_square);
    //update_kernel.setArg(9, learning_rate);
    update_kernel.setArg(10, local_mem_neighbourhood);
    update_kernel.setArg(11, local_mem_input_vec);
#endif
}

//it can be more efficient...
void OCL_SOM2d::uploadDataToDevice()
{
    int width = get2dWidth();
    int height = get2dHeight();

    int slice_size = width * height;

    std::cout << "slice_size: " << slice_size << std::endl;

    for(int i = 0; i < neurons.size(); ++i)
    {
        Row& row = neurons[i];

        for(int j = 0; j < row.size(); ++j)
        {
            Neuron& neuron = row[j];

            for(int k = 0; k < neuron.size(); ++k)
            {
                float* ptr = d_data.data + k * slice_size + i * width + j;
                //*ptr = 0;//neuron[k];
                float n_val = neuron[k];
                *ptr = n_val;
            }
        }
    }

    d_queue.enqueueWriteBuffer(d_som, CL_TRUE, 0, d_data.size * sizeof(float),
                               d_data.data);
}

void OCL_SOM2d::downloadDataFromDevice()
{
    d_queue.enqueueReadBuffer(d_som, CL_TRUE, 0, d_data.size * sizeof(float),
                              d_data.data);

    int width = get2dWidth();
    int height = get2dHeight();

    int slice_size = width * height;

    for(int i = 0; i < neurons.size(); ++i)
    {
        Row& row = neurons[i];

        for(int j = 0; j < row.size(); ++j)
        {
            Neuron& neuron = row[j];

            for(int k = 0; k < neuron.size(); ++k)
            {
                float* ptr = d_data.data + k * slice_size + i * width + j;
                neuron[k] = *ptr;
            }
        }
    }
}


void OCL_SOM2d::notifyTrainingStart()
{
    uploadDataToDevice();
}

void OCL_SOM2d::notifyTrainingEnd()
{
    downloadDataFromDevice();
}

SOM::iterator OCL_SOM2d::identifyWinnerNeuron(const multi_img::Pixel &inputVec)
{

#ifdef TIME_MEASURE
    vole::Stopwatch running_time("Identify winner time");
#endif

//    return SOM2d::identifyWinnerNeuron(inputVec);

//    input_vectors = cl::Buffer(d_context, CL_MEM_READ_WRITE,
//                               dim * sizeof(float));
    float* vec_ptr = (float*)(&(inputVec[0]));

#ifdef DEBUG_MODE
    for(int i = 0; i < dim; ++i)
    {
        std::cout << "vec " << i << ": " << vec_ptr[i] << std::endl;
        //vec_ptr[i] = 1;
    }
#endif

    d_queue.enqueueWriteBuffer(input_vectors, CL_TRUE, 0, dim * sizeof(float),
                               vec_ptr);

    int global_min = 0;

    try
    {
        d_queue.enqueueNDRangeKernel(find_nearest_kernel, cl::NullRange,
                                     find_nearest_global, find_nearest_local);

#ifdef DEBUG_MODE
        int* host_out_min_indexes = new int[reduction_kernel_total];
        float* host_out_min_values = new float[reduction_kernel_total];

        d_queue.enqueueReadBuffer(out_min_indexes, CL_TRUE, 0,
                                  reduction_kernel_total * sizeof(int),
                                  host_out_min_indexes);

        d_queue.enqueueReadBuffer(out_min_values, CL_TRUE, 0,
                                  reduction_kernel_total * sizeof(float),
                                  host_out_min_values);

        int som_size_x = get2dWidth();
        //int som_size_y = get2dHeight();

        for(int i = 0; i < reduction_kernel_total; ++i)
        {
            int x = host_out_min_indexes[i] >> 16;
            int y = host_out_min_indexes[i] & 0xFFFF;

            int idx = y * som_size_x + x;

            std::cout << i << ": (" << x << ", " << y << ")"
                      << " = " << host_out_min_values[i] << std::endl;
        }

        delete[] host_out_min_indexes;
        delete[] host_out_min_values;
#endif


        d_queue.enqueueNDRangeKernel(global_min_kernel, cl::NullRange,
                                     global_min_global, global_min_local);


        d_queue.enqueueReadBuffer(global_min_idx, CL_TRUE, 0,
                                  sizeof(int), &global_min);
    }
    catch(cl::Error error)
    {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    }

    int winner_x = global_min >> 16;
    int winner_y = global_min & 0xFFFF;

#ifdef DEBUG_MODE
    for(int i = 0; i < dim; ++i)
    {
        std::cout << neurons[winner_y][winner_x][i] << " ";
    }

    std::cout << std::endl;

    int som_size_x = get2dWidth();
    int som_size_y = get2dHeight();
    int slice_size = som_size_x * som_size_y;

    for(int i = 0; i < dim; ++i)
    {
        std::cout << d_data.data[slice_size * i + som_size_x * winner_y + winner_x] << " ";
    }

    std::cout << std::endl;

    std::cout << "global min index: (" << winner_x
              << ", " << winner_y << ")" << std::endl;
#endif


    return SOM::iterator(new Iterator2d(this, winner_x, winner_y));
}

int OCL_SOM2d::updateNeighborhood(iterator &neuron,
                                  const multi_img::Pixel &input,
                                  double sigma, double learnRate)
{

#ifdef TIME_MEASURE
    vole::Stopwatch running_time("Update time");
#endif

    try
    {

#ifdef DEBUG_MODE
        //update_kernel.setArg(0, d_som);
        update_kernel.setArg(9, (float)(sigma*sigma));
        update_kernel.setArg(10, (float)learnRate);
#else
        update_kernel.setArg(8, (float)(sigma*sigma));
        update_kernel.setArg(9, (float)learnRate);
#endif

        d_queue.enqueueNDRangeKernel(update_kernel, cl::NullRange,
                                   update_global, update_local);
    }
    catch(cl::Error error)
    {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    }

 //   downloadDataFromDevice(); // SHOULD BE REMOVED FROM THIS PLACE, INEFFICIENT

#ifdef DEBUG_MODE
    int som_size_x = get2dWidth();
    int som_size_y = get2dHeight();
    int slice_size = som_size_x * som_size_y;

    Iterator2d *it = static_cast<Iterator2d *>(neuron.getBase());
    cv::Point pos = it->getId();

    int winner_x = pos.x;
    int winner_y = pos.y;

    std::cout << "winner after update: " << std::endl;

    for(int i = 0; i < dim; ++i)
    {
        std::cout << d_data.data[slice_size * i + som_size_x * winner_y + winner_x] << " ";
    }

    std::cout << std::endl;
#endif

    return 42;
}
