#include "ocl_som2d_new.h"

#include "iostream"

#include <CL/cl.hpp>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>

#include "stopwatch.h"

#include "ocl_utils.h"

//#define TIME_MEASURE

som_data_new::som_data_new(int x, int y, int neuron_size)
    : x(x), y(y), neuron_size(neuron_size)
{
    if(x == 0 || y == 0 || neuron_size == 0)
    {
        size = 0;
        data = 0;
    }
    else
    {
        size = x * y * neuron_size;
        data = new float[size];
    }
}

som_data_new::~som_data_new()
{
    if(data)
        delete[] data;
}


OCL_SOM2d_new::OCL_SOM2d_new(const vole::EdgeDetectionConfig &conf,
                     const multi_img &data,
                     std::vector<multi_img_base::BandDesc> meta)
    : SOM2d(conf, data, meta),
      d_data(conf.sidelength,
             conf.type == vole::SOM_SQUARE ? conf.sidelength : 1, data.size())
{    
}

OCL_SOM2d_new::OCL_SOM2d_new(const vole::EdgeDetectionConfig &conf,
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

OCL_SOM2d_new::~OCL_SOM2d_new()
{
}

void OCL_SOM2d_new::initOpenCL()
{
    init_opencl(d_context, d_queue);

    std::cout << "ocl som2d_new hello world!" << std::endl;
    std::string source = read_source("kernels/som2d_new.cl");

#ifdef DEBUG_MODE
    program = build_cl_program(d_context, source, "-DDEBUG_MODE -Werror");
#else
    program = build_cl_program(d_context, source);
#endif

    // Make kernels
    calculate_distances_kernel = cl::Kernel(program, "calculate_distances");
    global_min_kernel = cl::Kernel(program, "find_global_min");
    update_kernel = cl::Kernel(program, "update_network");
}

void OCL_SOM2d_new::initParamters()
{
    int som_size_x = get2dWidth();
    int som_size_y = get2dHeight();

    const int pow2[] = {1, 2, 4, 8, 16, 32, 64};

    int k_size_x = 1;

    for(int i = 0; i < 7; ++i)
    {
        k_size_x = pow2[i];

        if(dim <= pow2[i])
            break;
    }

    kernel_size_x = k_size_x;
    kernel_size_y = 512 / k_size_x;

    kernel_global_x = k_size_x * som_size_x * (( som_size_y + kernel_size_y - 1) / kernel_size_y);
    kernel_global_y = ((som_size_y + kernel_size_y - 1) / kernel_size_y) * kernel_size_y;

    reduction_kernel_local_x = 512;
    reduction_kernel_global_x = ((som_size_x * som_size_y + reduction_kernel_local_x - 1)
                                 / reduction_kernel_local_x )* reduction_kernel_local_x;


    reduced_elems_count = ((reduction_kernel_global_x + reduction_kernel_local_x - 1)
                           / reduction_kernel_local_x);// * reduction_kernel_local_x;

//#ifdef DEBUG_MODE

    std::cout << "kernel_size_x: " << kernel_size_x << std::endl;
    std::cout << "kernel_size_y: " << kernel_size_y << std::endl;

    std::cout << "reduced elems count: " << reduced_elems_count << std::endl;
//#endif
}


void OCL_SOM2d_new::initLocalMemDims()
{

    local_mem_reduction = cl::__local(sizeof(int)
                                      * kernel_size_x
                                      * kernel_size_y);

    local_mem_reduction_global = cl::__local(sizeof(int)
                                             * reduction_kernel_local_x);

    local_mem_weights = cl::__local(sizeof(float)
                                          * kernel_size_x
                                          * kernel_size_y);

}

void OCL_SOM2d_new::initRanges()
{
    calculate_distances_global = cl::NDRange(kernel_global_x, kernel_global_y);
    calculate_distances_local = cl::NDRange(kernel_size_x, kernel_size_y);

    global_min_global = cl::NDRange(reduction_kernel_global_x);
    global_min_local = cl::NDRange(reduction_kernel_local_x);

//    update_global = cl::NDRange(kernel_global_x, kernel_global_y, dim);
//    update_local = cl::NDRange(kernel_size_x, kernel_size_y, dim);
}

void OCL_SOM2d_new::initDeviceBuffers()
{
    int som_size_x = get2dWidth();
    int som_size_y = get2dHeight();

    int slice_size = som_size_x * som_size_y;

    d_som = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                       d_data.size * sizeof(float));

    input_vector = cl::Buffer(d_context, CL_MEM_READ_ONLY,
                               dim * sizeof(float));

    distances = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                               slice_size * sizeof(float));

    //distances_host = new float[slice_size];

    out_min_indexes = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                                 reduced_elems_count * sizeof(int));

    out_min_values = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                                reduced_elems_count * sizeof(float));

}

void OCL_SOM2d_new::setKernelParams()
{
    // Set arguments to kernel
    calculate_distances_kernel.setArg(0, d_som);
    calculate_distances_kernel.setArg(1, input_vector);
    calculate_distances_kernel.setArg(2, distances);
    calculate_distances_kernel.setArg(3, get2dWidth());
    calculate_distances_kernel.setArg(4, get2dHeight());
    calculate_distances_kernel.setArg(5, dim);
    calculate_distances_kernel.setArg(6, local_mem_reduction);

    global_min_kernel.setArg(0, distances);
    global_min_kernel.setArg(1, out_min_values);
    global_min_kernel.setArg(2, out_min_indexes);
    global_min_kernel.setArg(3, local_mem_reduction_global);
    global_min_kernel.setArg(4, get2dWidth() * get2dHeight());


    //
    update_kernel.setArg(0, d_som);
    update_kernel.setArg(1, input_vector);
    update_kernel.setArg(2, get2dWidth());
    update_kernel.setArg(3, get2dHeight());
    update_kernel.setArg(4, dim);

    update_kernel.setArg(13, local_mem_weights);

}

//it can be more efficient...
void OCL_SOM2d_new::uploadDataToDevice()
{
    for(int i = 0; i < neurons.size(); ++i)
    {
        Row& row = neurons[i];
        float* row_ptr = d_data.data + d_data.x * d_data.neuron_size * i;

        for(int j = 0; j < row.size(); ++j)
        {
            Neuron& neuron = row[j];
            float* neuron_ptr = row_ptr + d_data.neuron_size * j;

            std::copy(neuron.begin(), neuron.end(), neuron_ptr);
        }
    }

    d_queue.enqueueWriteBuffer(d_som, CL_TRUE, 0, d_data.size * sizeof(float),
                               d_data.data);
}

void OCL_SOM2d_new::downloadDataFromDevice()
{
    d_queue.enqueueReadBuffer(d_som, CL_TRUE, 0, d_data.size * sizeof(float),
                              d_data.data);

    for(int i = 0; i < neurons.size(); ++i)
    {
        Row& row = neurons[i];
        float* row_ptr = d_data.data + d_data.x * d_data.neuron_size * i;

        for(int j = 0; j < row.size(); ++j)
        {
            Neuron& neuron = row[j];
            float* neuron_ptr = row_ptr + d_data.neuron_size * j;

            for(int k = 0; k < neuron.size(); ++k)
            {
                float* ptr = neuron_ptr + k;
                neuron[k] = *ptr;
            }
        }
    }
}


void OCL_SOM2d_new::notifyTrainingStart()
{
    uploadDataToDevice();

    update_radius = std::max(get2dWidth(), get2dHeight());

    final_min_vals = new float[reduced_elems_count];
    final_min_indexes = new int[reduced_elems_count];
}

void OCL_SOM2d_new::notifyTrainingEnd()
{
    downloadDataFromDevice();

    delete[] final_min_vals;
    delete[] final_min_indexes;
}

SOM::iterator OCL_SOM2d_new::identifyWinnerNeuron(const multi_img::Pixel &inputVec)
{
#ifdef TIME_MEASURE
    vole::Stopwatch running_time("Identify winner time");
#endif

    if(update_radius == 0)
        return SOM::iterator(new Iterator2d(this, 0, 0));;

    float* vec_ptr = (float*)(&(inputVec[0]));

    int global_min = 0;

    try
    {
        d_queue.enqueueWriteBuffer(input_vector, CL_TRUE, 0, dim * sizeof(float),
                                   vec_ptr);

        d_queue.enqueueNDRangeKernel(calculate_distances_kernel, cl::NullRange,
                                     calculate_distances_global, calculate_distances_local);

#if DEBUG_MODE
        d_queue.enqueueReadBuffer(distances, CL_TRUE, 0,
                                  get2dWidth() * get2dHeight() * sizeof(float),
                                  distances_host);

        std::cout << "distances: " << std::endl;

        float min_dist = FLT_MAX;
        float min_dist_host = FLT_MAX;

        for(int j = 0; j < get2dHeight(); ++j)
        {
            for(int i = 0; i < get2dWidth(); ++i)
            {
                int idx = j * get2dWidth() + i;
                float val = distances_host[idx];

                if(min_dist > val)
                    min_dist = val;
              //  std::cout << val << " ";
            }
           // std::cout << std::endl;
        }

        std::cout << "min value: " << sqrt(min_dist) << std::endl;


        for(int i = 0; i < (d_data.size / d_data.neuron_size); ++i)
        {
            float sum = 0;
            float* ptr = d_data.data + i * d_data.neuron_size;

            for(int j = 0; j < d_data.neuron_size; ++j)
            {
                float val = (*(ptr + j)) - inputVec[j];
            //    std::cout << val;
                sum += val * val;
            }

            if(sum < min_dist_host)
                min_dist_host = sum;
        }

        std::cout << "min value host: " << sqrt(min_dist_host) << std::endl;

#endif

        d_queue.enqueueNDRangeKernel(global_min_kernel, cl::NullRange,
                                     global_min_global, global_min_local);



        d_queue.enqueueReadBuffer(out_min_values, CL_TRUE, 0,
                                  sizeof(float) * reduced_elems_count, final_min_vals);

        d_queue.enqueueReadBuffer(out_min_indexes, CL_TRUE, 0,
                                  sizeof(int) * reduced_elems_count, final_min_indexes);

      //  int min_idx = 0;
        float min_val = FLT_MAX;
        global_min = final_min_indexes[0];

        for(int i = 0; i < reduced_elems_count; ++i)
        {
       //     std::cout << "partial val: " << final_min_vals[i] << std::endl;
       //     std::cout << "partial idx: " << final_min_indexes[i] << std::endl;

            float val = final_min_vals[i];
            if(val < min_val)
            {
                min_val = val;
                global_min = final_min_indexes[i];
            }
        }

        //std::cout << "global min value: " << min_val << std::endl;
       // std::cout << "global min index: " << global_min << std::endl;

    }
    catch(cl::Error error)
    {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    }

    int winner_x = global_min % get2dWidth();
    int winner_y = global_min / get2dWidth();

#ifdef DEBUG_MODE

    std::cout << "winner x: " << winner_x << " winner y: " << winner_y << std::endl;

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

int OCL_SOM2d_new::updateNeighborhood(iterator &neuron,
                                  const multi_img::Pixel &input,
                                  double sigma, double learnRate)
{
#ifdef TIME_MEASURE
    vole::Stopwatch running_time("Update time");
#endif

    // Get position of winner neuron in the 2d grid
    //Iterator2d *it = static_cast<Iterator2d *>(neuron.getBase());
    //cv::Point pos = it->getId();

    cv::Point pos = neuron.get2dCoordinates();
    double sigmaSquare = sigma * sigma;

    int x = pos.x;
    int y = pos.y;
    int width = get2dWidth();
    int height = get2dHeight();

    // update radius size

    while(update_radius > 0)
    {
        double dist = getDistanceSquared(pos,
                                         pos + cv::Point(update_radius, 0));
        double fakeGaussian = exp(-(dist)/(2.0*sigmaSquare));
        double weight = learnRate * fakeGaussian;

        if(weight < 0.01)
        {
            update_radius--;
        }
        else
        {
            break;
        }
    }

    if(update_radius == 0)
        return 0;

   // std::cout << "RADIUS: " << update_radius << std::endl;

    int left = x - update_radius;
    int right = x + update_radius;
    int top = y - update_radius;
    int bottom = y + update_radius;

    int left_dist = update_radius;
    int right_dist = update_radius;
    int top_dist = update_radius;
    int bottom_dist = update_radius;


    if(left < 0)
    {
        left_dist = x;
        left = 0;
    }

    if(right >= width)
    {
        right_dist = width - x - 1;
        right = width - 1;
    }

    if(top < 0)
    {
        top_dist = y;
        top = 0;
    }

    if(bottom >= height)
    {
        bottom_dist = height - y - 1;
        bottom = height - 1;
    }

    int update_area_width = right - left + 1;
    int update_area_height = bottom - top + 1;

    try
    {
        update_kernel.setArg(5, x);
        update_kernel.setArg(6, y);
        update_kernel.setArg(7, left);
        update_kernel.setArg(8, top);
        update_kernel.setArg(9, update_area_width);
        update_kernel.setArg(10, update_area_height);
        update_kernel.setArg(11, (float)sigmaSquare);
        update_kernel.setArg(12, (float)learnRate);

        int k_size_y = kernel_size_y;

        if(update_radius <= 16)
            k_size_y = 32;

        if(update_radius <= 8)
            k_size_y = 16;

        if(update_radius <= 4)
            k_size_y = 8;

        int global_x = kernel_size_x * update_area_width
                * ((update_area_height + k_size_y - 1) / k_size_y);
        int global_y = ((update_area_height + k_size_y - 1) / k_size_y)
                * k_size_y;

        //std::cout << "update | global_x: " << global_x << " global_y: " << global_y << std::endl;

        cl::NDRange update_global(global_x, global_y);
        cl::NDRange update_local(kernel_size_x, k_size_y);

        d_queue.enqueueNDRangeKernel(update_kernel, cl::NullRange,
                                     update_global, update_local);
    }
    catch(cl::Error error)
    {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    }

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
