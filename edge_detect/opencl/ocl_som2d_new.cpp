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

extern const char* som2d_new;

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
    //std::string source = read_source("kernels/som2d_new.cl");
    std::string source(som2d_new);

#ifdef DEBUG_MODE
    program = build_cl_program(d_context, source, "-DDEBUG_MODE -Werror");
#else
    program = build_cl_program(d_context, source);
#endif

    // Make kernels
    calculate_distances_kernel = cl::Kernel(program, "calculate_distances");
    global_min_first_kernel = cl::Kernel(program, "find_global_first_pass");
    global_min_kernel = cl::Kernel(program, "find_global_min");
    update_kernel = cl::Kernel(program, "update_network");
}

void OCL_SOM2d_new::initParamters()
{
    int som_size_x = get2dWidth();
    int som_size_y = get2dHeight();

    kernel_size_x = round_up_power2(dim);
    kernel_size_y = 512 / kernel_size_x;

    kernel_global_x = kernel_size_x * som_size_x;// * (( som_size_y + kernel_size_y - 1) / kernel_size_y);
    kernel_global_y = round_up(som_size_y, kernel_size_y);

    reduction_kernel_local_x = 512;
    reduction_kernel_global_x = round_up(som_size_x * som_size_y, reduction_kernel_local_x);


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

    local_mem_weights = cl::__local(sizeof(float) * kernel_size_y);

}

void OCL_SOM2d_new::initRanges()
{
    calculate_distances_global = cl::NDRange(kernel_global_x/2, kernel_global_y);
    calculate_distances_local = cl::NDRange(kernel_size_x/2, kernel_size_y);

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

    /* at least two uint values are needed to
     * store final winner x, y coordinates */
    out_min_indexes = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                                 std::max(2, reduced_elems_count) * sizeof(unsigned int));

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

    global_min_first_kernel.setArg(0, distances);
    global_min_first_kernel.setArg(1, out_min_values);
    global_min_first_kernel.setArg(2, out_min_indexes);
    global_min_first_kernel.setArg(3, local_mem_reduction_global);
    global_min_first_kernel.setArg(4, get2dWidth() * get2dHeight());

    global_min_kernel.setArg(0, out_min_values);
    global_min_kernel.setArg(1, out_min_indexes);
    global_min_kernel.setArg(2, get2dWidth());

    //
    update_kernel.setArg(0, d_som);
    update_kernel.setArg(1, input_vector);
    update_kernel.setArg(2, out_min_indexes);
    update_kernel.setArg(3, get2dWidth());
    update_kernel.setArg(4, get2dHeight());
    update_kernel.setArg(5, dim);

    update_kernel.setArg(12, local_mem_weights);

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

    update_radius = std::max(get2dWidth() / 2, get2dHeight() / 2);

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

    unsigned int global_min[2] = {0, 0};

    try
    {
        d_queue.enqueueWriteBuffer(input_vector, CL_TRUE, 0, dim * sizeof(float),
                                   vec_ptr);

        d_queue.enqueueNDRangeKernel(calculate_distances_kernel, cl::NullRange,
                                     calculate_distances_global, calculate_distances_local);


        d_queue.enqueueNDRangeKernel(global_min_first_kernel, cl::NullRange,
                                     global_min_global, global_min_local);


        int reduction_kernel_local = 512;
        int elems_to_reduce = reduced_elems_count;

        /* second step of reduction, only for very large soms there
         * will be more than one iteration performed*/
        do {

            if(elems_to_reduce < reduction_kernel_local)
            {
                reduction_kernel_local = round_up_power2(elems_to_reduce);
            }

            int reduction_global = round_up(elems_to_reduce,
                                            reduction_kernel_local);

            global_min_kernel.setArg(3, elems_to_reduce);

            global_min_kernel.setArg(4, cl::__local(sizeof(int)
                                                    * reduction_kernel_local));

            cl::NDRange range_global(reduction_global);
            cl::NDRange range_local(reduction_kernel_local);

            d_queue.enqueueNDRangeKernel(global_min_kernel, cl::NullRange,
                                         range_global, range_local);

            elems_to_reduce = ((reduction_global + reduction_kernel_local - 1)
                              / reduction_kernel_local);

        }while(elems_to_reduce > 1);


//        d_queue.enqueueNDRangeKernel(global_min_kernel, cl::NullRange,
//                                     global_min_global, global_min_local);

//        d_queue.enqueueReadBuffer(out_min_values, CL_TRUE, 0,
//                                  sizeof(float) * reduced_elems_count, final_min_vals);



//        d_queue.enqueueReadBuffer(out_min_indexes, CL_TRUE, 0,
//                                  sizeof(unsigned int) * 2, &global_min);

    }
    catch(cl::Error error)
    {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    }

    int winner_x = global_min[0];
    int winner_y = global_min[1];

#ifdef TIME_MEASURE
    d_queue.finish();
#endif

    //std::cout << "winner x: " << winner_x << " winner y: " << winner_y << std::endl;

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

int OCL_SOM2d_new::updateNeighborhood(iterator &neuron,
                                  const multi_img::Pixel &input,
                                  double sigma, double learnRate)
{
    //return 1;

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
        double dist = getDistanceSquared(cv::Point(0, 0), cv::Point(update_radius, 0));
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

//    std::cout << "x: " << x << " y: " << y
//              << " radius: " << update_radius
//              << " left: " << left << " top: " << top
//              << " width: " << update_area_width
//              << " height: " << update_area_height << std::endl;

    try
    {
     //   update_kernel.setArg(5, x);
     //   update_kernel.setArg(6, y);

        left = 0;
        top = 0;
        update_area_width = width;
        update_area_height = height;

        update_kernel.setArg(6, left);
        update_kernel.setArg(7, top);
        update_kernel.setArg(8, update_area_width);
        update_kernel.setArg(9, update_area_height);
        update_kernel.setArg(10, (float)sigmaSquare);
        update_kernel.setArg(11, (float)learnRate);

        int k_size_y = kernel_size_y;

        if(update_radius <= 16 || height <= 32)
            k_size_y = 32;

        if(update_radius <= 8)
            k_size_y = 16;

        if(update_radius <= 4)
            k_size_y = 8;

        k_size_y = std::min(k_size_y, kernel_size_y);

        int global_x = kernel_size_x * update_area_width;
                //* ((update_area_height + k_size_y - 1) / k_size_y);
        //int global_y = ((update_area_height + k_size_y - 1) / k_size_y)
          //      * k_size_y;
        int global_y = round_up(update_area_height, k_size_y);

//        std::cout << "x groups: " << update_area_width * ((update_area_height + k_size_y - 1) / k_size_y) << std::endl;
//        std::cout << "update | global_x: " << global_x << " global_y: " << global_y << std::endl;
//        std::cout << "update | local_x: " << kernel_size_x << " local_y: " << k_size_y << std::endl;

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

#ifdef TIME_MEASURE
    d_queue.finish();
#endif

    return 42;
}
