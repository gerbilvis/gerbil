#include "ocl_som3d.h"

#include "iostream"

#include <CL/cl.hpp>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>

#include "stopwatch.h"

#include "ocl_utils.h"

//#define TIME_MEASURE

extern const char* som2d;

Ocl_SOM3d::Ocl_SOM3d(const vole::EdgeDetectionConfig &conf,
                     const multi_img &data,
                     std::vector<multi_img_base::BandDesc> meta)
    : SOM3d(conf, data, meta),
      d_data(conf.sidelength,
             (conf.type == vole::SOM_SQUARE || conf.type == vole::SOM_CUBE) ? conf.sidelength : 1,
             conf.type == vole::SOM_CUBE ? conf.sidelength : 1,
             round_up_power2(data.size()))
{    
}

Ocl_SOM3d::Ocl_SOM3d(const vole::EdgeDetectionConfig &conf,
                     int dimension,
                     std::vector<multi_img_base::BandDesc> meta)
    : SOM3d(conf, dimension, meta),
      d_data(conf.sidelength,
             (conf.type == vole::SOM_SQUARE  || conf.type == vole::SOM_CUBE) ? conf.sidelength : 1,
             conf.type == vole::SOM_CUBE ? conf.sidelength : 1,
             round_up_power2(dimension))
{
    initOpenCL();
    initParamters();
    initLocalMemDims();
    initRanges();
    initDeviceBuffers();
    //uploadDataToDevice();
    setKernelParams();
}

Ocl_SOM3d::~Ocl_SOM3d()
{
}


void Ocl_SOM3d::initOpenCL()
{
    init_opencl(d_context, d_queue);

    std::cout << "ocl som2d_new hello world!" << std::endl;
    //std::string source = read_source("kernels/som2d_new.cl");
    std::string source(som2d);

    cl::Device device = d_queue.getInfo<CL_QUEUE_DEVICE>();

    preffered_group_size = 512;
    max_group_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    neuron_size_rounded = round_up_power2(dim);

#ifdef DEBUG_MODE
    std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: "
              << CL_DEVICE_MAX_WORK_GROUP_SIZE << std::endl;
    std::cout << "CL_DEVICE_NAMEL: " << device.getInfo<CL_DEVICE_NAME>();
#endif

    std::stringstream stream;
    stream << "-DX_DIM=" << (neuron_size_rounded / 2);
    stream << " -DSOM_SIZE_X=" << width;
    stream << " -DSOM_SIZE_Y=" << height;
    stream << " -DSOM_SIZE_Z=" << depth;
    stream << " -DSOM_3D";

#ifdef DEBUG_MODE
    stream << " -DDEBUG_MODE -Werror";
#endif

    program = build_cl_program(d_context, source, stream.str());

    // Make kernels
    calculate_distances_kernel = cl::Kernel(program, "calculate_distances");
    global_min_first_kernel = cl::Kernel(program, "find_global_first_pass");
    global_min_kernel = cl::Kernel(program, "find_global_min");
    update_kernel = cl::Kernel(program, "update_network");
}

void Ocl_SOM3d::initParamters()
{
    total_size = width * height * depth;
    total_size_rounded = round_up_power2(total_size);

    group_size = std::min(preffered_group_size, max_group_size);

    dist_find_local_x = neuron_size_rounded / 2;
    dist_find_local_y = group_size / dist_find_local_x;
    dist_find_local_z = 1;

    dist_find_global_x = dist_find_local_x * width;
    dist_find_global_y = round_up(height, dist_find_local_y);
    dist_find_global_z = depth;

    reduction_local = group_size;
    reduction_global = std::max(round_up_power2(total_size), reduction_local);

    assert(reduction_global % reduction_local == 0);

    reduced_elems_count = ((reduction_global + reduction_local - 1)
                          / reduction_local);
}


void Ocl_SOM3d::initLocalMemDims()
{

    dist_find_reduct_buff_local = cl::__local(sizeof(float)
                                   * dist_find_local_x
                                   * dist_find_local_y);

    reduct_buff_local = cl::__local(sizeof(float) * reduction_local);

    //update_weights_buff_local = cl::__local(sizeof(float) * kernel_size_y);

}


void Ocl_SOM3d::initRanges()
{
//#ifdef DEBUG_MODE
    std::cout << "dist_find_local_x: " << dist_find_local_x << std::endl;
    std::cout << "dist_find_local_y: " << dist_find_local_y << std::endl;
    std::cout << "dist_find_local_z: " << dist_find_local_z << std::endl;

    std::cout << "dist_find_global_x: " << dist_find_global_x << std::endl;
    std::cout << "dist_find_global_y: " << dist_find_global_y << std::endl;
    std::cout << "dist_find_global_z: " << dist_find_global_z << std::endl;
//#endif

    calc_dist_range_local = cl::NDRange(dist_find_local_x,
                                        dist_find_local_y,
                                        dist_find_local_z);

    calc_dist_range_global = cl::NDRange(dist_find_global_x,
                                         dist_find_global_y,
                                         dist_find_global_z);

    reduction_range_local = cl::NDRange(reduction_local);
    reduction_range_global = cl::NDRange(reduction_global);

//    update_global = cl::NDRange(kernel_global_x, kernel_global_y, dim);
//    update_local = cl::NDRange(kernel_size_x, kernel_size_y, dim);
}

void Ocl_SOM3d::initDeviceBuffers()
{
    d_som = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                       d_data.size * sizeof(float));

    input_vector = cl::Buffer(d_context, CL_MEM_READ_ONLY,
                              neuron_size_rounded * sizeof(float));

    distances = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                               total_size * sizeof(float));

    //distances_host = new float[slice_size];

    float* zero_vector = new float[neuron_size_rounded];
    std::fill(zero_vector, zero_vector + neuron_size_rounded, 0.f);

    d_queue.enqueueWriteBuffer(input_vector, CL_TRUE, 0,
                               neuron_size_rounded * sizeof(float),
                               zero_vector);

    delete[] zero_vector;

    /* at least two uint values are needed to
     * store final winner x, y coordinates */
    out_min_indexes = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                                 std::max(3, reduced_elems_count)
                                 * sizeof(int));

    out_min_values = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                                reduced_elems_count * sizeof(float));

//    float f_max[] = {FLT_MAX, FLT_MAX};

//    d_queue.enqueueWriteBuffer(out_min_indexes, CL_TRUE, 0, sizeof(float) * 2,
//                               &f_max);
}

void Ocl_SOM3d::setKernelParams()
{
    // Set arguments to kernel
    calculate_distances_kernel.setArg(0, d_som);
    calculate_distances_kernel.setArg(1, input_vector);
    calculate_distances_kernel.setArg(2, distances);
//    calculate_distances_kernel.setArg(3, width);
//    calculate_distances_kernel.setArg(4, height);
    calculate_distances_kernel.setArg(3, dist_find_reduct_buff_local);

    global_min_first_kernel.setArg(0, distances);
    global_min_first_kernel.setArg(1, out_min_values);
    global_min_first_kernel.setArg(2, out_min_indexes);
    global_min_first_kernel.setArg(3, reduct_buff_local);
    global_min_first_kernel.setArg(4, total_size);

    global_min_kernel.setArg(0, out_min_values);
    global_min_kernel.setArg(1, out_min_indexes);
//    global_min_kernel.setArg(2, width);
//    global_min_kernel.setArg(3, height);

    update_kernel.setArg(0, d_som);
    update_kernel.setArg(1, input_vector);
    update_kernel.setArg(2, out_min_indexes);
//    update_kernel.setArg(3, width);
//    update_kernel.setArg(4, height);
//    update_kernel.setArg(5, depth);
}

void Ocl_SOM3d::uploadDataToDevice()
{
    std::fill(d_data.data, d_data.data + d_data.size, 0.f);

    int field_size = d_data.x * d_data.y * d_data.neuron_size;

    for(int i = 0; i < neurons.size(); ++i)
    {
        Field& field = neurons[i];

        float* field_ptr = d_data.data + field_size * i;

        for(int j = 0; j < field.size(); ++j)
        {
            Row& row = field[j];
            float* row_ptr = field_ptr + d_data.x * d_data.neuron_size * j;

            for(int k = 0; k < row.size(); ++k)
            {
                Neuron& neuron = row[k];
                float* neuron_ptr = row_ptr + d_data.neuron_size * k;
                std::copy(neuron.begin(), neuron.end(), neuron_ptr);
            }
        }
    }

    cl_int st = d_queue.enqueueWriteBuffer(d_som, CL_TRUE,
                                           0, d_data.size * sizeof(float),
                                           d_data.data);

    std::cout << "upload status: " << st << std::endl;
}

void Ocl_SOM3d::downloadDataFromDevice()
{
    d_queue.enqueueReadBuffer(d_som, CL_TRUE, 0, d_data.size * sizeof(float),
                              d_data.data);

    int field_size = d_data.x * d_data.y * d_data.neuron_size;

    for(int i = 0; i < neurons.size(); ++i)
    {
        Field& field = neurons[i];

        float* field_ptr = d_data.data + field_size * i;

        for(int j = 0; j < field.size(); ++j)
        {
            Row& row = field[j];
            float* row_ptr = field_ptr + d_data.x * d_data.neuron_size * j;

            for(int k = 0; k < row.size(); ++k)
            {
                Neuron& neuron = row[k];
                float* neuron_ptr = row_ptr + d_data.neuron_size * k;

               // std::copy(neuron.begin(), neuron.end(), neuron_ptr);

                for(int l = 0; l < neuron.size(); ++l)
                {
                    float* ptr = neuron_ptr + l;
                    neuron[l] = *ptr;
                }
            }
        }
    }
}


void Ocl_SOM3d::notifyTrainingStart()
{
    uploadDataToDevice();

    update_radius = std::max(get2dWidth() - 1, get2dHeight() - 1);
}

void Ocl_SOM3d::notifyTrainingEnd()
{
    downloadDataFromDevice();
}

SOM::iterator Ocl_SOM3d::identifyWinnerNeuron(const multi_img::Pixel &inputVec)
{

#ifdef TIME_MEASURE
    vole::Stopwatch running_time("Identify winner time");
#endif

    if(update_radius == 0)
        return SOM::iterator(new Iterator3d(this, 0, 0, 0));

    float* vec_ptr = (float*)(&(inputVec[0]));

    unsigned int global_min[3] = {0, 0, 0};

    try
    {
        d_queue.enqueueWriteBuffer(input_vector, CL_TRUE, 0,
                                   dim * sizeof(float), vec_ptr);

//        float f_max[] = {FLT_MAX, FLT_MAX};
//        d_queue.enqueueWriteBuffer(out_min_indexes, CL_TRUE, 0, sizeof(float) * 2,
//                                   &f_max);

        d_queue.enqueueNDRangeKernel(calculate_distances_kernel,
                                     cl::NullRange,
                                     calc_dist_range_global,
                                     calc_dist_range_local);

#ifdef DEBUG_MODE

        float* host_distances = new float[total_size];

        d_queue.enqueueReadBuffer(distances, CL_TRUE, 0,
                                  sizeof(float) * total_size, host_distances);

        float f1 = host_distances[0];
        float f2 = host_distances[1];
        float f3 = host_distances[2];
        float f4 = host_distances[3];

        delete[] host_distances;
#endif


        d_queue.enqueueNDRangeKernel(global_min_first_kernel,
                                     cl::NullRange,
                                     reduction_range_global,
                                     reduction_range_local);


        int second_reduct_local = std::min(max_group_size,
                                           preffered_group_size);
        int elems_to_reduce = reduced_elems_count;

        /* second step of reduction, only for very large soms there
         * will be more than one iteration performed*/
        do {

            if(elems_to_reduce < second_reduct_local)
            {
                second_reduct_local = round_up_power2(elems_to_reduce);
            }

            int second_reduct_global = round_up(elems_to_reduce,
                                       second_reduct_local);

            global_min_kernel.setArg(2, elems_to_reduce);

            global_min_kernel.setArg(3, cl::__local(sizeof(int)
                                                    * second_reduct_local));

            cl::NDRange range_global(second_reduct_global);
            cl::NDRange range_local(second_reduct_local);

            d_queue.enqueueNDRangeKernel(global_min_kernel, cl::NullRange,
                                         range_global, range_local);

            elems_to_reduce = ((second_reduct_global + second_reduct_local - 1)
                              / second_reduct_local);

        }while(elems_to_reduce > 1);

//#ifdef DEBUG_MODE
        d_queue.enqueueReadBuffer(out_min_indexes, CL_TRUE, 0,
                                  sizeof(int) * 3, &global_min);
//#endif

    }
    catch(cl::Error error)
    {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    }

//#ifdef DEBUG_MODE
    int winner_x = global_min[0];
    int winner_y = global_min[1];
    int winner_z = global_min[2];

    assert(winner_x >=0 && winner_x < width);
    assert(winner_y >=0 && winner_y < height);
    assert(winner_z >=0 && winner_z < depth);
//#endif
//    int winner_x_t = global_min[1] >> 16;
//    int winner_y_t = global_min[1] & 0xFFFF;

#ifdef TIME_MEASURE
    d_queue.finish();
#endif

    //return SOM::iterator(new Iterator2d(this, winner_x, winner_y));
    return SOM::iterator(new Iterator3d(this, 0, 0, 0));
}

int Ocl_SOM3d::updateNeighborhood(iterator &neuron,
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

    while(update_radius > 0)
    {
        double dist = getDistanceSquared(cv::Point3d(0, 0, 0),
                                         cv::Point3d(update_radius, 0, 0));
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
        return 42;

    int update_area_width = std::min((int)update_radius * 2 + 1, width);
    int update_area_height = std::min((int)update_radius * 2 + 1, height);
    int update_area_depth = std::min((int)update_radius * 2 + 1, depth);

    try
    {
        int local_x = neuron_size_rounded / 2;
        int local_y = std::min(group_size / local_x,
                               round_up_power2(update_radius * 2 + 1));

        int global_x = local_x * update_area_width;
        int global_y = round_up(update_area_height, local_y);

        cl::LocalSpaceArg update_weights_buff_local
                = cl::__local(sizeof(float) * local_y);

        update_kernel.setArg(3, (int)update_radius);
        update_kernel.setArg(4, (float)sigmaSquare);
        update_kernel.setArg(5, (float)learnRate);
        update_kernel.setArg(6, update_weights_buff_local);

#ifdef DEBUG_MODE
        std::cerr << "update | global_x: " << global_x
                  << " global_y: " << global_y << std::endl;
        std::cerr << "update | local_x: " << local_x
                  << " local_y: " << local_y << std::endl;
#endif
        cl::NDRange update_global(global_x, global_y, update_area_depth);
        cl::NDRange update_local(local_x, local_y, 1);

        d_queue.enqueueNDRangeKernel(update_kernel, cl::NullRange,
                                     update_global, update_local);
    }
    catch(cl::Error error)
    {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    }

#ifdef TIME_MEASURE
    d_queue.finish();
#endif

    return 42;
}
