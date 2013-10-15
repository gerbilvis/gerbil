#include "ocl_som2d.h"

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

Ocl_SOM2d::Ocl_SOM2d(const vole::EdgeDetectionConfig &conf,
                     const multi_img &data,
                     std::vector<multi_img_base::BandDesc> meta)
    : SOM2d(conf, data, meta),
      d_data(conf.sidelength,
             conf.type == vole::SOM_SQUARE ? conf.sidelength : 1,
             1, round_up_power2(data.size()))
{
    local_distances = 0;
}

Ocl_SOM2d::Ocl_SOM2d(const vole::EdgeDetectionConfig &conf,
                     int dimension,
                     std::vector<multi_img_base::BandDesc> meta)
    : SOM2d(conf, dimension, meta),
      d_data(conf.sidelength,
             conf.type == vole::SOM_SQUARE ? conf.sidelength : 1,
             1, round_up_power2(dimension))
{
    local_distances = 0;

    initOpenCL();
    initParamters();
    initLocalMemDims();
    initRanges();
    initDeviceBuffers();
    //uploadDataToDevice();
}

Ocl_SOM2d::~Ocl_SOM2d()
{
    if(local_distances)
        delete[] local_distances;
}


void Ocl_SOM2d::initOpenCL()
{
    init_opencl(d_context, d_queue);

    std::cout << "ocl som2d_new hello world!" << std::endl;
    //std::string source = read_source("kernels/som2d_new.cl");
    std::string source(som2d);

    cl::Device device = d_queue.getInfo<CL_QUEUE_DEVICE>();

    int device_type = device.getInfo<CL_DEVICE_TYPE>();

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
    stream << " -DSOM_SIZE_Z=" << 1;

    if(device_type == CL_DEVICE_TYPE_CPU)
        stream << " -DCPU";

#ifdef DEBUG_MODE
    stream << " -DDEBUG_MODE -Werror";
#endif

    program = build_cl_program(d_context, source, stream.str());

    // Make kernels
    calculate_distances_kernel = cl::Kernel(program, "calculate_distances");
    find_local_minima = cl::Kernel(program, "find_global_first_pass");
    find_global_minima = cl::Kernel(program, "find_global_min");
    update_kernel = cl::Kernel(program, "update_network");
}

void Ocl_SOM2d::initParamters()
{
    total_size = width * height;
    total_size_rounded = round_up_power2(total_size);

    group_size = std::min(preffered_group_size, max_group_size);

    dist_find_local_x = neuron_size_rounded / 2;
    dist_find_local_y = group_size / dist_find_local_x;

    dist_find_global_x = dist_find_local_x * width;
    dist_find_global_y = round_up(height, dist_find_local_y);

    reduction_local = group_size;
    //reduction_global = round_up_power2(total_size);
    reduction_global = std::max(round_up_power2(total_size), reduction_local);

    assert(reduction_global % reduction_local == 0);

    reduced_elems_count = ((reduction_global + reduction_local - 1)
                          / reduction_local);
}


void Ocl_SOM2d::initLocalMemDims()
{

    dist_find_reduct_buff_local = cl::__local(sizeof(float)
                                   * dist_find_local_x
                                   * dist_find_local_y);

    reduct_buff_local = cl::__local(sizeof(float) * reduction_local);

    //update_weights_buff_local = cl::__local(sizeof(float) * kernel_size_y);

}


void Ocl_SOM2d::initRanges()
{
#ifdef DEBUG_MODE
    std::cout << "dist_find_local_x: " << dist_find_local_x << std::endl;
    std::cout << "dist_find_local_y: " << dist_find_local_y << std::endl;

    std::cout << "dist_find_global_x: " << dist_find_global_x << std::endl;
    std::cout << "dist_find_global_y: " << dist_find_global_y << std::endl;
#endif

    calc_dist_range_local = cl::NDRange(dist_find_local_x, dist_find_local_y);
    calc_dist_range_global = cl::NDRange(dist_find_global_x, dist_find_global_y);


    reduction_range_local = cl::NDRange(reduction_local);
    reduction_range_global = cl::NDRange(reduction_global);

//    update_global = cl::NDRange(kernel_global_x, kernel_global_y, dim);
//    update_local = cl::NDRange(kernel_size_x, kernel_size_y, dim);
}

void Ocl_SOM2d::initDeviceBuffers()
{
    d_som = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                       d_data.size * sizeof(float));

//    input_vector = cl::Buffer(d_context, CL_MEM_READ_ONLY,
//                              neuron_size_rounded * sizeof(float));

    distances = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                               total_size * sizeof(float));

    //distances_host = new float[slice_size];

//    float* zero_vector = new float[neuron_size_rounded];
//    std::fill(zero_vector, zero_vector + neuron_size_rounded, 0.f);

//    d_queue.enqueueWriteBuffer(input_vector, CL_TRUE, 0,
//                               neuron_size_rounded * sizeof(float),
//                               zero_vector);
//    delete[] zero_vector;

    /* at least two uint values are needed to
     * store final winner x, y coordinates */
    out_min_indexes = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                                 std::max(2, reduced_elems_count)
                                 * sizeof(int));

    out_min_values = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                                reduced_elems_count * sizeof(float));

//    float f_max[] = {FLT_MAX, FLT_MAX};

//    d_queue.enqueueWriteBuffer(out_min_indexes, CL_TRUE, 0, sizeof(float) * 2,
//                               &f_max);
}

void Ocl_SOM2d::setKernelParams()
{
    // Set arguments to kernel
    calculate_distances_kernel.setArg(0, d_som);
    calculate_distances_kernel.setArg(1, input_vectors);
    calculate_distances_kernel.setArg(2, distances);
    //calculate_distances_kernel.setArg(3, 0);
    calculate_distances_kernel.setArg(4, dist_find_reduct_buff_local);

    find_local_minima.setArg(0, distances);
    find_local_minima.setArg(1, out_min_values);
    find_local_minima.setArg(2, out_min_indexes);
    find_local_minima.setArg(3, reduct_buff_local);
    find_local_minima.setArg(4, total_size);

    find_global_minima.setArg(0, out_min_values);
    find_global_minima.setArg(1, out_min_indexes);
//    global_min_kernel.setArg(2, width);

    update_kernel.setArg(0, d_som);
    update_kernel.setArg(1, input_vectors);
    update_kernel.setArg(2, out_min_indexes);
//    update_kernel.setArg(3, 0);
//    update_kernel.setArg(3, width);
//    update_kernel.setArg(4, height);
}

void Ocl_SOM2d::uploadDataToDevice()
{
    std::fill(d_data.data, d_data.data + d_data.size, 0.f);

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

    d_queue.enqueueWriteBuffer(d_som, CL_TRUE,
                               0, d_data.size * sizeof(float),
                               d_data.data);
}

void Ocl_SOM2d::downloadDataFromDevice()
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

void Ocl_SOM2d::uploadTrainingVectors()
{
    std::cout << "uploading training vectors!" << std::endl;

    int vectors_num = training_vectors.size();
    int buff_size = vectors_num * neuron_size_rounded;


    input_vectors = cl::Buffer(d_context, CL_MEM_READ_ONLY,
                              buff_size * sizeof(float));


    float* data = new float[buff_size];

    std::fill(data, data + buff_size, 0.f);

    for(int i = 0; i < vectors_num; ++i)
    {
        float* ptr = data + i * neuron_size_rounded;
        multi_img::Pixel& vec = training_vectors[i];

        std::copy(vec.begin(), vec.end(), ptr);
    }

    d_queue.enqueueWriteBuffer(input_vectors, CL_TRUE,
                               0, buff_size * sizeof(float),
                               data);

    delete[] data;

    std::cout << "uploading training vectors finished!" << std::endl;
}


void Ocl_SOM2d::notifyTrainingStart()
{
    uploadDataToDevice();
    update_radius = std::max(get2dWidth() - 1, get2dHeight() - 1);
}

void Ocl_SOM2d::notifyTrainingEnd()
{

    std::cout << "notify training end!" << std::endl;

    //int training_vec_num = training_vectors.size();
    uploadTrainingVectors();
    setKernelParams();

//    calculate_distances_kernel.setArg(1, input_vector);
//    update_kernel.setArg(1, input_vector);

    for(int i = 0; i < training_vectors.size(); ++i)
    {
        unsigned int global_min[2] = {0, 0};

        try
        {
            calculate_distances_kernel.setArg(3, i);

            d_queue.enqueueNDRangeKernel(calculate_distances_kernel,
                                         cl::NullRange,
                                         calc_dist_range_global,
                                         calc_dist_range_local);

            d_queue.enqueueNDRangeKernel(find_local_minima,
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

                find_global_minima.setArg(2, elems_to_reduce);

                find_global_minima.setArg(3, cl::__local(sizeof(int)
                                                        * second_reduct_local));

                cl::NDRange range_global(second_reduct_global);
                cl::NDRange range_local(second_reduct_local);

                d_queue.enqueueNDRangeKernel(find_global_minima, cl::NullRange,
                                             range_global, range_local);

                elems_to_reduce = ((second_reduct_global + second_reduct_local - 1)
                                  / second_reduct_local);

            }while(elems_to_reduce > 1);

#ifdef DEBUG_MODE
            d_queue.enqueueReadBuffer(out_min_indexes, CL_TRUE, 0,
                                      sizeof(int) * 2, &global_min);
#endif

        }
        catch(cl::Error error)
        {
            std::cout << error.what() << "(" << error.err() << ")" << std::endl;
        }

//        int winner_x = global_min[0];
//        int winner_y = global_min[1];

//        std::cout << "winnder_x: " << winner_x << ", winner_y: "
//                  << winner_y << std::endl;

        //int winner_x_t = global_min[1] >> 16;
        //int winner_y_t = global_min[1] & 0xFFFF;

        // ***************
        // UPDATE !
        // ***************

        double sigma = training_params[i].first;
        double learnRate = training_params[i].second;

        //cv::Point pos = neuron.get2dCoordinates();
        double sigmaSquare = sigma * sigma;

        while(update_radius > 0)
        {
            double dist = getDistanceSquared(cv::Point(0, 0),
                                             cv::Point(update_radius, 0));
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
            break;

        int update_area_width = std::min(update_radius * 2 + 1, width);
        int update_area_height = std::min(update_radius * 2 + 1, height);

        try
        {
            int local_x = neuron_size_rounded / 2;
            int local_y = std::min(group_size / local_x,
                                   round_up_power2(update_radius * 2 + 1));

            int global_x = local_x * update_area_width;
            int global_y = round_up(update_area_height, local_y);

            cl::LocalSpaceArg update_weights_buff_local
                    = cl::__local(sizeof(float) * local_y);

            update_kernel.setArg(3, i);
            update_kernel.setArg(4, update_radius);
            update_kernel.setArg(5, (float)sigmaSquare);
            update_kernel.setArg(6, (float)learnRate);
            update_kernel.setArg(7, update_weights_buff_local);

            cl::NDRange update_global(global_x, global_y);
            cl::NDRange update_local(local_x, local_y);

            d_queue.enqueueNDRangeKernel(update_kernel, cl::NullRange,
                                         update_global, update_local);
        }
        catch(cl::Error error)
        {
            std::cout << error.what() << "(" << error.err() << ")" << std::endl;
        }

    //#ifdef TIME_MEASURE
    //    d_queue.finish();
  //  #endif
    }

    downloadDataFromDevice();
}

/*
 * IDENTIFY WINNER NEURON
 */
SOM::iterator Ocl_SOM2d::identifyWinnerNeuron(const multi_img::Pixel &inputVec)
{
#ifdef TIME_MEASURE
    vole::Stopwatch running_time("Identify winner time");
#endif

    training_vectors.push_back(inputVec);

    //return SOM::iterator(new Iterator2d(this, winner_x, winner_y));
    return SOM::iterator(new Iterator2d(this, 0, 0));
}

/*
 * UPDATE NEIGHBORHOOD
 */
int Ocl_SOM2d::updateNeighborhood(iterator &neuron,
                                  const multi_img::Pixel &input,
                                  double sigma, double learnRate)
{
    training_params.push_back(std::pair<double, double>(sigma, learnRate));

    return 42;
}


static bool sortpair(std::pair<double, SOM::iterator> i,
                     std::pair<double, SOM::iterator> j) {
    return (i.first < j.first);
}

void Ocl_SOM2d::closestN(const multi_img::Pixel &inputVec,
                         std::vector<std::pair<double, iterator> > &heap)
{
//    //std::cout << "finding closest N!" << std::endl;

//    if(local_distances == 0)
//        local_distances = new float[total_size_rounded];

//    float* input_vec = new float[neuron_size_rounded];
//    std::fill(input_vec, input_vec + neuron_size_rounded, 0.f);
//    std::copy(inputVec.begin(), inputVec.end(), input_vec);

//    d_queue.enqueueWriteBuffer(input_vectors, CL_TRUE, 0,
//                           neuron_size_rounded * sizeof(float), input_vec);

//    delete[] input_vec;


//    calculate_distances_kernel.setArg(3, 0);
//    d_queue.enqueueNDRangeKernel(calculate_distances_kernel,
//                                 cl::NullRange,
//                                 calc_dist_range_global,
//                                 calc_dist_range_local);

//    d_queue.enqueueReadBuffer(distances, CL_TRUE, 0,
//                          total_size * sizeof(float), local_distances);


//    // initialize with maximum values
//    for (int i = 0; i < heap.size(); ++i)
//        heap[i].first = std::numeric_limits<double>::max();

//    // find closest Neurons to inputVec in the SOM
//    // iterate over all neurons in grid
//    for (int i = 0; i < total_size; ++i)
//    {
//        double dist = local_distances[i];
//        /* compare current distance with the maximum of the N shortest
//         * found distances */
//        if (dist < heap[0].first) {
//            // remove max. value in heap

//            //std::cout << "min: " << dist << std::endl;

//            std::pop_heap(heap.begin(), heap.end(), sortpair);

//            /* max element is now on position "back" and should be popped
//             * instead we overwrite it directly with the new element */
//            std::pair<double, SOM::iterator> &back = heap.back();
//            back.first = dist;
//            back.second = SOM::iterator(new Iterator2d(this, i % width, i / width));;

//            std::push_heap(heap.begin(), heap.end(), sortpair);
//        }
//    }

//    assert(heap[0].first != std::numeric_limits<double>::max());
//    std::sort_heap(heap.begin(), heap.end(), sortpair); // sort ascending


    SOM::closestN(inputVec, heap);
}
