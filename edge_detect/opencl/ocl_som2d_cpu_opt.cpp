#include "ocl_som2d_cpu_opt.h"

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

inline int round_up(int value, int multiplicity)
{
    return ((value + multiplicity - 1) / multiplicity) * multiplicity;
}


som_data_cpu_opt::som_data_cpu_opt(int x, int y, int neuron_size)
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

        std::fill_n(data, size, 0.f);
    }
}

som_data_cpu_opt::~som_data_cpu_opt()
{
    if(data)
        delete[] data;
}


OCL_SOM2d_cpu_opt::OCL_SOM2d_cpu_opt(const vole::EdgeDetectionConfig &conf,
                     const multi_img &data,
                     std::vector<multi_img_base::BandDesc> meta)
    : SOM2d(conf, data, meta),
      d_data(conf.sidelength,
             conf.type == vole::SOM_SQUARE ? conf.sidelength : 1, data.size())
{    
}

OCL_SOM2d_cpu_opt::OCL_SOM2d_cpu_opt(const vole::EdgeDetectionConfig &conf,
                     int dimension,
                     std::vector<multi_img_base::BandDesc> meta)
    : SOM2d(conf, dimension, meta),
      d_data(conf.sidelength,
             conf.type == vole::SOM_SQUARE ? conf.sidelength : 1, round_up(dimension, 4))
{
    initOpenCL();
    initParamters();
    initLocalMemDims();
    initRanges();
    initDeviceBuffers();
    //uploadDataToDevice();
    setKernelParams();
}

OCL_SOM2d_cpu_opt::~OCL_SOM2d_cpu_opt()
{
}

void OCL_SOM2d_cpu_opt::initOpenCL()
{
    init_opencl(d_context, d_queue);    

    std::cout << "ocl som2d_new hello world!" << std::endl;

    cl::Device queue_device = d_queue.getInfo<CL_QUEUE_DEVICE>();

    compute_units = queue_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

    std::cout << "compute units: " << compute_units << std::endl;

    std::string source = read_source("kernels/som2d_cpu_opt.cl");

    std::stringstream stream;
    stream << "-DVEC_SIZE=" << (round_up(dim, 4) >> 2);

#ifdef DEBUG_MODE
    stream << " -DDEBUG_MODE -Werror"
#endif

    program = build_cl_program(d_context, source, stream.str());

    // Make kernels
    calculate_distances_kernel = cl::Kernel(program, "calculate_distances");
    global_min_kernel = cl::Kernel(program, "find_global_min");
    update_kernel = cl::Kernel(program, "update_network");
}

void OCL_SOM2d_cpu_opt::initParamters()
{
    reduction_kernel_local_x = 1;
    reduction_kernel_global_x = compute_units;

    reduced_elems_count = reduction_kernel_global_x;
}


void OCL_SOM2d_cpu_opt::initLocalMemDims()
{
}

void OCL_SOM2d_cpu_opt::initRanges()
{
    calculate_distances_global = cl::NDRange(get2dWidth() * get2dHeight());
    calculate_distances_local = cl::NullRange;

    global_min_global = cl::NDRange(reduction_kernel_global_x);
    global_min_local = cl::NDRange(reduction_kernel_local_x);
}

void OCL_SOM2d_cpu_opt::initDeviceBuffers()
{
    int som_size_x = get2dWidth();
    int som_size_y = get2dHeight();

    int slice_size = som_size_x * som_size_y;

    d_som = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                       d_data.size * sizeof(float));

    input_vector = cl::Buffer(d_context, CL_MEM_READ_ONLY,
                               round_up(dim, 4) * sizeof(float));

    distances = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                               slice_size * sizeof(float));

    //distances_host = new float[slice_size];

    out_min_indexes = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                                 reduced_elems_count * sizeof(int));

    out_min_values = cl::Buffer(d_context, CL_MEM_READ_WRITE,
                                reduced_elems_count * sizeof(float));

}

void OCL_SOM2d_cpu_opt::setKernelParams()
{
    // Set arguments to kernel
    calculate_distances_kernel.setArg(0, d_som);
    calculate_distances_kernel.setArg(1, input_vector);
    calculate_distances_kernel.setArg(2, distances);
//    calculate_distances_kernel.setArg(3, get2dWidth());
//    calculate_distances_kernel.setArg(4, get2dHeight());

    global_min_kernel.setArg(0, distances);
    global_min_kernel.setArg(1, out_min_values);
    global_min_kernel.setArg(2, out_min_indexes);
    global_min_kernel.setArg(3, get2dWidth() * get2dHeight() / reduction_kernel_global_x);

    update_kernel.setArg(0, d_som);
    update_kernel.setArg(1, input_vector);
    update_kernel.setArg(2, get2dWidth());
  //  update_kernel.setArg(3, get2dHeight());
}

void OCL_SOM2d_cpu_opt::uploadDataToDevice()
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

    float* zero_vector = new float[d_data.neuron_size];
    std::fill_n(zero_vector, d_data.neuron_size, 0.f);

    d_queue.enqueueWriteBuffer(input_vector, CL_TRUE, 0, d_data.neuron_size * sizeof(float),
                               zero_vector);
    delete[] zero_vector;
}

void OCL_SOM2d_cpu_opt::downloadDataFromDevice()
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


void OCL_SOM2d_cpu_opt::notifyTrainingStart()
{
    uploadDataToDevice();

    update_radius = std::max(get2dWidth(), get2dHeight());

    final_min_vals = new float[reduced_elems_count];
    final_min_indexes = new int[reduced_elems_count];
}

void OCL_SOM2d_cpu_opt::notifyTrainingEnd()
{
    downloadDataFromDevice();

    delete[] final_min_vals;
    delete[] final_min_indexes;
}

SOM::iterator OCL_SOM2d_cpu_opt::identifyWinnerNeuron(const multi_img::Pixel &inputVec)
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

        d_queue.enqueueNDRangeKernel(global_min_kernel, cl::NullRange,
                                     global_min_global, global_min_local);

        d_queue.enqueueReadBuffer(out_min_values, CL_TRUE, 0,
                                  sizeof(float) * reduced_elems_count, final_min_vals);

        d_queue.enqueueReadBuffer(out_min_indexes, CL_TRUE, 0,
                                  sizeof(int) * reduced_elems_count, final_min_indexes);

        float min_val = FLT_MAX;
        global_min = final_min_indexes[0];

        for(int i = 0; i < reduced_elems_count; ++i)
        {
            float val = final_min_vals[i];
            if(val < min_val)
            {
                min_val = val;
                global_min = final_min_indexes[i];
            }
        }
    }
    catch(cl::Error error)
    {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    }

    int winner_x = global_min % get2dWidth();
    int winner_y = global_min / get2dWidth();

    return SOM::iterator(new Iterator2d(this, winner_x, winner_y));
}

int OCL_SOM2d_cpu_opt::updateNeighborhood(iterator &neuron,
                                  const multi_img::Pixel &input,
                                  double sigma, double learnRate)
{
#ifdef TIME_MEASURE
    vole::Stopwatch running_time("Update time");
#endif

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

    int left = x - update_radius;
    int right = x + update_radius;
    int top = y - update_radius;
    int bottom = y + update_radius;

    if(left < 0)
        left = 0;

    if(right >= width)
        right = width - 1;

    if(top < 0)
        top = 0;

    if(bottom >= height)
        bottom = height - 1;

    int update_area_width = right - left + 1;
    int update_area_height = bottom - top + 1;

    int winner[] = {x, y};
    int offset[] = {left, top};

    try
    {
        update_kernel.setArg(3, sizeof(winner), winner);
        update_kernel.setArg(4, sizeof(offset), offset);
        update_kernel.setArg(5, (float)sigmaSquare);
        update_kernel.setArg(6, (float)learnRate);

        cl::NDRange update_global(update_area_width, update_area_height);

        d_queue.enqueueNDRangeKernel(update_kernel, cl::NullRange,
                                     update_global, cl::NullRange);
    }
    catch(cl::Error error)
    {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    }

    return 42;
}
