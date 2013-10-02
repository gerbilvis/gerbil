#ifndef OCLSOM2D_CPU_OPT_H
#define OCLSOM2D_CPU_OPT_H

#include <cstdlib>
#include "som2d.h"
#include "ocl_utils.h"

void ocl_som2d_test();

//#define DEBUG_MODE

class som_data_cpu_opt
{
public:
    som_data_cpu_opt(int x, int y, int neuron_size);
    ~som_data_cpu_opt();

    int x, y, neuron_size;
    int size;
    float* data;
};

class OCL_SOM2d_cpu_opt : public SOM2d
{
public:
    OCL_SOM2d_cpu_opt(const vole::EdgeDetectionConfig &conf, int dimension,
              std::vector<multi_img_base::BandDesc> meta);
    OCL_SOM2d_cpu_opt(const vole::EdgeDetectionConfig &conf, const multi_img &data,
              std::vector<multi_img_base::BandDesc> meta);

    ~OCL_SOM2d_cpu_opt();

    SOM::iterator identifyWinnerNeuron(const multi_img::Pixel &inputVec);
    int updateNeighborhood(iterator &neuron, const multi_img::Pixel &input,
                           double sigma, double learnRate);

    void notifyTrainingStart();
    void notifyTrainingEnd();

private:

    void initOpenCL();
    void initParamters();
    void initLocalMemDims();
    void initRanges();
    void initDeviceBuffers();
    void setKernelParams();
    void uploadDataToDevice();
    void downloadDataFromDevice();

    som_data_cpu_opt d_data;
    int total_size;
    cl::Context d_context;
    cl::CommandQueue d_queue;
    cl::Program program;

    cl::Kernel calculate_distances_kernel;
    cl::Kernel global_min_kernel;
    cl::Kernel update_kernel;

    /* OCL BUFFERS */
    cl::Buffer d_som;
    cl::Buffer input_vector;
    cl::Buffer distances;

    cl::Buffer out_min_indexes;
    cl::Buffer out_min_values;

    /* OCL RUNTIME PARAMETERS */

    int compute_units;

    int kernel_size_x;
    int kernel_size_y;

    int reduction_kernel_global_x;
    int reduction_kernel_local_x;
    int reduced_elems_count;

    int kernel_global_x;
    int kernel_global_y;

    int update_radius;

    /* LOCAL MEMORY OCCUPANCY */

    // for find_nearest_kernel
    cl::LocalSpaceArg local_mem_reduction;
    cl::LocalSpaceArg local_mem_reduction_global;
    cl::LocalSpaceArg local_mem_weights;

    /* RANGES */

    cl::NDRange calculate_distances_global;
    cl::NDRange calculate_distances_local;

    cl::NDRange global_min_global;
    cl::NDRange global_min_local;

    float* final_min_vals;
    int* final_min_indexes;
};
#endif
