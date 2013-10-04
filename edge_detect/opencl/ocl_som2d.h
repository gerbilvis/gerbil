#ifndef OCLSOM2D_H
#define OCLSOM2D_H

#include <cstdlib>
#include "som2d.h"
#include "ocl_utils.h"

#include "ocl_som_types.h"

void ocl_som2d_test();

//#define DEBUG_MODE

class OCL_SOM2d : public SOM2d
{
public:
    OCL_SOM2d(const vole::EdgeDetectionConfig &conf, int dimension,
              std::vector<multi_img_base::BandDesc> meta);
    OCL_SOM2d(const vole::EdgeDetectionConfig &conf, const multi_img &data,
              std::vector<multi_img_base::BandDesc> meta);

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

    ocl_som_data d_data;
    int total_size;
    cl::Context d_context;
    cl::CommandQueue d_queue;
    cl::Program program;

    cl::Kernel find_nearest_kernel;
    cl::Kernel global_min_kernel;
    cl::Kernel update_kernel;

    /* OCL BUFFERS */
    cl::Buffer d_som;
    cl::Buffer input_vectors;

    cl::Buffer out_min_indexes;
    cl::Buffer out_min_values;
    cl::Buffer global_min_idx;

#ifdef DEBUG_MODE
    cl::Buffer neighbourhood_verify;
#endif

    /* OCL RUNTIME PARAMETERS */

    int preferred_max_block_size;
    int kernel_size_x;
    int kernel_size_y;

    int reduction_kernel_global_x;
    int reduction_kernel_global_y;

    int reduction_kernel_total;

    int kernel_global_x;
    int kernel_global_y;


    /* LOCAL MEMORY OCCUPANCY */

    // for find_nearest_kernel
    cl::LocalSpaceArg local_mem_subsom;
    cl::LocalSpaceArg local_mem_reduction;
    // for global_min_kernel
    cl::LocalSpaceArg local_mem_reduction_global;
    // for update kernel
    cl::LocalSpaceArg local_mem_neighbourhood;
    cl::LocalSpaceArg local_mem_input_vec;

    /* RANGES */

    cl::NDRange find_nearest_global;
    cl::NDRange find_nearest_local;

    cl::NDRange global_min_global;
    cl::NDRange global_min_local;

    cl::NDRange update_global;
    cl::NDRange update_local;
};



#endif
