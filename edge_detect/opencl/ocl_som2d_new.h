#ifndef OCLSOM2D_NEW_H
#define OCLSOM2D_NEW_H

#include <cstdlib>
#include "som2d.h"
#include "ocl_som_types.h"
#include "ocl_utils.h"

void ocl_som2d_test();

//#define DEBUG_MODE

class OCL_SOM2d_new : public SOM2d
{
public:
    OCL_SOM2d_new(const vole::EdgeDetectionConfig &conf, int dimension,
              std::vector<multi_img_base::BandDesc> meta);
    OCL_SOM2d_new(const vole::EdgeDetectionConfig &conf, const multi_img &data,
              std::vector<multi_img_base::BandDesc> meta);

    ~OCL_SOM2d_new();

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
    cl::Context d_context;
    cl::CommandQueue d_queue;
    cl::Program program;

    cl::Kernel calculate_distances_kernel;
    cl::Kernel global_min_first_kernel;
    cl::Kernel global_min_kernel;
    cl::Kernel update_kernel;

    /* OCL BUFFERS */
    cl::Buffer d_som;
    cl::Buffer input_vector;
    cl::Buffer distances;

    cl::Buffer out_min_indexes;
    cl::Buffer out_min_values;

    /* OCL RUNTIME PARAMETERS */

    int neuron_size_rounded;
    int total_size;
    int total_size_rounded;
    int group_size;

    int preffered_group_size;
    int max_group_size;

    int dist_find_local_x;
    int dist_find_local_y;

    int dist_find_global_x;
    int dist_find_global_y;

    int reduction_global;
    int reduction_local;
    int reduced_elems_count;

    unsigned int update_radius;

    /* LOCAL MEMORY OCCUPANCY */

    // for find_nearest_kernel
    cl::LocalSpaceArg dist_find_reduct_buff_local;
    cl::LocalSpaceArg reduct_buff_local;

    /* RANGES */

    cl::NDRange calc_dist_range_local;
    cl::NDRange calc_dist_range_global;

    cl::NDRange reduction_range_local;
    cl::NDRange reduction_range_global;

    float* final_min_vals;
    int* final_min_indexes;
};



#endif
