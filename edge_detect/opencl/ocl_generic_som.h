#ifndef OCLGENERICSOM_H
#define OCLGENERICSOM_H

#include "som.h"
#include "ocl_som_types.h"
#include "ocl_utils.h"

#define OPENCL_PROFILE

class Ocl_GenericSOM
{
public:
    Ocl_GenericSOM(int som_width, int som_height, int som_depth,
                   int neuron_size, const std::string& ocl_params);

    virtual ~Ocl_GenericSOM();

    void train();

    void calculateAllDistances(const multi_img& image, float* distances);
    void calculateAllDistances();


    void clearSomData();

protected:
    ocl_som_data d_data;
    std::vector<multi_img::Pixel> training_vectors;
    std::vector<std::pair<double, double> > training_params;

    virtual void uploadDataToDevice();
    virtual void downloadDataFromDevice();
    virtual void uploadTrainingVectors();

private:
    void findMinima(int vector_idx);
    void updateRadius(int vector_idx);
    void update(int vector_idx);

    std::string oclParams;

    int neuron_size_rounded;
    int total_size;
    int total_size_rounded;

    void initOpenCL();
    void initParamters();
    void initLocalMemDims();
    void initRanges();
    void initDeviceBuffers();
    void setKernelParams();

    int group_size;

    int preffered_group_size;
    int max_group_size;

    int dist_find_local_x;
    int dist_find_local_y;
    int dist_find_local_z;

    int dist_find_global_x;
    int dist_find_global_y;
    int dist_find_global_z;

    int reduction_global;
    int reduction_local;
    int reduced_elems_count;

    int update_radius;

    cl::Context d_context;
    cl::CommandQueue d_queue;
    cl::Program program;

    cl::Kernel calc_dist_kernel;
    cl::Kernel local_min_kernel;
    cl::Kernel global_min_kernel;
    cl::Kernel update_kernel;
    cl::Kernel calc_all_dist_kernel;

    /* OCL BUFFERS */
    cl::Buffer d_som;
    cl::Buffer input_vectors;
    cl::Buffer distances;

    cl::Buffer out_min_indexes;
    cl::Buffer out_min_values;

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

    float* local_distances; // N-closest neighbours finding

    void verify_distance_calculation(float* input_1, float* input_2,
                                     int length_1, int length_2,
                                     int vector_size, float* output);

#ifdef OPENCL_PROFILE
    void initProfiling();
    void showProfilingInfo();

    std::vector<cl::Event> calc_dist_events;
    std::vector<cl::Event> local_min_events;
    std::vector<cl::Event> global_min_events;
    std::vector<cl::Event> update_events;
#endif

};

#endif
