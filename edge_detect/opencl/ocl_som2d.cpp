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
      Ocl_GenericSOM(conf.sidelength, conf.sidelength,
                     1, data.size(), "")
{
}

Ocl_SOM2d::Ocl_SOM2d(const vole::EdgeDetectionConfig &conf,
                     int dimension,
                     std::vector<multi_img_base::BandDesc> meta)
    : SOM2d(conf, dimension, meta),
      Ocl_GenericSOM(conf.sidelength, conf.sidelength,
                     1, dimension, "")
{
}

Ocl_SOM2d::~Ocl_SOM2d()
{
}


void Ocl_SOM2d::uploadDataToDevice()
{
    clearSomData();

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

    Ocl_GenericSOM::uploadDataToDevice();
}

void Ocl_SOM2d::downloadDataFromDevice()
{
    Ocl_GenericSOM::downloadDataFromDevice();

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

void Ocl_SOM2d::notifyTrainingStart()
{
    uploadDataToDevice();
}

void Ocl_SOM2d::notifyTrainingEnd()
{
    std::cout << "notify training end!" << std::endl;

    uploadTrainingVectors();

    train();

    downloadDataFromDevice();

    calculateAllDistances();

}

SOM::iterator Ocl_SOM2d::identifyWinnerNeuron(const multi_img::Pixel &inputVec)
{
    training_vectors.push_back(inputVec);
    return SOM::iterator(new Iterator2d(this, 0, 0));
}

int Ocl_SOM2d::updateNeighborhood(iterator &neuron,
                                  const multi_img::Pixel &input,
                                  double sigma, double learnRate)
{
    training_params.push_back(std::pair<double, double>(sigma, learnRate));
    return 42;
}


static bool sortpair(std::pair<double, SOM::iterator>& i,
                     std::pair<double, SOM::iterator>& j) {
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

SOM::DistanceCache* Ocl_SOM2d::createDistanceCache(int img_height, int img_width)
{
    return new Ocl_DistanceCache(*this);
}



void Ocl_DistanceCache::preload(const multi_img &image)
{
    int som_size = som.get2dWidth() * som.get2dWidth();
    image_size = image.width * image.height;

    if(host_distances)
        delete[] host_distances;

    host_distances = new float[som_size * image_size];

    som.calculateAllDistances(image, host_distances);

    preloaded = true;
}

float Ocl_DistanceCache::getDistance(int index, SOM::iterator& iterator)
{    
    if(preloaded)
    {
        cv::Point point = iterator.get2dCoordinates();
        int som_index = point.y * som_width + point.x;

        return host_distances[index * som_size + som_index];
    }
    else
    {
        return 0.f;
    }
}

void Ocl_DistanceCache::closestN(int index,
                         std::vector<std::pair<double, SOM::iterator> > &heap)
{
    // initialize with maximum values
    for (int i = 0; i < heap.size(); ++i)
        heap[i].first = std::numeric_limits<double>::max();

    // find closest Neurons to inputVec in the SOM
    // iterate over all neurons in grid
    for (SOM::iterator neuron = som.begin(); neuron != som.end(); ++neuron)
    {
        //double dist = distfun->getSimilarity(*neuron, inputVec);

        double dist = getDistance(index, neuron);

        /* compare current distance with the maximum of the N shortest
         * found distances */
        if (dist < heap[0].first) {
            // remove max. value in heap
            std::pop_heap(heap.begin(), heap.end(), sortpair);

            /* max element is now on position "back" and should be popped
             * instead we overwrite it directly with the new element */
            std::pair<double, SOM::iterator> &back = heap.back();
            back.first = dist;
            back.second = neuron;

            std::push_heap(heap.begin(), heap.end(), sortpair);
        }
    }

    assert(heap[0].first != std::numeric_limits<double>::max());
    std::sort_heap(heap.begin(), heap.end(), sortpair); // sort ascending
}

