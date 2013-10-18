#ifndef OCLSOM2D_H
#define OCLSOM2D_H

#include <cstdlib>
#include "som2d.h"
#include "ocl_som_types.h"
#include "ocl_utils.h"

#include "ocl_generic_som.h"

void ocl_som2d_test();

//#define DEBUG_MODE

class Ocl_SOM2d : public SOM2d, public Ocl_GenericSOM
{
public:
    Ocl_SOM2d(const vole::EdgeDetectionConfig &conf, int dimension,
              std::vector<multi_img_base::BandDesc> meta);
    Ocl_SOM2d(const vole::EdgeDetectionConfig &conf, const multi_img &data,
              std::vector<multi_img_base::BandDesc> meta);

    ~Ocl_SOM2d();

    SOM::iterator identifyWinnerNeuron(const multi_img::Pixel &inputVec);
    int updateNeighborhood(iterator &neuron, const multi_img::Pixel &input,
                           double sigma, double learnRate);

    void notifyTrainingStart();
    void notifyTrainingEnd();
    void closestN(const multi_img::Pixel &inputVec,
                  std::vector<std::pair<double, iterator> > &coords);

    SOM::DistanceCache* createDistanceCache(int img_height, int img_width);

private:

    void uploadDataToDevice();
    void downloadDataFromDevice();
  //  void uploadTrainingVectors();

};


class Ocl_DistanceCache : public SOM2d::DistanceCache
{
public:

    Ocl_DistanceCache(Ocl_SOM2d& som) : som(som), host_distances(0),
                                image_size(0),
                                som_size(som.get2dWidth() * som.get2dHeight()),
                                som_width(som.get2dWidth()){}

    ~Ocl_DistanceCache()
    {
        if(host_distances) delete[] host_distances;
    }

    void preload(const multi_img &image);
    float getDistance(int index, SOM::iterator& iterator);
    void closestN(int index, std::vector<std::pair<double, SOM::iterator> > &heap);

private:
    Ocl_SOM2d& som;
    float* host_distances;
    int image_size;
    int som_size;
    int som_width;
};

#endif
