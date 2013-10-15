#ifndef OCLSOM3D_H
#define OCLSOM3D_H

#include <cstdlib>
#include "som3d.h"
#include "ocl_som_types.h"
#include "ocl_utils.h"

#include "ocl_generic_som.h"

//#define DEBUG_MODE

class Ocl_SOM3d : public SOM3d, public Ocl_GenericSOM
{
public:
    Ocl_SOM3d(const vole::EdgeDetectionConfig &conf, int dimension,
              std::vector<multi_img_base::BandDesc> meta);
    Ocl_SOM3d(const vole::EdgeDetectionConfig &conf, const multi_img &data,
              std::vector<multi_img_base::BandDesc> meta);

    ~Ocl_SOM3d();

    SOM::iterator identifyWinnerNeuron(const multi_img::Pixel &inputVec);
    int updateNeighborhood(iterator &neuron, const multi_img::Pixel &input,
                           double sigma, double learnRate);

    void notifyTrainingStart();
    void notifyTrainingEnd();

private:

    void uploadDataToDevice();
    void downloadDataFromDevice();
};

#endif
