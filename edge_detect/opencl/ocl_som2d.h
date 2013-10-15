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


private:

    void uploadDataToDevice();
    void downloadDataFromDevice();
  //  void uploadTrainingVectors();

};

#endif
