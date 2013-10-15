#include "ocl_som3d.h"

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

Ocl_SOM3d::Ocl_SOM3d(const vole::EdgeDetectionConfig &conf,
                     const multi_img &data,
                     std::vector<multi_img_base::BandDesc> meta)
    : SOM3d(conf, data, meta),
      Ocl_GenericSOM(conf.sidelength, conf.sidelength,
                     conf.sidelength, data.size(), "-DSOM_3D")
{
}

Ocl_SOM3d::Ocl_SOM3d(const vole::EdgeDetectionConfig &conf,
                     int dimension,
                     std::vector<multi_img_base::BandDesc> meta)
    : SOM3d(conf, dimension, meta),
      Ocl_GenericSOM(conf.sidelength, conf.sidelength,
                     conf.sidelength, dimension, "-DSOM_3D")
{
}

Ocl_SOM3d::~Ocl_SOM3d()
{
}

void Ocl_SOM3d::uploadDataToDevice()
{
    clearSomData();

    int field_size = d_data.x * d_data.y * d_data.neuron_size;

    for(int i = 0; i < neurons.size(); ++i)
    {
        Field& field = neurons[i];

        float* field_ptr = d_data.data + field_size * i;

        for(int j = 0; j < field.size(); ++j)
        {
            Row& row = field[j];
            float* row_ptr = field_ptr + d_data.x * d_data.neuron_size * j;

            for(int k = 0; k < row.size(); ++k)
            {
                Neuron& neuron = row[k];
                float* neuron_ptr = row_ptr + d_data.neuron_size * k;
                std::copy(neuron.begin(), neuron.end(), neuron_ptr);
            }
        }
    }

    Ocl_GenericSOM::uploadDataToDevice();
}

void Ocl_SOM3d::downloadDataFromDevice()
{
    Ocl_GenericSOM::downloadDataFromDevice();

    int field_size = d_data.x * d_data.y * d_data.neuron_size;

    for(int i = 0; i < neurons.size(); ++i)
    {
        Field& field = neurons[i];

        float* field_ptr = d_data.data + field_size * i;

        for(int j = 0; j < field.size(); ++j)
        {
            Row& row = field[j];
            float* row_ptr = field_ptr + d_data.x * d_data.neuron_size * j;

            for(int k = 0; k < row.size(); ++k)
            {
                Neuron& neuron = row[k];
                float* neuron_ptr = row_ptr + d_data.neuron_size * k;

                for(int l = 0; l < neuron.size(); ++l)
                {
                    float* ptr = neuron_ptr + l;
                    neuron[l] = *ptr;
                }
            }
        }
    }
}

void Ocl_SOM3d::notifyTrainingStart()
{
    uploadDataToDevice();
}

void Ocl_SOM3d::notifyTrainingEnd()
{
    std::cout << "notify training end!" << std::endl;

    uploadTrainingVectors();

    train();

    downloadDataFromDevice();
}

SOM::iterator Ocl_SOM3d::identifyWinnerNeuron(const multi_img::Pixel &inputVec)
{
    training_vectors.push_back(inputVec);
    return SOM::iterator(new Iterator3d(this, 0, 0, 0));
}

int Ocl_SOM3d::updateNeighborhood(iterator &neuron,
                                  const multi_img::Pixel &input,
                                  double sigma, double learnRate)
{
    training_params.push_back(std::pair<double, double>(sigma, learnRate));
    return 42;
}
