#ifndef OCLSOMDATA_H
#define OCLSOMDATA_H

class ocl_som_data
{
public:
    ocl_som_data(int x, int y, int z, int neuron_size);
    ~ocl_som_data();

    int x, y, z, neuron_size;
    int size;
    float* data;
};


#endif //OCLSOMDATA_H
