#ifndef OCLSOMDATA_H
#define OCLSOMDATA_H

class ocl_som_data
{
public:
    ocl_som_data(int x, int y, int neuron_size);
    ~ocl_som_data();

    int x, y, neuron_size;
    int size;
    float* data;
};


#endif //OCLSOMDATA_H
