#ifndef OCLSOM2D_H
#define OCLSOM2D_H

#include <cstdlib>

void ocl_som2d_test();

class som_data
{
public:
    som_data(int x, int y, int z);
    ~som_data();

    int x, y, z;
    int size;
    float* data;
};



#endif
