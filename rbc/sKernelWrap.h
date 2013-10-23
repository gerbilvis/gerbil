/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef SKERNELWRAP_H
#define SKERNELWRAP_H

#include "defs.h"

//void getCountsWrap(unint*,charMatrix,intMatrix);

void getCountsWrap(cl::Buffer&,ocl_charMatrix&,ocl_intMatrix&);
void buildMapWrap(ocl_intMatrix&,ocl_charMatrix&,ocl_intMatrix&,unint);
void sumWrap(ocl_charMatrix&,ocl_intMatrix&);


#endif
