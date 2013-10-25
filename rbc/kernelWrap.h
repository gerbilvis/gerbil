/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */
#ifndef KERNELWRAP_H
#define KERNELWRAP_H

#include "defs.h"

void dist1Wrap(const ocl_matrix&,const ocl_matrix&,ocl_matrix&);
void kMinsWrap(matrix,matrix,intMatrix);
//void findRangeWrap(const matrix,real*,unint);
void findRangeWrap(const ocl_matrix&,cl::Buffer&,unint);

//void rangeSearchWrap(const matrix,const real*,charMatrix);

void rangeSearchWrap(const ocl_matrix&,cl::Buffer&,ocl_charMatrix&);

//void nnWrap(const matrix,const matrix,real*,unint*);

void nnWrap(const ocl_matrix&,const ocl_matrix&,cl::Buffer&,cl::Buffer&);

//void knnWrap(const matrix,const matrix,matrix,intMatrix);
void knnWrap(const ocl_matrix&,const ocl_matrix&,ocl_matrix&,ocl_intMatrix&);


//void rangeCountWrap(const matrix,const matrix,real*,unint*);
void rangeCountWrap(const ocl_matrix&,const ocl_matrix&, cl::Buffer&, cl::Buffer&);



void planNNWrap(const matrix,const unint*,const matrix,const intMatrix,real*,unint*,compPlan,unint);


//void planKNNWrap(const matrix,const unint*,const matrix,const intMatrix,matrix,intMatrix,compPlan,unint);
void planKNNWrap(const ocl_matrix&,const cl::Buffer&,const ocl_matrix&,
                 const ocl_intMatrix&,ocl_matrix&,ocl_intMatrix&,
                 const ocl_compPlan&,unint);

#endif
