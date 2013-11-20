/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */
#ifndef KERNELWRAP_H
#define KERNELWRAP_H

#include "defs.h"

void dist1Wrap(const ocl_matrix&, const ocl_matrix&, ocl_matrix&, unint);

void kMinsWrap(matrix, matrix, intMatrix);

void findRangeWrap(const ocl_matrix&, cl::Buffer&,unint,unint);

void rangeSearchWrap(const ocl_matrix&, cl::Buffer&, ocl_charMatrix&);

void nnWrap(const ocl_matrix&, const ocl_matrix&, cl::Buffer&, cl::Buffer&);

void knnWrap(const ocl_matrix&, const ocl_matrix&, ocl_matrix&, ocl_intMatrix&);

void rangeCountWrap(const ocl_matrix&, const ocl_matrix&,
                    cl::Buffer&, cl::Buffer&);

void planNNWrap(const matrix, const unint*, const matrix,
                const intMatrix, real*,unint*, compPlan,unint);

void planKNNWrap(const ocl_matrix&,const cl::Buffer&,const ocl_matrix&,
                 const ocl_intMatrix&,ocl_matrix&,ocl_intMatrix&,
                 const ocl_compPlan&,unint);

void meanshiftPlanKNNWrap(const ocl_matrix& dq, const cl::Buffer& dqMap,
                          const ocl_matrix& dx, const ocl_intMatrix& dxMap,
                          const ocl_compPlan& dcP, const cl::Buffer& windows,
                          cl::Buffer& outputMeans,
                          cl::Buffer& outputMeansNum, cl::Buffer& newWindows,
                          int maxPointsNum, unint compLength);

void meanshiftMeanWrap(const ocl_matrix& input,
                       const cl::Buffer& selectedPoints,
                       const cl::Buffer& selectedPointsNum,
                       unint maxPointsNum,
                       ocl_matrix& output);


//void findRangeWrap(const matrix,real*,unint);
//void rangeSearchWrap(const matrix,const real*,charMatrix);
//void nnWrap(const matrix,const matrix,real*,unint*);
//void knnWrap(const matrix,const matrix,matrix,intMatrix);
//void rangeCountWrap(const matrix,const matrix,real*,unint*);
//void planKNNWrap(const matrix,const unint*,const matrix,const intMatrix,matrix,intMatrix,compPlan,unint);

#endif
