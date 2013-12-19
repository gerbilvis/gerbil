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

//void planKNNWrap(const ocl_matrix& dq, const ocl_matrix& dx,
//                 const ocl_intMatrix& dxMap, const cl::Buffer& repsIDs,
//                 ocl_matrix& dMins, ocl_intMatrix& dMinIDs);


void meanshiftPlanKNNWrap(const ocl_matrix& dq, const cl::Buffer& dqMap,
                          const ocl_matrix& dx, const ocl_intMatrix& dxMap,
                          const ocl_compPlan& dcP, const cl::Buffer& windows,
                          const cl::Buffer& weights,
                          cl::Buffer& selectedPoints,
                          cl::Buffer& selectedDistances,
                          cl::Buffer& selectedPointsNum,
                          cl::Buffer& hmodes,
                          unint maxPointsNum, unint startPos, unint length);//unint compLength);

void meanshiftMeanWrap(const ocl_matrix& input,
                       const cl::Buffer& selectedPoints,
                       const cl::Buffer& selectedDistances,
                       const cl::Buffer& selectedPointsNum,
                       const cl::Buffer& windows,
                       const cl::Buffer& weights,
                       unint maxPointsNum,
                       ocl_matrix& output);

void meanshiftWeightsWrap(const cl::Buffer& pilots, cl::Buffer& weights,
                          unint size, unint dimensionality);

void simpleDistanceKernelWrap(const ocl_matrix& in_1, const ocl_matrix& in_2,
                              cl::Buffer& out);

void clearRealKernelWrap(cl::Buffer& buffer, unint size);
void clearIntKernelWrap(cl::Buffer& buffer, unint size);

void initIndexesKernelWrap(cl::Buffer& buffer, unint size);

void meanshiftPackKernelWrap(const ocl_matrix& prev_iteration,
                             const ocl_matrix& curr_iteration,
                             ocl_matrix& next_iteration,
                             ocl_matrix& final_modes,
                             cl::Buffer& old_indexes,
                             cl::Buffer& new_indexes,
                             unint current_size, unint& result_size,
                             cl::Buffer iterationMap,
                             cl::Buffer partial_hmodes,
                             cl::Buffer final_hmodes,
                             unint iterationNum);

//void findRangeWrap(const matrix,real*,unint);
//void rangeSearchWrap(const matrix,const real*,charMatrix);
//void nnWrap(const matrix,const matrix,real*,unint*);
//void knnWrap(const matrix,const matrix,matrix,intMatrix);
//void rangeCountWrap(const matrix,const matrix,real*,unint*);
//void planKNNWrap(const matrix,const unint*,const matrix,const intMatrix,matrix,intMatrix,compPlan,unint);

#endif
