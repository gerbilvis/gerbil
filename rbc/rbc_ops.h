/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef RBC_OPS_H
#define RBC_OPS_H

#include "defs.h"


void buildRBC(const matrix x,
              ocl_rbcStruct* rbcS,
              unint numReps,
              unint pointsPerRep,
              cl::Buffer pilots = cl::Buffer(),
              int threshold = 0);

//NEVER USED void queryRBC(const matrix,const rbcStruct,unint*,real*);

void kqueryRBC(const matrix q, const ocl_rbcStruct rbcS,
               intMatrix NNs, matrix NNdists);

//void simpleKqueryRBC(const matrix q, const ocl_rbcStruct rbcS,
//               intMatrix NNs, matrix NNdists);

void meanshiftKQueryRBC(const ocl_matrix input, const ocl_rbcStruct rbcS,
                        const cl::Buffer &pilots,
                        const cl::Buffer &weights,
                        cl::Buffer selectedPoints,
                        cl::Buffer selectedDistances,
                        cl::Buffer selectedPointsNum,
                        cl::Buffer hmodes,
                        int maxPointsNum);

void computePilotsAndWeights(const matrix q,
                             const cl::Buffer &repsPilots,
                             const cl::Buffer &repsWeights,
                             const ocl_rbcStruct& rbcS,
                             cl::Buffer &pilots,
                             cl::Buffer &weights);

void destroyRBC(ocl_rbcStruct*);
void distSubMat(ocl_matrix&,ocl_matrix&,ocl_matrix&,unint,unint);

void computeReps(const ocl_matrix&, const ocl_matrix&,unint*,real*);

void computeReps(const ocl_matrix& dq, const ocl_matrix& dr,
                 cl::Buffer& dMinIDs, cl::Buffer& dMins);

void computeRepsNoHost(const ocl_matrix& dq, const ocl_matrix& dr,
                       cl::Buffer& indexes);

void bindPilotsAndWeights(const cl::Buffer &indexes,
                          const cl::Buffer &repsPilots,
                          const cl::Buffer &repsWeights,
                          cl::Buffer &pilots, cl::Buffer &weights,
                          int size);


void computeRadii(unint*,real*,real*,unint,unint);
void computeCounts(unint*,unint,unint*);
void buildQMap(unint,unint*,unint*,unint,unint*);
void idIntersection(charMatrix);
void fullIntersection(charMatrix);
void initCompPlan(ocl_compPlan*, const charMatrix,
                  const unint*, const unint*, unint);
void freeCompPlan(ocl_compPlan*);

//NEVER USED
//void computeNNs(matrix,intMatrix,matrix,unint*,compPlan,unint*,real*,unint);
//void computeNNs(ocl_matrix&,ocl_intMatrix&,ocl_matrix&,
//                cl::Buffer&,ocl_compPlan&,unint*,real*,unint);

//void computeKNNs(matrix,intMatrix,matrix,unint*,compPlan,intMatrix,matrix,unint);
void computeKNNs(const ocl_matrix&, const ocl_intMatrix&,
                 const ocl_matrix&, const cl::Buffer&,
                 const ocl_compPlan&, intMatrix, matrix, unint);

///** simple version */
//void computeKNNs(const ocl_matrix& dx, const ocl_intMatrix& dxMap,
//                 const ocl_matrix& dq, const cl::Buffer& repIDs,
//                 intMatrix NNs, matrix NNdists);


void meanshiftComputeKNNs(const ocl_matrix& dx, const ocl_intMatrix& dxMap,
                          const ocl_matrix& dq, const cl::Buffer& dqMap,
                          const ocl_compPlan& dcP,
                          cl::Buffer windows, cl::Buffer weights,
                          cl::Buffer selectedPoints,
                          cl::Buffer selectedDistances,
                          cl::Buffer selectedPointsNums,
                          cl::Buffer hmodes,
                          int maxPointsNum,// unint compLength);
                          unint startPos, unint length);

void setupReps(matrix,ocl_rbcStruct*,unint);

#endif
