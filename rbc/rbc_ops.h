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

void simpleKqueryRBC(const matrix q, const ocl_rbcStruct rbcS,
               intMatrix NNs, matrix NNdists);

void meanshiftKQueryRBC(const matrix input, const ocl_rbcStruct rbcS,
                        ocl_matrix output_means, const cl::Buffer &pilots,
                        cl::Buffer& newPilots, int maxPointsNum);

void computePilots(const matrix q, const cl::Buffer &repsPilots,
                   const ocl_rbcStruct& rbcS, cl::Buffer &pilots);

void destroyRBC(ocl_rbcStruct*);
void distSubMat(ocl_matrix&,ocl_matrix&,ocl_matrix&,unint,unint);

void computeReps(const ocl_matrix&, const ocl_matrix&,unint*,real*);

void computeReps(const ocl_matrix& dq, const ocl_matrix& dr,
                 cl::Buffer& dMinIDs, cl::Buffer& dMins);

void computeRepsNoHost(const ocl_matrix& dq, const ocl_matrix& dr,
                       cl::Buffer& indexes);

void bindPilots(const cl::Buffer& indexes, const cl::Buffer& repsPilots,
                cl::Buffer& pilots, int pilots_size);


void computeRadii(unint*,real*,real*,unint,unint);
void computeCounts(unint*,unint,unint*);
void buildQMap(matrix,unint*,unint*,unint,unint*);
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

/** simple version */
void computeKNNs(const ocl_matrix& dx, const ocl_intMatrix& dxMap,
                 const ocl_matrix& dq, const cl::Buffer& repIDs,
                 intMatrix NNs, matrix NNdists);


void meanshiftComputeKNNs(const ocl_matrix& dx, const ocl_intMatrix& dxMap,
                          const ocl_matrix& dq, const cl::Buffer& dqMap,
                          const ocl_compPlan& dcP,
                          cl::Buffer windows, cl::Buffer outputMeans,
                          cl::Buffer outputMeansNums, cl::Buffer newWindows,
                          int maxPointsNum, unint compLength);

void setupReps(matrix,ocl_rbcStruct*,unint);

#endif
