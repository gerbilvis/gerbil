/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef RBC_OPS_H
#define RBC_OPS_H

#include "defs.h"


void buildRBC(const matrix,ocl_rbcStruct*,unint, unint);

//NEVER USED void queryRBC(const matrix,const rbcStruct,unint*,real*);

//void kqueryRBC(const matrix,const rbcStruct,intMatrix,matrix);
void kqueryRBC(const matrix,const ocl_rbcStruct,intMatrix,matrix);

void destroyRBC(ocl_rbcStruct*);
void distSubMat(ocl_matrix&,ocl_matrix&,ocl_matrix&,unint,unint);
void computeReps(const ocl_matrix&, const ocl_matrix&,unint*,real*);
void computeRadii(unint*,real*,real*,unint,unint);
void computeCounts(unint*,unint,unint*);
void buildQMap(matrix,unint*,unint*,unint,unint*);
void idIntersection(charMatrix);
void fullIntersection(charMatrix);
void initCompPlan(ocl_compPlan*,charMatrix,unint*,unint*,unint);
void freeCompPlan(ocl_compPlan*);

//NEVER USED
//void computeNNs(matrix,intMatrix,matrix,unint*,compPlan,unint*,real*,unint);
//void computeNNs(ocl_matrix&,ocl_intMatrix&,ocl_matrix&,
//                cl::Buffer&,ocl_compPlan&,unint*,real*,unint);

//void computeKNNs(matrix,intMatrix,matrix,unint*,compPlan,intMatrix,matrix,unint);
void computeKNNs(const ocl_matrix&, const ocl_intMatrix&,const ocl_matrix&,cl::Buffer&,ocl_compPlan&,intMatrix,matrix,unint);

void setupReps(matrix,ocl_rbcStruct*,unint);

#endif
