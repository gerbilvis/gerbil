/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef UTILS_H
#define UTILS_H

#include "defs.h"

void swap(unint*,unint*);
void randPerm(unint,unint*);
void subRandPerm(unint,unint,unint*);
unint randBetween(unint,unint);
void printMat(matrix);
void printMatWithIDs(matrix,unint*);
void printCharMat(charMatrix);
void printIntMat(intMatrix);
void printVector(real*,unint);
void copyVector(real*,real*,unint);
real distVec(matrix,matrix,unint,unint);
double timeDiff(struct timeval,struct timeval);
void copyMat(matrix*,matrix*);

void initMat(matrix*,unint,unint);
void initMat(ocl_matrix*,unint,unint);
void initIntMat(intMatrix*,unint,unint);
void initIntMat(ocl_intMatrix*,unint,unint);


size_t sizeOfMatB(matrix);
size_t sizeOfMatB(ocl_matrix&);

size_t sizeOfIntMatB(intMatrix);
size_t sizeOfIntMatB(ocl_intMatrix&);


size_t sizeOfMat(matrix);
size_t sizeOfIntMat(intMatrix);
#endif
