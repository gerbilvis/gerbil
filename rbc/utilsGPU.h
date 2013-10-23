/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef UTILSGPU_H
#define UTILSGPU_H

#include "defs.h"
#include<cuda.h>


class OclContextHolder
{
public:
    static cl::Context context;
    static cl::CommandQueue queue;

    static cl::Kernel dist1Kernel;

    static void oclInit();
};


void copyAndMove(ocl_matrix*,const matrix*);
void copyAndMoveI(ocl_intMatrix*,const intMatrix*);
void copyAndMoveC(ocl_charMatrix*,const charMatrix*);

//void copyAndMove(matrix*,const matrix*);
//void copyAndMoveI(intMatrix*,const intMatrix*);
//void copyAndMoveC(charMatrix*,const charMatrix*);

//void checkErr(cudaError_t);
//void checkErr(char*,cudaError_t );

void checkErr(cl_int);
void checkErr(char*, cl_int);

#endif
