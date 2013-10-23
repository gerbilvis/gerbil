/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef KERNELWRAP_CU
#define KERNELWRAP_CU

#include<cuda.h>
#include<stdio.h>
#include "kernels.h"
#include "defs.h"
#include "utilsGPU.h"

void dist1Wrap(const ocl_matrix& dq, const ocl_matrix& dx, ocl_matrix& dD){

    cl::Kernel& dist1Kernel = OclContextHolder::dist1Kernel;

    cl::NDRange local(BLOCK_SIZE, BLOCK_SIZE);

    unint todoX, todoY, numDoneX, numDoneY;

    numDoneX = 0;
    while ( numDoneX < dx.pr ){
      todoX = MIN( dx.pr - numDoneX, MAX_BS*BLOCK_SIZE );

      int global_x = todoX;

      numDoneY = 0;
      while( numDoneY < dq.pr ){

        todoY = MIN( dq.pr - numDoneY, MAX_BS*BLOCK_SIZE );
        int global_y = todoY;

        cl::NDRange global(global_x, global_y);

        //dist1Kernel<<<grid,block>>>(dq, numDoneY, dx, numDoneX, dD);

        cl::Context& context = OclContextHolder::context;
        cl::CommandQueue& queue = OclContextHolder::queue;

        dist1Kernel.setArg(0, dq.mat);
        dist1Kernel.setArg(1, dq.r);
        dist1Kernel.setArg(2, dq.c);
        dist1Kernel.setArg(3, dq.pr);
        dist1Kernel.setArg(4, dq.pc);
        dist1Kernel.setArg(5, dq.ld);
        dist1Kernel.setArg(6, numDoneY);
        dist1Kernel.setArg(7, dx.mat);
        dist1Kernel.setArg(8, dx.r);
        dist1Kernel.setArg(9, dx.c);
        dist1Kernel.setArg(10, dx.pr);
        dist1Kernel.setArg(11, dx.pc);
        dist1Kernel.setArg(12, dx.ld);
        dist1Kernel.setArg(13, numDoneX);
        dist1Kernel.setArg(14, dD.mat);
        dist1Kernel.setArg(15, dD.r);
        dist1Kernel.setArg(16, dD.c);
        dist1Kernel.setArg(17, dD.pr);
        dist1Kernel.setArg(18, dD.pc);
        dist1Kernel.setArg(19, dD.ld);

        queue.enqueueNDRangeKernel(dist1Kernel, cl::NullRange, global, local);

        numDoneY += todoY;
      }
      numDoneX += todoX;
    }

/*
  dim3 block(BLOCK_SIZE,BLOCK_SIZE);
  dim3 grid;
  
  unint todoX, todoY, numDoneX, numDoneY;

  numDoneX = 0;
  while ( numDoneX < dx.pr ){
    todoX = MIN( dx.pr - numDoneX, MAX_BS*BLOCK_SIZE );
    grid.x = todoX/BLOCK_SIZE;
    numDoneY = 0;
    while( numDoneY < dq.pr ){
      todoY = MIN( dq.pr - numDoneY, MAX_BS*BLOCK_SIZE );
      grid.y = todoY/BLOCK_SIZE;
      dist1Kernel<<<grid,block>>>(dq, numDoneY, dx, numDoneX, dD);
      numDoneY += todoY;
    }
    numDoneX += todoX;
  }

  cudaThreadSynchronize();
*/
}


void findRangeWrap(const ocl_matrix& dD, cl::Buffer& dranges, unint cntWant){
/*  dim3 block(4*BLOCK_SIZE,BLOCK_SIZE/4);
  dim3 grid(1,4*(dD.pr/BLOCK_SIZE));
  unint numDone, todo;
  
  numDone=0;
  while( numDone < dD.pr ){
    todo = MIN ( dD.pr - numDone, MAX_BS*BLOCK_SIZE/4 );
    grid.y = 4*(todo/BLOCK_SIZE);
    findRangeKernel<<<grid,block>>>(dD, numDone, dranges, cntWant);
    numDone += todo;
  }
  cudaThreadSynchronize();*/
}


void rangeSearchWrap(const ocl_matrix& dD, cl::Buffer& dranges, ocl_charMatrix& dir){
/*  dim3 block(BLOCK_SIZE,BLOCK_SIZE);
  dim3 grid;

  unint todoX, todoY, numDoneX, numDoneY;
  
  numDoneX = 0;
  while ( numDoneX < dD.pc ){
    todoX = MIN( dD.pc - numDoneX, MAX_BS*BLOCK_SIZE );
    grid.x = todoX/BLOCK_SIZE;
    numDoneY = 0;
    while( numDoneY < dD.pr ){
      todoY = MIN( dD.pr - numDoneY, MAX_BS*BLOCK_SIZE );
      grid.y = todoY/BLOCK_SIZE;
      rangeSearchKernel<<<grid,block>>>(dD, numDoneX, numDoneY, dranges, dir);
      numDoneY += todoY;
    }
    numDoneX += todoX;
  }

  cudaThreadSynchronize();*/
}

void nnWrap(const ocl_matrix& dq, const ocl_matrix& dx,
            cl::Buffer& dMins, cl::Buffer& dMinIDs){
//void nnWrap(const matrix dq, const matrix dx, real *dMins, unint *dMinIDs){
/*  dim3 block(BLOCK_SIZE,BLOCK_SIZE);
  dim3 grid;
  unint numDone, todo;
  
  grid.x = 1;

  numDone = 0;
  while( numDone < dq.pr ){
    todo = MIN( dq.pr - numDone, MAX_BS*BLOCK_SIZE );
    grid.y = todo/BLOCK_SIZE;
    nnKernel<<<grid,block>>>(dq,numDone,dx,dMins,dMinIDs);
    numDone += todo;
  }
  cudaThreadSynchronize();
*/
}


void knnWrap(const ocl_matrix& dq, const ocl_matrix& dx, ocl_matrix& dMins, ocl_intMatrix& dMinIDs){
/*  dim3 block(BLOCK_SIZE,BLOCK_SIZE);
  dim3 grid;
  unint numDone, todo;
  
  grid.x = 1;

  numDone = 0;
  while( numDone < dq.pr ){
    todo = MIN( dq.pr - numDone, MAX_BS*BLOCK_SIZE );
    grid.y = todo/BLOCK_SIZE;
    knnKernel<<<grid,block>>>(dq,numDone,dx,dMins,dMinIDs);
    numDone += todo;
  }
  cudaThreadSynchronize();
*/
}


void planNNWrap(const matrix dq, const unint *dqMap, const matrix dx, const intMatrix dxMap, real *dMins, unint *dMinIDs, compPlan dcP, unint compLength){
/* NEVER USED
  dim3 block(BLOCK_SIZE,BLOCK_SIZE);
  dim3 grid;
  unint todo;

  grid.x = 1;
  unint numDone = 0;
  while( numDone<compLength ){
    todo = MIN( (compLength-numDone) , MAX_BS*BLOCK_SIZE );
    grid.y = todo/BLOCK_SIZE;
    planNNKernel<<<grid,block>>>(dq,dqMap,dx,dxMap,dMins,dMinIDs,dcP,numDone);
    numDone += todo;
  }
  cudaThreadSynchronize();*/
}


//void planKNNWrap(const matrix dq, const unint *dqMap, const matrix dx, const intMatrix dxMap, matrix dMins, intMatrix dMinIDs, compPlan dcP, unint compLength){
void planKNNWrap(const ocl_matrix& dq, cl::Buffer& dqMap, const ocl_matrix& dx,
                 const ocl_intMatrix& dxMap, ocl_matrix& dMins,
                 ocl_intMatrix& dMinIDs, ocl_compPlan& dcP, unint compLength){
/*
  dim3 block(BLOCK_SIZE,BLOCK_SIZE);
  dim3 grid;
  unint todo;

  grid.x = 1;
  unint numDone = 0;
  while( numDone<compLength ){
    todo = MIN( (compLength-numDone) , MAX_BS*BLOCK_SIZE );
    grid.y = todo/BLOCK_SIZE;
    planKNNKernel<<<grid,block>>>(dq,dqMap,dx,dxMap,dMins,dMinIDs,dcP,numDone);
    numDone += todo;
  }
  cudaThreadSynchronize();
*/
}



//void rangeCountWrap(const matrix dq, const matrix dx, real *dranges, unint *dcounts){
void rangeCountWrap(const ocl_matrix& dq, const ocl_matrix& dx, cl::Buffer& dranges, cl::Buffer& dcounts){
/*  dim3 block(BLOCK_SIZE,BLOCK_SIZE);
  dim3 grid;
  unint numDone, todo;

  grid.x=1;

  numDone = 0;
  while( numDone < dq.pr ){
    todo = MIN( dq.pr - numDone, MAX_BS*BLOCK_SIZE );
    grid.y = todo/BLOCK_SIZE;
    rangeCountKernel<<<grid,block>>>(dq,numDone,dx,dranges,dcounts);
    numDone += todo;
  }
  cudaThreadSynchronize();*/
}

#endif
