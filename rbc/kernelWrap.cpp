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

    cl::Kernel& findRangeKernel = OclContextHolder::findRangeKernel;

    cl::NDRange local(4*BLOCK_SIZE,BLOCK_SIZE/4);

    unint numDone, todo;

    numDone=0;
    while( numDone < dD.pr ){
      todo = MIN ( dD.pr - numDone, MAX_BS*BLOCK_SIZE/4 );
      //grid.y = 4*(todo/BLOCK_SIZE);

      cl::NDRange global(4*BLOCK_SIZE, todo);

      findRangeKernel.setArg(0, dD.mat);
      findRangeKernel.setArg(1, dD.r);
      findRangeKernel.setArg(2, dD.c);
      findRangeKernel.setArg(3, dD.pr);
      findRangeKernel.setArg(4, dD.pc);
      findRangeKernel.setArg(5, dD.ld);
      findRangeKernel.setArg(6, numDone);
      findRangeKernel.setArg(7, dranges);
      findRangeKernel.setArg(8, cntWant);

      cl::CommandQueue& queue = OclContextHolder::queue;

      queue.enqueueNDRangeKernel(findRangeKernel, cl::NullRange,
                                 global, local);


      numDone += todo;
    }



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

    cl::Kernel& rangeSearchKernel = OclContextHolder::rangeSearchKernel;

    cl::NDRange local(BLOCK_SIZE,BLOCK_SIZE);

    unint todoX, todoY, numDoneX, numDoneY;

    numDoneX = 0;
    while ( numDoneX < dD.pc ){
      todoX = MIN( dD.pc - numDoneX, MAX_BS*BLOCK_SIZE );
      //grid.x = todoX/BLOCK_SIZE;
      numDoneY = 0;
      while( numDoneY < dD.pr ){
        todoY = MIN( dD.pr - numDoneY, MAX_BS*BLOCK_SIZE );
        //grid.y = todoY/BLOCK_SIZE;

        cl::NDRange global(todoX, todoY);

        rangeSearchKernel.setArg(0, dD.mat);
        rangeSearchKernel.setArg(1, dD.r);
        rangeSearchKernel.setArg(2, dD.c);
        rangeSearchKernel.setArg(3, dD.pr);
        rangeSearchKernel.setArg(4, dD.pc);
        rangeSearchKernel.setArg(5, dD.ld);
        rangeSearchKernel.setArg(6, numDoneX);
        rangeSearchKernel.setArg(7, numDoneY);
        rangeSearchKernel.setArg(8, dranges);
        rangeSearchKernel.setArg(9, dir.mat);
        rangeSearchKernel.setArg(10, dir.r);
        rangeSearchKernel.setArg(11, dir.c);
        rangeSearchKernel.setArg(12, dir.pr);
        rangeSearchKernel.setArg(13, dir.pc);
        rangeSearchKernel.setArg(14, dir.ld);

        cl::CommandQueue& queue = OclContextHolder::queue;

        queue.enqueueNDRangeKernel(rangeSearchKernel, cl::NullRange,
                                   global, local);

        //rangeSearchKernel<<<grid,block>>>(dD, numDoneX, numDoneY, dranges, dir);
        numDoneY += todoY;
      }
      numDoneX += todoX;
    }



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

    cl::NDRange local(BLOCK_SIZE, BLOCK_SIZE);

    //dim3 grid;
    unint numDone, todo;

    //grid.x = 1;

    numDone = 0;
    while( numDone < dq.pr ){
        todo = MIN( dq.pr - numDone, MAX_BS*BLOCK_SIZE );
        //grid.y = todo/BLOCK_SIZE;

        //cl::NDRange global(todo, BLOCK_SIZE);
        cl::NDRange global(BLOCK_SIZE, todo);

        cl::CommandQueue& queue = OclContextHolder::queue;
        cl::Kernel& nnKernel = OclContextHolder::nnKernel;

        nnKernel.setArg(0, dq.mat);
        nnKernel.setArg(1, dq.r);
        nnKernel.setArg(2, dq.c);
        nnKernel.setArg(3, dq.pr);
        nnKernel.setArg(4, dq.pc);
        nnKernel.setArg(5, dq.ld);
        nnKernel.setArg(6, numDone);
        nnKernel.setArg(7, dx.mat);
        nnKernel.setArg(8, dx.r);
        nnKernel.setArg(9, dx.c);
        nnKernel.setArg(10, dx.pr);
        nnKernel.setArg(11, dx.pc);
        nnKernel.setArg(12, dx.ld);
        nnKernel.setArg(13, dMins);
        nnKernel.setArg(14, dMinIDs);

        queue.enqueueNDRangeKernel(nnKernel, cl::NullRange,
                                   global, local);

        //nnKernel<<<grid,block>>>(dq,numDone,dx,dMins,dMinIDs);
        numDone += todo;
    }
      //cudaThreadSynchronize();

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
void planKNNWrap(const ocl_matrix& dq, const cl::Buffer& dqMap, const ocl_matrix& dx,
                 const ocl_intMatrix& dxMap, ocl_matrix& dMins,
                 ocl_intMatrix& dMinIDs, const ocl_compPlan& dcP, unint compLength){


    cl::NDRange local(BLOCK_SIZE,BLOCK_SIZE);
    //dim3 grid;
    unint todo;

    //grid.x = 1;
    unint numDone = 0;
    while( numDone<compLength ){
      todo = MIN( (compLength-numDone) , MAX_BS*BLOCK_SIZE );
      //grid.y = todo/BLOCK_SIZE;

      cl::NDRange global(BLOCK_SIZE, todo);

      cl::CommandQueue& queue = OclContextHolder::queue;
      cl::Kernel& planKNNKernel = OclContextHolder::planKNNKernel;

      planKNNKernel.setArg(0, dq.mat);
      planKNNKernel.setArg(1, dq.r);
      planKNNKernel.setArg(2, dq.c);
      planKNNKernel.setArg(3, dq.pr);
      planKNNKernel.setArg(4, dq.pc);
      planKNNKernel.setArg(5, dq.ld);
      planKNNKernel.setArg(6, dqMap);
      planKNNKernel.setArg(7, dx.mat);
      planKNNKernel.setArg(8, dx.r);
      planKNNKernel.setArg(9, dx.c);
      planKNNKernel.setArg(10, dx.pr);
      planKNNKernel.setArg(11, dx.pc);
      planKNNKernel.setArg(12, dx.ld);
      planKNNKernel.setArg(13, dxMap.mat);
      planKNNKernel.setArg(14, dxMap.r);
      planKNNKernel.setArg(15, dxMap.c);
      planKNNKernel.setArg(16, dxMap.pr);
      planKNNKernel.setArg(17, dxMap.pc);
      planKNNKernel.setArg(18, dxMap.ld);
      planKNNKernel.setArg(19, dMins.mat);
      planKNNKernel.setArg(20, dMins.r);
      planKNNKernel.setArg(21, dMins.c);
      planKNNKernel.setArg(22, dMins.pr);
      planKNNKernel.setArg(23, dMins.pc);
      planKNNKernel.setArg(24, dMins.ld);
      planKNNKernel.setArg(25, dMinIDs.mat);
      planKNNKernel.setArg(26, dMinIDs.r);
      planKNNKernel.setArg(27, dMinIDs.c);
      planKNNKernel.setArg(28, dMinIDs.pr);
      planKNNKernel.setArg(29, dMinIDs.pc);
      planKNNKernel.setArg(30, dMinIDs.ld);
      planKNNKernel.setArg(31, dcP.numGroups);
      planKNNKernel.setArg(32, dcP.groupCountX);
      planKNNKernel.setArg(33, dcP.qToQGroup);
      planKNNKernel.setArg(34, dcP.qGroupToXGroup);
      planKNNKernel.setArg(35, dcP.ld);
      planKNNKernel.setArg(36, numDone);

      queue.enqueueNDRangeKernel(planKNNKernel, cl::NullRange,
                                 global, local);

      //planKNNKernel<<<grid,block>>>(dq,dqMap,dx,dxMap,dMins,dMinIDs,dcP,numDone);

      numDone += todo;
    }
    //cudaThreadSynchronize();

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
