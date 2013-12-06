/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef KERNELWRAP_CU
#define KERNELWRAP_CU

//#include<cuda.h>
#include <cstdio>
#include <cassert>

#include "kernels.h"
#include "defs.h"
#include "utilsGPU.h"
#include <iostream>

void dist1Wrap(const ocl_matrix& dq, const ocl_matrix& dx,
               ocl_matrix& dD, unint dq_offset)
{
    cl::Kernel& dist1Kernel = OclContextHolder::dist1Kernel;
    cl::NDRange local(BLOCK_SIZE, BLOCK_SIZE);

    unint todoX, todoY, numDoneX, numDoneY;

    numDoneX = 0;
    while ( numDoneX < dx.pr )
    {
        todoX = MIN( dx.pr - numDoneX, MAX_BS*BLOCK_SIZE );

        numDoneY = 0;

        while(numDoneY < dq.pr)
        {
            todoY = MIN(dq.pr - numDoneY, MAX_BS*BLOCK_SIZE);

            cl::NDRange global(todoX, todoY);

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
            dist1Kernel.setArg(20, dq_offset);

            queue.enqueueNDRangeKernel(dist1Kernel, cl::NullRange,
                                       global, local);
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


void findRangeWrap(const ocl_matrix& dD, cl::Buffer& dranges,
                   unint cntWant, unint offset)
{
    cl::Kernel& findRangeKernel = OclContextHolder::findRangeKernel;

    cl::NDRange local(4*BLOCK_SIZE,BLOCK_SIZE/4);

    unint numDone, todo;

    numDone = 0;

    while(numDone < dD.pr)
    {
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
        findRangeKernel.setArg(9, offset);

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


void rangeSearchWrap(const ocl_matrix& dD, cl::Buffer& dranges,
                     ocl_charMatrix& dir)
{

    cl::Kernel& rangeSearchKernel = OclContextHolder::rangeSearchKernel;

    cl::NDRange local(BLOCK_SIZE,BLOCK_SIZE);

    unint todoX, todoY, numDoneX, numDoneY;

    numDoneX = 0;
    while ( numDoneX < dD.pc )
    {
        todoX = MIN( dD.pc - numDoneX, MAX_BS*BLOCK_SIZE );

        numDoneY = 0;

        while( numDoneY < dD.pr )
        {
            todoY = MIN( dD.pr - numDoneY, MAX_BS*BLOCK_SIZE );

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

    unint numDone = 0;
    unint todo;

    while(numDone < dq.pr)
    {
        todo = MIN(dq.pr - numDone, MAX_BS*BLOCK_SIZE);

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

        cl_int err = queue.enqueueNDRangeKernel(nnKernel, cl::NullRange,
                                                global, local);

        checkErr(err);

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


void knnWrap(const ocl_matrix& dq, const ocl_matrix& dx,
             ocl_matrix& dMins, ocl_intMatrix& dMinIDs){

    cl::NDRange local(BLOCK_SIZE, BLOCK_SIZE);
    //dim3 grid;
    unint numDone, todo;
//    grid.x = 1;

    numDone = 0;
    while( numDone < dq.pr )
    {
        todo = MIN(dq.pr - numDone, MAX_BS*BLOCK_SIZE);

        cl::NDRange global(BLOCK_SIZE, todo);

        cl::CommandQueue& queue = OclContextHolder::queue;
        cl::Kernel& nn32Kernel = OclContextHolder::nn32Kernel;

        nn32Kernel.setArg(0, dq.mat);
        nn32Kernel.setArg(1, dq.r);
        nn32Kernel.setArg(2, dq.c);
        nn32Kernel.setArg(3, dq.pr);
        nn32Kernel.setArg(4, dq.pc);
        nn32Kernel.setArg(5, dq.ld);
        nn32Kernel.setArg(6, numDone);
        nn32Kernel.setArg(7, dx.mat);
        nn32Kernel.setArg(8, dx.r);
        nn32Kernel.setArg(9, dx.c);
        nn32Kernel.setArg(10, dx.pr);
        nn32Kernel.setArg(11, dx.pc);
        nn32Kernel.setArg(12, dx.ld);
        nn32Kernel.setArg(13, dMins.mat);
        nn32Kernel.setArg(14, dMins.r);
        nn32Kernel.setArg(15, dMins.c);
        nn32Kernel.setArg(16, dMins.pr);
        nn32Kernel.setArg(17, dMins.pc);
        nn32Kernel.setArg(18, dMins.ld);
        nn32Kernel.setArg(19, dMinIDs.mat);
        nn32Kernel.setArg(20, dMinIDs.r);
        nn32Kernel.setArg(21, dMinIDs.c);
        nn32Kernel.setArg(22, dMinIDs.pr);
        nn32Kernel.setArg(23, dMinIDs.pc);
        nn32Kernel.setArg(24, dMinIDs.ld);

        queue.enqueueNDRangeKernel(nn32Kernel, cl::NullRange,
                                   global, local);

        numDone += todo;
    }



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


void planKNNWrap(const ocl_matrix& dq, const cl::Buffer& dqMap,
                 const ocl_matrix& dx, const ocl_intMatrix& dxMap,
                 ocl_matrix& dMins, ocl_intMatrix& dMinIDs,
                 const ocl_compPlan& dcP, unint compLength)
{
    cl::NDRange local(BLOCK_SIZE, BLOCK_SIZE);
    unint todo;
    unint numDone = 0;

    while(numDone < compLength)
    {        
        todo = MIN((compLength-numDone), MAX_BS*BLOCK_SIZE);

        //std::cout << "planKNN kernel todo: " << todo << std::endl;

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

        cl_int err = queue.enqueueNDRangeKernel(planKNNKernel, cl::NullRange,
                                                global, local);
        checkErr(err);

        numDone += todo;
    }
}

///** simple version */
//void planKNNWrap(const ocl_matrix& dq,
//                 const ocl_matrix& dx, const ocl_intMatrix& dxMap,
//                 const cl::Buffer& repsIDs,
//                 ocl_matrix& dMins, ocl_intMatrix& dMinIDs)
//{
//    cl::NDRange local(BLOCK_SIZE, BLOCK_SIZE);
//    unint todo;
//    unint numDone = 0;

//    while(numDone < dq.pr)
//    {
//        todo = MIN((dq.pr-numDone), MAX_BS*BLOCK_SIZE);

//        //std::cout << "planKNN kernel todo: " << todo << std::endl;

//        cl::NDRange global(BLOCK_SIZE, todo);

//        cl::CommandQueue& queue = OclContextHolder::queue;
//        cl::Kernel& simplePlanKNNKernel = OclContextHolder::simplePlanKNNKernel;

//        simplePlanKNNKernel.setArg(0, dq.mat);
//        simplePlanKNNKernel.setArg(1, dq.r);
//        simplePlanKNNKernel.setArg(2, dq.c);
//        simplePlanKNNKernel.setArg(3, dq.pr);
//        simplePlanKNNKernel.setArg(4, dq.pc);
//        simplePlanKNNKernel.setArg(5, dq.ld);
//        simplePlanKNNKernel.setArg(6, dx.mat);
//        simplePlanKNNKernel.setArg(7, dx.r);
//        simplePlanKNNKernel.setArg(8, dx.c);
//        simplePlanKNNKernel.setArg(9, dx.pr);
//        simplePlanKNNKernel.setArg(10, dx.pc);
//        simplePlanKNNKernel.setArg(11, dx.ld);
//        simplePlanKNNKernel.setArg(12, dxMap.mat);
//        simplePlanKNNKernel.setArg(13, dxMap.r);
//        simplePlanKNNKernel.setArg(14, dxMap.c);
//        simplePlanKNNKernel.setArg(15, dxMap.pr);
//        simplePlanKNNKernel.setArg(16, dxMap.pc);
//        simplePlanKNNKernel.setArg(17, dxMap.ld);
//        simplePlanKNNKernel.setArg(18, dMins.mat);
//        simplePlanKNNKernel.setArg(19, dMins.r);
//        simplePlanKNNKernel.setArg(20, dMins.c);
//        simplePlanKNNKernel.setArg(21, dMins.pr);
//        simplePlanKNNKernel.setArg(22, dMins.pc);
//        simplePlanKNNKernel.setArg(23, dMins.ld);
//        simplePlanKNNKernel.setArg(24, dMinIDs.mat);
//        simplePlanKNNKernel.setArg(25, dMinIDs.r);
//        simplePlanKNNKernel.setArg(26, dMinIDs.c);
//        simplePlanKNNKernel.setArg(27, dMinIDs.pr);
//        simplePlanKNNKernel.setArg(28, dMinIDs.pc);
//        simplePlanKNNKernel.setArg(29, dMinIDs.ld);
//        simplePlanKNNKernel.setArg(30, repsIDs);
//        simplePlanKNNKernel.setArg(31, numDone);

//        cl_int err = queue.enqueueNDRangeKernel(simplePlanKNNKernel, cl::NullRange,
//                                                global, local);
//        checkErr(err);

//        numDone += todo;
//    }
//}



void meanshiftPlanKNNWrap(const ocl_matrix& dq, const cl::Buffer& dqMap,
                          const ocl_matrix& dx, const ocl_intMatrix& dxMap,
                          const ocl_compPlan& dcP,
                          const cl::Buffer& windows,
                          const cl::Buffer& weights,
                          cl::Buffer& selectedPoints,
                          cl::Buffer& selectedDistances,
                          cl::Buffer& selectedPointsNum,
                          unint maxPointsNum,// unint compLength,
                          unint startPos, unint length)
{
    cl::NDRange local(BLOCK_SIZE, BLOCK_SIZE);
   // unint todo;
    //unint numDone = 0;

    assert((length % BLOCK_SIZE) == 0);

   // while(numDone < compLength)
   // {
   //     todo = MIN((compLength - numDone), MAX_BS*BLOCK_SIZE);
        //std::cout << "planKNN kernel todo: " << todo << std::endl;
        //cl::NDRange global(BLOCK_SIZE, todo);

        cl::NDRange global(BLOCK_SIZE, length);

        cl::CommandQueue& queue = OclContextHolder::queue;
        cl::Kernel& meanshiftPlanKNNKernel
                = OclContextHolder::meanshiftPlanKNNKernel;

        meanshiftPlanKNNKernel.setArg(0, dq.mat);
        meanshiftPlanKNNKernel.setArg(1, dq.r);
        meanshiftPlanKNNKernel.setArg(2, dq.c);
        meanshiftPlanKNNKernel.setArg(3, dq.pr);
        meanshiftPlanKNNKernel.setArg(4, dq.pc);
        meanshiftPlanKNNKernel.setArg(5, dq.ld);
        meanshiftPlanKNNKernel.setArg(6, dqMap);
        meanshiftPlanKNNKernel.setArg(7, dx.mat);
        meanshiftPlanKNNKernel.setArg(8, dx.r);
        meanshiftPlanKNNKernel.setArg(9, dx.c);
        meanshiftPlanKNNKernel.setArg(10, dx.pr);
        meanshiftPlanKNNKernel.setArg(11, dx.pc);
        meanshiftPlanKNNKernel.setArg(12, dx.ld);
        meanshiftPlanKNNKernel.setArg(13, dxMap.mat);
        meanshiftPlanKNNKernel.setArg(14, dxMap.r);
        meanshiftPlanKNNKernel.setArg(15, dxMap.c);
        meanshiftPlanKNNKernel.setArg(16, dxMap.pr);
        meanshiftPlanKNNKernel.setArg(17, dxMap.pc);
        meanshiftPlanKNNKernel.setArg(18, dxMap.ld);
        meanshiftPlanKNNKernel.setArg(19, dcP.numGroups);
        meanshiftPlanKNNKernel.setArg(20, dcP.groupCountX);
        meanshiftPlanKNNKernel.setArg(21, dcP.qToQGroup);
        meanshiftPlanKNNKernel.setArg(22, dcP.qGroupToXGroup);
        meanshiftPlanKNNKernel.setArg(23, dcP.ld);
        //meanshiftPlanKNNKernel.setArg(24, numDone);
        meanshiftPlanKNNKernel.setArg(24, startPos);
        meanshiftPlanKNNKernel.setArg(25, windows);
        meanshiftPlanKNNKernel.setArg(26, weights);
        meanshiftPlanKNNKernel.setArg(27, selectedPoints);
        meanshiftPlanKNNKernel.setArg(28, selectedDistances);
        meanshiftPlanKNNKernel.setArg(29, selectedPointsNum);
        meanshiftPlanKNNKernel.setArg(30, maxPointsNum);

        cl_int err = queue.enqueueNDRangeKernel(meanshiftPlanKNNKernel, cl::NullRange,
                                                global, local);
        checkErr(err);

   //     numDone += todo;
  //  }
}

void meanshiftMeanWrap(const ocl_matrix& input,
                       const cl::Buffer& selectedPoints,
                       const cl::Buffer& selectedDistances,
                       const cl::Buffer& selectedPointsNum,
                       const cl::Buffer& windows,
                       const cl::Buffer& weights,
                       unint maxPointsNum,
                       ocl_matrix& output)
{

    cl::NDRange local(BLOCK_SIZE, BLOCK_SIZE);
    cl::NDRange global(BLOCK_SIZE, output.pr);

    cl::CommandQueue& queue = OclContextHolder::queue;
    cl::Kernel& meanshiftMeanKernel = OclContextHolder::meanshiftMeanKernel;

    meanshiftMeanKernel.setArg(0, input.mat);
    meanshiftMeanKernel.setArg(1, input.r);
    meanshiftMeanKernel.setArg(2, input.c);
    meanshiftMeanKernel.setArg(3, input.pr);
    meanshiftMeanKernel.setArg(4, input.pc);
    meanshiftMeanKernel.setArg(5, input.ld);
    meanshiftMeanKernel.setArg(6, output.mat);
    meanshiftMeanKernel.setArg(7, output.r);
    meanshiftMeanKernel.setArg(8, output.c);
    meanshiftMeanKernel.setArg(9, output.pr);
    meanshiftMeanKernel.setArg(10, output.pc);
    meanshiftMeanKernel.setArg(11, output.ld);
    meanshiftMeanKernel.setArg(12, selectedPoints);
    meanshiftMeanKernel.setArg(13, selectedDistances);
    meanshiftMeanKernel.setArg(14, selectedPointsNum);
    meanshiftMeanKernel.setArg(15, windows);
    meanshiftMeanKernel.setArg(16, weights);
    meanshiftMeanKernel.setArg(17, maxPointsNum);

    cl_int err = queue.enqueueNDRangeKernel(meanshiftMeanKernel, cl::NullRange,
                                            global, local);
    checkErr(err);
}

void meanshiftWeightsWrap(const cl::Buffer& pilots, cl::Buffer& weights,
                          unint size, unint dimensionality)
{
    int local_size = BLOCK_SIZE * 8;

    cl::NDRange local(local_size);
    cl::NDRange global(((size + local_size - 1) / local_size) * local_size);

    cl::CommandQueue& queue = OclContextHolder::queue;
    cl::Kernel& meanshifWeightsKernel = OclContextHolder::meanshiftWeightsKernel;

    meanshifWeightsKernel.setArg(0, pilots);
    meanshifWeightsKernel.setArg(1, weights);
    meanshifWeightsKernel.setArg(2, size);
    meanshifWeightsKernel.setArg(3, dimensionality);

    cl_int err = queue.enqueueNDRangeKernel(meanshifWeightsKernel,
                                            cl::NullRange, global, local);
    checkErr(err);
}


void simpleDistanceKernelWrap(const ocl_matrix& in_1, const ocl_matrix& in_2,
                              cl::Buffer& out)
{
    cl::NDRange local(BLOCK_SIZE, BLOCK_SIZE);
    cl::NDRange global(BLOCK_SIZE, in_1.pr);

    cl::CommandQueue& queue = OclContextHolder::queue;
    cl::Kernel& simpleDistancesKernel = OclContextHolder::simpleDistancesKernel;

    simpleDistancesKernel.setArg(0, in_1.mat);
    simpleDistancesKernel.setArg(1, in_2.mat);
    simpleDistancesKernel.setArg(2, out);
    simpleDistancesKernel.setArg(3, in_1.pc);

    cl_int err = queue.enqueueNDRangeKernel(simpleDistancesKernel,
                                            cl::NullRange, global, local);
    checkErr(err);
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
