/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */
#ifndef SKERNELWRAP_CU
#define SKERNELWRAP_CU

#include "sKernel.h"
#include<cuda.h>
#include "defs.h"
#include "utilsGPU.h"
#include<stdio.h>

#include <cmath>

//void getCountsWrap(unint *counts, charMatrix ir, intMatrix sums){
void getCountsWrap(cl::Buffer& counts, ocl_charMatrix& ir, ocl_intMatrix& sums){

    cl::NDRange local(BLOCK_SIZE,1);
    //dim3 grid;
    //grid.y=1;
    unint todo, numDone;
    numDone = 0;

    while(numDone < ir.pr){
      todo = MIN( ir.pr - numDone, MAX_BS*BLOCK_SIZE );
      //grid.x = todo/BLOCK_SIZE;
      cl::NDRange global(todo, 1);

      cl::Kernel& getCountsKernel = OclContextHolder::getCountsKernel;
      cl::CommandQueue& queue = OclContextHolder::queue;

      getCountsKernel.setArg(0, counts);
      getCountsKernel.setArg(1, numDone);
      getCountsKernel.setArg(2, ir.mat);
      getCountsKernel.setArg(3, ir.r);
      getCountsKernel.setArg(4, ir.c);
      getCountsKernel.setArg(5, ir.pr);
      getCountsKernel.setArg(6, ir.pc);
      getCountsKernel.setArg(7, ir.ld);
      getCountsKernel.setArg(8, sums.mat);
      getCountsKernel.setArg(9, sums.r);
      getCountsKernel.setArg(10, sums.c);
      getCountsKernel.setArg(11, sums.pr);
      getCountsKernel.setArg(12, sums.pc);
      getCountsKernel.setArg(13, sums.ld);

      //getCountsKernel<<<grid,block>>>(counts, numDone, ir, sums);
      queue.enqueueNDRangeKernel(getCountsKernel, cl::NullRange, global, local);

      numDone += todo;
    }


/*  dim3 block(BLOCK_SIZE,1);
  dim3 grid;
  grid.y=1;
  unint todo, numDone;
  
  numDone = 0;
  while(numDone < ir.pr){
    todo = MIN( ir.pr - numDone, MAX_BS*BLOCK_SIZE );
    grid.x = todo/BLOCK_SIZE;
    getCountsKernel<<<grid,block>>>(counts, numDone, ir, sums);
    numDone += todo;
  }*/
}


void buildMapWrap(ocl_intMatrix& map, ocl_charMatrix& ir, ocl_intMatrix& sums, unint offSet){

    unint numScans = (ir.c+SCAN_WIDTH-1)/SCAN_WIDTH;

    cl::NDRange local(SCAN_WIDTH / 2, 1);

    unint todo, numDone;

    //grid.x = numScans;
    numDone = 0;
    while( numDone < ir.r ){
      todo = MIN( ir.r-numDone, MAX_BS );
      //grid.y = todo;

      cl::NDRange global(numScans * SCAN_WIDTH / 2, todo);

      cl::Kernel& buildMapKernel = OclContextHolder::buildMapKernel;
      cl::CommandQueue& queue = OclContextHolder::queue;

      buildMapKernel.setArg(0, map.mat);
      buildMapKernel.setArg(1, map.r);
      buildMapKernel.setArg(2, map.c);
      buildMapKernel.setArg(3, map.pr);
      buildMapKernel.setArg(4, map.pc);
      buildMapKernel.setArg(5, map.ld);
      buildMapKernel.setArg(6, ir.mat);
      buildMapKernel.setArg(7, ir.r);
      buildMapKernel.setArg(8, ir.c);
      buildMapKernel.setArg(9, ir.pr);
      buildMapKernel.setArg(10, ir.pc);
      buildMapKernel.setArg(11, ir.ld);
      buildMapKernel.setArg(12, sums.mat);
      buildMapKernel.setArg(13, sums.r);
      buildMapKernel.setArg(14, sums.c);
      buildMapKernel.setArg(15, sums.pr);
      buildMapKernel.setArg(16, sums.pc);
      buildMapKernel.setArg(17, sums.ld);
      buildMapKernel.setArg(18, offSet + numDone);

      queue.enqueueNDRangeKernel(buildMapKernel, cl::NullRange, global, local);

      //buildMapKernel<<<grid,block>>>(map, ir, sums, offSet+numDone);

      numDone += todo;
    }

/*  unint numScans = (ir.c+SCAN_WIDTH-1)/SCAN_WIDTH;
  dim3 block( SCAN_WIDTH/2, 1 );
  dim3 grid;
  unint todo, numDone;

  grid.x = numScans;
  numDone = 0;
  while( numDone < ir.r ){
    todo = MIN( ir.r-numDone, MAX_BS );
    grid.y = todo;
    buildMapKernel<<<grid,block>>>(map, ir, sums, offSet+numDone);
    numDone += todo;
  }*/
}



void sumWrap(ocl_charMatrix& in, ocl_intMatrix& sum){

    int i;
      unint todo, numDone, temp;
      unint n = in.c;
      unint numScans = (n+SCAN_WIDTH-1)/SCAN_WIDTH;
      unint depth = ceil( log(n) / log(SCAN_WIDTH) ) -1 ;
      unint *width = (unint*)calloc( depth+1, sizeof(*width) );

//      intMatrix *dAux;
//      dAux = (intMatrix*)calloc( depth+1, sizeof(*dAux) );

      cl::Context& context = OclContextHolder::context;
      cl::CommandQueue& queue = OclContextHolder::queue;


      ocl_intMatrix* dAux = new ocl_intMatrix[depth + 1];

      for( i=0, temp=n; i<=depth; i++){
        temp = (temp+SCAN_WIDTH-1)/SCAN_WIDTH;
        dAux[i].r=dAux[i].pr=in.r; dAux[i].c=dAux[i].pc=dAux[i].ld=temp;

        int byte_size = dAux[i].pr*dAux[i].pc*sizeof(unint);
        cl_int err;

        dAux[i].mat = cl::Buffer(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
        checkErr(err);

        //checkErr( cudaMalloc( (void**)&dAux[i].mat, dAux[i].pr*dAux[i].pc*sizeof(*dAux[i].mat)));
      }

      //dim3 block( SCAN_WIDTH/2, 1 );
      cl::NDRange local(SCAN_WIDTH/2, 1);
      //dim3 grid;

      numDone=0;
      while( numDone < in.r ){
        todo = MIN( in.r - numDone, MAX_BS );
        numScans = (n+SCAN_WIDTH-1)/SCAN_WIDTH;
        dAux[0].r=dAux[0].pr=todo;
//        grid.x = numScans;
//        grid.y = todo;

        cl::NDRange global(numScans * SCAN_WIDTH / 2, todo);

        cl::Kernel& sumKernel = OclContextHolder::sumKernel;

        sumKernel.setArg(0, in.mat);
        sumKernel.setArg(1, in.r);
        sumKernel.setArg(2, in.c);
        sumKernel.setArg(3, in.pr);
        sumKernel.setArg(4, in.pc);
        sumKernel.setArg(5, in.ld);
        sumKernel.setArg(6, sum.mat);
        sumKernel.setArg(7, sum.r);
        sumKernel.setArg(8, sum.c);
        sumKernel.setArg(9, sum.pr);
        sumKernel.setArg(10, sum.pc);
        sumKernel.setArg(11, sum.ld);
        sumKernel.setArg(12, dAux[0].mat);
        sumKernel.setArg(13, dAux[0].r);
        sumKernel.setArg(14, dAux[0].c);
        sumKernel.setArg(15, dAux[0].pr);
        sumKernel.setArg(16, dAux[0].pc);
        sumKernel.setArg(17, dAux[0].ld);
        sumKernel.setArg(18, n);

        queue.enqueueNDRangeKernel(sumKernel, cl::NullRange, global, local);
        //sumKernel<<<grid,block>>>(in, sum, dAux[0], n);

        //cudaThreadSynchronize();

        width[0] = numScans; // Necessary because following loop might not be entered
        for( i=0; i<depth; i++ ){
          width[i] = numScans;
          numScans = (numScans+SCAN_WIDTH-1)/SCAN_WIDTH;
          dAux[i+1].r=dAux[i+1].pr=todo;

          //grid.x = numScans;
          cl::NDRange global(numScans * SCAN_WIDTH / 2, todo);

          cl::Kernel& sumKernelI = OclContextHolder::sumKernelI;

          sumKernelI.setArg(0, dAux[i].mat);
          sumKernelI.setArg(1, dAux[i].r);
          sumKernelI.setArg(2, dAux[i].c);
          sumKernelI.setArg(3, dAux[i].pr);
          sumKernelI.setArg(4, dAux[i].pc);
          sumKernelI.setArg(5, dAux[i].ld);
          sumKernelI.setArg(6, dAux[i].mat);
          sumKernelI.setArg(7, dAux[i].r);
          sumKernelI.setArg(8, dAux[i].c);
          sumKernelI.setArg(9, dAux[i].pr);
          sumKernelI.setArg(10, dAux[i].pc);
          sumKernelI.setArg(11, dAux[i].ld);
          sumKernelI.setArg(12, dAux[i+1].mat);
          sumKernelI.setArg(13, dAux[i+1].r);
          sumKernelI.setArg(14, dAux[i+1].c);
          sumKernelI.setArg(15, dAux[i+1].pr);
          sumKernelI.setArg(16, dAux[i+1].pc);
          sumKernelI.setArg(17, dAux[i+1].ld);
          sumKernelI.setArg(18, width[i]);

          queue.enqueueNDRangeKernel(sumKernelI, cl::NullRange, global, local);

//          sumKernelI<<<grid,block>>>(dAux[i], dAux[i], dAux[i+1], width[i]);
//          cudaThreadSynchronize();
        }

        for( i=depth-1; i>0; i-- ){
        //  grid.x = width[i];

            cl::NDRange global(width[i] * SCAN_WIDTH / 2, todo);

            cl::Kernel& combineSumKernel = OclContextHolder::combineSumKernel;

            combineSumKernel.setArg(0, dAux[i-1].mat);
            combineSumKernel.setArg(1, dAux[i-1].r);
            combineSumKernel.setArg(2, dAux[i-1].c);
            combineSumKernel.setArg(3, dAux[i-1].pr);
            combineSumKernel.setArg(4, dAux[i-1].pc);
            combineSumKernel.setArg(5, dAux[i-1].ld);
            combineSumKernel.setArg(6, numDone);
            combineSumKernel.setArg(7, dAux[i].mat);
            combineSumKernel.setArg(8, dAux[i].r);
            combineSumKernel.setArg(9, dAux[i].c);
            combineSumKernel.setArg(10, dAux[i].pr);
            combineSumKernel.setArg(11, dAux[i].pc);
            combineSumKernel.setArg(12, dAux[i].ld);
            combineSumKernel.setArg(13, width[i-1]);

            queue.enqueueNDRangeKernel(combineSumKernel, cl::NullRange, global, local);

         // combineSumKernel<<<grid,block>>>(dAux[i-1], numDone, dAux[i], width[i-1]);
         // cudaThreadSynchronize();
        }

       // grid.x = width[0];
       // combineSumKernel<<<grid,block>>>(sum, numDone, dAux[0], n);
       // cudaThreadSynchronize();

        global = cl::NDRange(width[0] * SCAN_WIDTH / 2, todo);

        cl::Kernel& combineSumKernel = OclContextHolder::combineSumKernel;

        combineSumKernel.setArg(0, sum.mat);
        combineSumKernel.setArg(1, sum.r);
        combineSumKernel.setArg(2, sum.c);
        combineSumKernel.setArg(3, sum.pr);
        combineSumKernel.setArg(4, sum.pc);
        combineSumKernel.setArg(5, sum.ld);
        combineSumKernel.setArg(6, numDone);
        combineSumKernel.setArg(7, dAux[0].mat);
        combineSumKernel.setArg(8, dAux[0].r);
        combineSumKernel.setArg(9, dAux[0].c);
        combineSumKernel.setArg(10, dAux[0].pr);
        combineSumKernel.setArg(11, dAux[0].pc);
        combineSumKernel.setArg(12, dAux[0].ld);
        combineSumKernel.setArg(13, n);

        queue.enqueueNDRangeKernel(combineSumKernel, cl::NullRange, global, local);


        numDone += todo;
      }

//      for( i=0; i<=depth; i++)
//       cudaFree(dAux[i].mat);
      //free(dAux);

      delete[] dAux;

      free(width);





/*  int i;
  unint todo, numDone, temp;
  unint n = in.c;
  unint numScans = (n+SCAN_WIDTH-1)/SCAN_WIDTH;
  unint depth = ceil( log(n) / log(SCAN_WIDTH) ) -1 ;
  unint *width = (unint*)calloc( depth+1, sizeof(*width) );
    
  intMatrix *dAux;
  dAux = (intMatrix*)calloc( depth+1, sizeof(*dAux) );
  
  for( i=0, temp=n; i<=depth; i++){
    temp = (temp+SCAN_WIDTH-1)/SCAN_WIDTH;
    dAux[i].r=dAux[i].pr=in.r; dAux[i].c=dAux[i].pc=dAux[i].ld=temp;
    checkErr( cudaMalloc( (void**)&dAux[i].mat, dAux[i].pr*dAux[i].pc*sizeof(*dAux[i].mat) ) );
  }

  dim3 block( SCAN_WIDTH/2, 1 );
  dim3 grid;
  
  numDone=0;
  while( numDone < in.r ){
    todo = MIN( in.r - numDone, MAX_BS );
    numScans = (n+SCAN_WIDTH-1)/SCAN_WIDTH;
    dAux[0].r=dAux[0].pr=todo;
    grid.x = numScans;
    grid.y = todo;
    sumKernel<<<grid,block>>>(in, sum, dAux[0], n);
    cudaThreadSynchronize();
    
    width[0] = numScans; // Necessary because following loop might not be entered
    for( i=0; i<depth; i++ ){
      width[i] = numScans;
      numScans = (numScans+SCAN_WIDTH-1)/SCAN_WIDTH;
      dAux[i+1].r=dAux[i+1].pr=todo;
      
      grid.x = numScans;
      sumKernelI<<<grid,block>>>(dAux[i], dAux[i], dAux[i+1], width[i]);
      cudaThreadSynchronize();
    }
  
    for( i=depth-1; i>0; i-- ){
      grid.x = width[i];
      combineSumKernel<<<grid,block>>>(dAux[i-1], numDone, dAux[i], width[i-1]);
      cudaThreadSynchronize();
    }
    
    grid.x = width[0];
    combineSumKernel<<<grid,block>>>(sum, numDone, dAux[0], n);
    cudaThreadSynchronize();
    
    numDone += todo;
  }

  for( i=0; i<=depth; i++)
   cudaFree(dAux[i].mat);
  free(dAux);
  free(width);
  */
}


#endif
