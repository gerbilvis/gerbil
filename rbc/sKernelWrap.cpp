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

//void getCountsWrap(unint *counts, charMatrix ir, intMatrix sums){
void getCountsWrap(cl::Buffer& counts, ocl_charMatrix& ir, ocl_intMatrix& sums){
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
