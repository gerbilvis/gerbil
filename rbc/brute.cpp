/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef BRUTE_CU
#define BRUTE_CU

#include "utilsGPU.h"
#include "utils.h"
#include "rbc.h"
#include "defs.h"
#include "kernels.h"
#include "kernelWrap.h"
#include "brute.h"
#include<stdio.h>
//#include<cuda.h>
//#include<gsl/gsl_sort.h>

void bruteRangeCount(matrix x, matrix q, real *ranges, unint *cnts){
  ocl_matrix dx, dq;
//  real *dranges;
//  unint *dcnts;

  cl::Buffer dranges;
  cl::Buffer dcnts;
  
  cl::Context& context = OclContextHolder::context;
  cl::CommandQueue& queue = OclContextHolder::queue;

  copyAndMove(&dx, &x);
  copyAndMove(&dq, &q);

  int byte_size = q.pr*sizeof(real);
  cl_int err;

  dranges = cl::Buffer(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
  checkErr(err);

  byte_size = q.r*sizeof(real);

  err = queue.enqueueWriteBuffer(dranges, CL_TRUE, 0, byte_size, ranges);
  checkErr(err);

  //checkErr( cudaMalloc( (void**)&dranges, q.pr*sizeof(*dranges) ) );
  //cudaMemcpy( dranges, ranges, q.r*sizeof(*dranges), cudaMemcpyHostToDevice );

  //checkErr( cudaMalloc( (void**)&dcnts, q.pr*sizeof(*dcnts) ) );
  
  rangeCountWrap(dq, dx, dranges, dcnts);
  
  byte_size = q.r*sizeof(*cnts);

  err = queue.enqueueReadBuffer(dranges, CL_TRUE, 0, byte_size, cnts);
  checkErr(err);

  //cudaMemcpy(cnts, dcnts, q.r*sizeof(*cnts), cudaMemcpyDeviceToHost );
//  cudaFree(dcnts);
//  cudaFree(dranges);
//  cudaFree(dx.mat);
//  cudaFree(dq.mat);
}


/* NEVER USED
void bruteSearch(matrix x, matrix q, unint *NNs){
  real *dMins;
  unint *dMinIDs;
  matrix dx, dq;

  copyAndMove( &dx, &x );
  copyAndMove( &dq, &q );

  checkErr( cudaMalloc((void**)&dMins, q.pr*sizeof(*dMins)) );
  checkErr( cudaMalloc((void**)&dMinIDs, q.pr*sizeof(*dMinIDs)) );
  
  nnWrap( dq, dx, dMins, dMinIDs );

  cudaMemcpy( NNs, dMinIDs, dq.r*sizeof(*NNs), cudaMemcpyDeviceToHost );
  
  cudaFree( dMins );
  cudaFree( dMinIDs );
  cudaFree( dx.mat );
  cudaFree( dq.mat );
}*/


void bruteK(matrix x, matrix q, intMatrix NNs, matrix NNdists){
  ocl_matrix dNNdists;
  ocl_intMatrix dMinIDs;
  ocl_matrix dx, dq;

  copyAndMove( &dx, &x );
  copyAndMove( &dq, &q );
  
  cl::Context& context = OclContextHolder::context;
  cl::CommandQueue& queue = OclContextHolder::queue;

  initMat( &dNNdists, q.r, KMAX );

  int byte_size = sizeOfMatB(dNNdists);
  cl_int err;

  dNNdists.mat = cl::Buffer(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
  checkErr(err);

  //checkErr( cudaMalloc((void**)&dNNdists.mat, sizeOfMatB(dNNdists) ) );

  initIntMat( &dMinIDs, q.r, KMAX );

  byte_size = sizeOfIntMatB(dMinIDs);

  dMinIDs.mat = cl::Buffer(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
  checkErr(err);

 // checkErr( cudaMalloc((void**)&dMinIDs.mat, sizeOfIntMatB(dMinIDs) ) );

  knnWrap( dq, dx, dNNdists, dMinIDs );

  err = queue.enqueueReadBuffer(dMinIDs.mat, CL_TRUE, 0, sizeOfIntMatB(NNs), NNs.mat);
  checkErr(err);

  err = queue.enqueueReadBuffer(dNNdists.mat, CL_TRUE, 0, sizeOfMatB(NNdists), NNdists.mat);
  checkErr(err);

//  cudaMemcpy( NNs.mat, dMinIDs.mat, sizeOfIntMatB(NNs), cudaMemcpyDeviceToHost );
//  cudaMemcpy( NNdists.mat, dNNdists.mat, sizeOfMatB(NNdists), cudaMemcpyDeviceToHost );
  
//  cudaFree( dNNdists.mat );
//  cudaFree( dMinIDs.mat );
//  cudaFree( dx.mat );
//  cudaFree( dq.mat );
}


void bruteCPU(matrix X, matrix Q, unint *NNs){
  real *dtoNNs; 
  real temp;

  unint i, j;

  dtoNNs = (real*)calloc(Q.r,sizeof(*dtoNNs));
  
  for( i=0; i<Q.r; i++ ){
    dtoNNs[i] = MAX_REAL;
    NNs[i] = 0;
    for(j=0; j<X.r; j++ ){
      temp = distVec( Q, X, i, j );
      if( temp < dtoNNs[i]){
	NNs[i] = j;
	dtoNNs[i] = temp;
      }
    }
  }
  
  free(dtoNNs);  
}


//The following method works properly, but requires the GNU scientific
//library.  If you want to use it, uncomment the code, uncomment the include
//above, and adjust the makefile.  It was used for debugging purposes, but
//is not required by anything else.

/* void bruteKCPU(matrix x, matrix q, intMatrix NNs){ */
/*   int i, j; */

/*   float **d; */
/*   d = (float**)calloc(q.pr, sizeof(*d)); */
/*   size_t **t; */
/*   t = (size_t**)calloc(q.pr, sizeof(*t)); */
/*   for( i=0; i<q.pr; i++){ */
/*     d[i] = (float*)calloc(x.pr, sizeof(**d)); */
/*     t[i] = (size_t*)calloc(x.pr, sizeof(**t)); */
/*   } */

/*   for( i=0; i<q.r; i++){ */
/*     for( j=0; j<x.r; j++) */
/*       d[i][j] = distVec( q, x, i, j ); */
/*     gsl_sort_float_index(t[i], d[i], 1, x.r); */
/*     for ( j=0; j<KMAX; j++) */
/*       NNs.mat[IDX( i, j, NNs.ld )] = t[i][j]; */
/*   } */

/*   for( i=0; i<q.pr; i++){ */
/*     free(t[i]); */
/*     free(d[i]); */
/*   } */
/*   free(t); */
/*   free(d); */
/* } */
#endif
