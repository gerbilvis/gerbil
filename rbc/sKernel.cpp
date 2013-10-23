/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef SKERNEL_CU
#define SKERNEL_CU

#include<stdio.h>
#include<math.h>
#include "sKernel.h"
#include "defs.h"
#include "utils.h"

__global__ void sumKernel(charMatrix in, intMatrix sum, intMatrix sumaux, unint n){
  unint id = threadIdx.x;
  unint bo = blockIdx.x*SCAN_WIDTH; //block offset
  unint r = blockIdx.y;
  unint d, t;
  
  const unint l=SCAN_WIDTH; //length

  unint off=1;

  __shared__ unint ssum[l];

  ssum[2*id] = (bo+2*id < n) ? in.mat[IDX( r, bo+2*id, in.ld )] : 0;
  ssum[2*id+1] = (bo+2*id+1 < n) ? in.mat[IDX( r, bo+2*id+1, in.ld)] : 0;
    
  //up-sweep
  for( d=l>>1; d > 0; d>>=1 ){
    __syncthreads();
    
    if( id < d ){
      ssum[ off*(2*id+2)-1 ] += ssum[ off*(2*id+1)-1 ];
    }
    off *= 2;
  }
  
  __syncthreads();
  
  if ( id == 0 ){
    sumaux.mat[IDX( r, blockIdx.x, sumaux.ld )] = ssum[ l-1 ];
    ssum[ l-1 ] = 0;
  }
  
  //down-sweep
  for ( d=1; d<l; d*=2 ){
    off >>= 1;
    __syncthreads();
    
    if( id < d ){
      t = ssum[ off*(2*id+1)-1 ];
      ssum[ off*(2*id+1)-1 ] = ssum[ off*(2*id+2)-1 ];
      ssum[ off*(2*id+2)-1 ] += t;
    }   
  }

  __syncthreads();
 
  if( bo+2*id < n ) 
    sum.mat[IDX( r, bo+2*id, sum.ld )] = ssum[2*id];
  if( bo+2*id+1 < n )
    sum.mat[IDX( r, bo+2*id+1, sum.ld )] = ssum[2*id+1];
}


//This is the same as sumKernel, but takes an int matrix as input.
__global__ void sumKernelI(intMatrix in, intMatrix sum, intMatrix sumaux, unint n){
  unint id = threadIdx.x;
  unint bo = blockIdx.x*SCAN_WIDTH; //block offset
  unint r = blockIdx.y;
  unint d, t;
  
  const unint l=SCAN_WIDTH; //length

  unint off=1;

  __shared__ unint ssum[l];

  ssum[2*id] = (bo+2*id < n) ? in.mat[IDX( r, bo+2*id, in.ld )] : 0;
  ssum[2*id+1] = (bo+2*id+1 < n) ? in.mat[IDX( r, bo+2*id+1, in.ld)] : 0;

  //up-sweep
  for( d=l>>1; d > 0; d>>=1 ){
    __syncthreads();
    
    if( id < d ){
      ssum[ off*(2*id+2)-1 ] += ssum[ off*(2*id+1)-1 ];
    }
    off *= 2;
  }
  
  __syncthreads();
  
  if ( id == 0 ){
    sumaux.mat[IDX( r, blockIdx.x, sumaux.ld )] = ssum[ l-1 ];
    ssum[ l-1 ] = 0;
  }
  
  //down-sweep
  for ( d=1; d<l; d*=2 ){
    off >>= 1;
    __syncthreads();
    
    if( id < d ){
      t = ssum[ off*(2*id+1)-1 ];
      ssum[ off*(2*id+1)-1 ] = ssum[ off*(2*id+2)-1 ];
      ssum[ off*(2*id+2)-1 ] += t;
    }
  }
  
  __syncthreads();
  
  if( bo+2*id < n )
    sum.mat[IDX( r, bo+2*id, sum.ld )] = ssum[2*id];
  
  if( bo+2*id+1 < n )
    sum.mat[IDX( r, bo+2*id+1, sum.ld )] = ssum[2*id+1];
}



__global__ void combineSumKernel(intMatrix sum, unint numDone, intMatrix daux, unint n){
  unint id = threadIdx.x;
  unint bo = blockIdx.x * SCAN_WIDTH;
  unint r = blockIdx.y+numDone;
  
  if(bo+2*id < n)
    sum.mat[IDX( r, bo+2*id, sum.ld )] += daux.mat[IDX( r, blockIdx.x, daux.ld )];
  if(bo+2*id+1 < n)
    sum.mat[IDX( r, bo+2*id+1, sum.ld )] += daux.mat[IDX( r, blockIdx.x, daux.ld )];
  
}


__global__ void getCountsKernel(unint *counts, unint numDone, charMatrix ir, intMatrix sums){
  unint r = blockIdx.x*BLOCK_SIZE + threadIdx.x + numDone;
  if ( r < ir.r ){
    counts[r] = ir.mat[IDX( r, ir.c-1, ir.ld )] ? sums.mat[IDX( r, sums.c-1, sums.ld )]+1 : sums.mat[IDX( r, sums.c-1, sums.ld )];
  }
}


__global__ void buildMapKernel(intMatrix map, charMatrix ir, intMatrix sums, unint offSet){
  unint id = threadIdx.x;
  unint bo = blockIdx.x * SCAN_WIDTH;
  unint r = blockIdx.y;

  if(bo+2*id < ir.c && ir.mat[IDX( r, bo+2*id, ir.ld )])
    map.mat[IDX( r+offSet, sums.mat[IDX( r, bo+2*id, sums.ld )], map.ld)] = bo+2*id;
  if(bo+2*id+1 < ir.c && ir.mat[IDX( r, bo+2*id+1, ir.ld )])
    map.mat[IDX( r+offSet, sums.mat[IDX( r, bo+2*id+1, sums.ld )], map.ld)] = bo+2*id+1;
}


#endif
