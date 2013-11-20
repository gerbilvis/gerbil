/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef RBC_OPS_CU
#define RBC_OPS_CU

#include<sys/time.h>
#include<stdio.h>
#include <iostream>
#include <cassert>
//#include<cuda.h>
#include "utils.h"
#include "defs.h"
#include "utilsGPU.h"
#include "rbc_ops.h"
#include "kernels.h"
#include "kernelWrap.h"
#include "sKernelWrap.h"

/* NEVER USED
void queryRBC(const matrix q, const rbcStruct rbcS, unint *NNs, real* NNdists){
  unint m = q.r;
  unint numReps = rbcS.dr.r;
  unint compLength;
  compPlan dcP;
  unint *qMap, *dqMap;
  qMap = (unint*)calloc(PAD(m+(BLOCK_SIZE-1)*PAD(numReps)),sizeof(*qMap));
  matrix dq;
  copyAndMove(&dq, &q);
  
  charMatrix cM;
  cM.r=cM.c=numReps; cM.pr=cM.pc=cM.ld=PAD(numReps);
  cM.mat = (char*)calloc( cM.pr*cM.pc, sizeof(*cM.mat) );
  
  unint *repIDsQ;
  repIDsQ = (unint*)calloc( m, sizeof(*repIDsQ) );
  real *distToRepsQ;
  distToRepsQ = (real*)calloc( m, sizeof(*distToRepsQ) );
  unint *groupCountQ;
  groupCountQ = (unint*)calloc( PAD(numReps), sizeof(*groupCountQ) );
  
  computeReps(dq, rbcS.dr, repIDsQ, distToRepsQ);

  //How many points are assigned to each group?
  computeCounts(repIDsQ, m, groupCountQ);
  
  //Set up the mapping from groups to queries (qMap).
  buildQMap(q, qMap, repIDsQ, numReps, &compLength);
  
  // Setup the computation matrix.  Currently, the computation matrix is 
  // just the identity matrix: each query assigned to a particular 
  // representative is compared only to that representative's points.  
  idIntersection(cM);

  initCompPlan(&dcP, cM, groupCountQ, rbcS.groupCount, numReps);

  checkErr( cudaMalloc( (void**)&dqMap, compLength*sizeof(*dqMap) ) );
  cudaMemcpy( dqMap, qMap, compLength*sizeof(*dqMap), cudaMemcpyHostToDevice );
  
  computeNNs(rbcS.dx, rbcS.dxMap, dq, dqMap, dcP, NNs, NNdists, compLength);
  
  free(qMap);
  cudaFree(dqMap);
  freeCompPlan(&dcP);
  cudaFree(dq.mat);
  free(cM.mat);
  free(repIDsQ);
  free(distToRepsQ);
  free(groupCountQ);
}*/

/** This function is very similar to queryRBC, with a couple of
 *  basic changes to handle
  * k-nn.
  * q - query
  * rbcS - rbc structure
  * NNs - output matrix of indexes
  * NNdists - output matrix of distances
  */
void kqueryRBC(const matrix q, const ocl_rbcStruct rbcS,
               intMatrix NNs, matrix NNdists)
{
    unint m = q.r;
    unint numReps = rbcS.dr.r;
    unint compLength;
    ocl_compPlan dcP;
    unint *qMap;
    cl::Buffer dqMap;

    DBG_DEVICE_MATRIX_WRITE(rbcS.dr, "rbcS_dr.txt");
    DBG_DEVICE_MATRIX_WRITE(rbcS.dx, "rbcS_dx.txt");
    DBG_DEVICE_MATRIX_WRITE(rbcS.dxMap, "rbcS_dxMap.txt");
    DBG_HOST_MATRIX_WRITE(q, "q.txt");

    qMap = (unint*)calloc(PAD(m + (BLOCK_SIZE - 1) * PAD(numReps)),
                          sizeof(*qMap));

    ocl_matrix dq;
    copyAndMove(&dq, &q);

    DBG_DEVICE_MATRIX_WRITE(dq, "dq.txt");

    charMatrix cM;
    cM.r = cM.c = numReps;
    cM.pr = cM.pc = cM.ld = PAD(numReps);
    cM.mat = (char*)calloc(cM.pr * cM.pc, sizeof(*cM.mat));

    unint *repIDsQ;
    repIDsQ = (unint*)calloc(m, sizeof(*repIDsQ));
    real *distToRepsQ;
    distToRepsQ = (real*)calloc(m, sizeof(*distToRepsQ));
    unint *groupCountQ;
    groupCountQ = (unint*)calloc(PAD(numReps), sizeof(*groupCountQ));

    /**
    * dq - queries
    * rbcS.dr - representatives
    * repIDsQ - indexes of nearest represetatives for consecutive
    *           elements from dq
    * distToRepsQ - distances to nearest representatives
    */
    computeReps(dq, rbcS.dr, repIDsQ, distToRepsQ);

    DBG_ARRAY_WRITE(repIDsQ, m, "repIDsQ.txt");
    DBG_ARRAY_WRITE(distToRepsQ, m, "distToRepsQ.txt");


    /** How many points are assigned to each group?
    * m - numer of query points
    * groupCountQ - representative occurence histogram
    */
    computeCounts(repIDsQ, m, groupCountQ);

    /** Set up the mapping from groups to queries (qMap). */
    buildQMap(q, qMap, repIDsQ, numReps, &compLength);

    printf("comp len: %u\n", compLength);

    /** Setup the computation matrix.  Currently, the computation matrix is
      * just the identity matrix: each query assigned to a particular
      * representative is compared only to that representative's points.
      *
      * NOTE: currently, idIntersection is the *only* computation matrix
      * that will work properly with k-nn search (this is not true for 1-nn above).
      */
    idIntersection(cM);

    initCompPlan(&dcP, cM, groupCountQ, rbcS.groupCount, numReps);

    cl::Context& context = OclContextHolder::context;
    cl::CommandQueue& queue = OclContextHolder::queue;

    int byte_size = compLength*sizeof(unint);
    cl_int err;

    dqMap = cl::Buffer(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    err = queue.enqueueWriteBuffer(dqMap, CL_TRUE, 0, byte_size, qMap);
    checkErr(err);

    computeKNNs(rbcS.dx, rbcS.dxMap, dq, dqMap, dcP, NNs, NNdists, compLength);

    free(qMap);
    freeCompPlan(&dcP);
    free(cM.mat);
    free(repIDsQ);
    free(distToRepsQ);
    free(groupCountQ);
}

void meanshiftKQueryRBC(const matrix input, const ocl_rbcStruct rbcS,
                        ocl_matrix output_means, const cl::Buffer &pilots,
                        cl::Buffer& newPilots, int maxPointsNum)
{
    unint m = input.r;
    unint numReps = rbcS.dr.r;
    unint compLength;
    ocl_compPlan dcP;
    unint *qMap;
    cl::Buffer dqMap;

    DBG_DEVICE_MATRIX_WRITE(rbcS.dr, "rbcS_dr.txt");
    DBG_DEVICE_MATRIX_WRITE(rbcS.dx, "rbcS_dx.txt");
    DBG_DEVICE_MATRIX_WRITE(rbcS.dxMap, "rbcS_dxMap.txt");
    DBG_HOST_MATRIX_WRITE(input, "q.txt");

    qMap = (unint*)calloc(PAD(m + (BLOCK_SIZE - 1) * PAD(numReps)),
                          sizeof(*qMap));

    ocl_matrix dq;
    copyAndMove(&dq, &input);

    DBG_DEVICE_MATRIX_WRITE(dq, "dq.txt");

    charMatrix cM;
    cM.r = cM.c = numReps;
    cM.pr = cM.pc = cM.ld = PAD(numReps);
    cM.mat = (char*)calloc(cM.pr * cM.pc, sizeof(*cM.mat));

    unint *repIDsQ;
    repIDsQ = (unint*)calloc(m, sizeof(*repIDsQ));
    real *distToRepsQ;
    distToRepsQ = (real*)calloc(m, sizeof(*distToRepsQ));
    unint *groupCountQ;
    groupCountQ = (unint*)calloc(PAD(numReps), sizeof(*groupCountQ));

    /**
    * dq - queries
    * rbcS.dr - representatives
    * repIDsQ - indexes of nearest represetatives for consecutive
    *           elements from dq
    * distToRepsQ - distances to nearest representatives
    */
    computeReps(dq, rbcS.dr, repIDsQ, distToRepsQ);

    DBG_ARRAY_WRITE(repIDsQ, m, "repIDsQ.txt");
    DBG_ARRAY_WRITE(distToRepsQ, m, "distToRepsQ.txt");


    /** How many points are assigned to each group?
    * m - numer of query points
    * groupCountQ - representative occurence histogram
    */
    computeCounts(repIDsQ, m, groupCountQ);

    /** Set up the mapping from groups to queries (qMap). */
    buildQMap(input, qMap, repIDsQ, numReps, &compLength);

    printf("comp len: %u\n", compLength);

    /** Setup the computation matrix.  Currently, the computation matrix is
      * just the identity matrix: each query assigned to a particular
      * representative is compared only to that representative's points.
      *
      * NOTE: currently, idIntersection is the *only* computation matrix
      * that will work properly with k-nn search (this is not true for 1-nn above).
      */
    idIntersection(cM);

    initCompPlan(&dcP, cM, groupCountQ, rbcS.groupCount, numReps);

    cl::Context& context = OclContextHolder::context;
    cl::CommandQueue& queue = OclContextHolder::queue;

    int byte_size = compLength*sizeof(unint);
    cl_int err;

    dqMap = cl::Buffer(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    err = queue.enqueueWriteBuffer(dqMap, CL_TRUE, 0, byte_size, qMap);
    checkErr(err);

//    computeKNNs(rbcS.dx, rbcS.dxMap, dq, dqMap, dcP, NNs, NNdists, compLength);

    byte_size = sizeof(unint) * input.pr * maxPointsNum;

    cl::Buffer selectedPoints(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    byte_size = sizeof(unint) * input.pr;

    cl::Buffer selectedPointsNum(context, CL_MEM_READ_WRITE,
                                 byte_size, 0, &err);
    checkErr(err);

    meanshiftComputeKNNs(rbcS.dx, rbcS.dxMap, dq, dqMap, dcP, pilots,
                         selectedPoints, selectedPointsNum, newPilots,
                         maxPointsNum, compLength);

    ocl_matrix output;
    output.r = input.r;
    output.c = input.c;
    output.pr = input.pr;
    output.pc = input.pc;
    output.ld = input.ld;

    byte_size = output.pr * output.pc * sizeof(real);

    output.mat = cl::Buffer(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    meanshiftMeanWrap(dq, selectedPoints, selectedPointsNum, maxPointsNum, output);

//dbg

    byte_size = sizeof(unint) * input.pr;

    unint* selectedNumPointsHost = new unint[input.pr];

    err = queue.enqueueReadBuffer(selectedPointsNum, CL_TRUE, 0,
                                  byte_size, selectedNumPointsHost, 0, 0);
    checkErr(err);

    for(int i = 0; i < input.pr; ++i)
    {
        if(i < 20)
            std::cout << "max num points: "
                      << selectedNumPointsHost[i] << std::endl;

        if(selectedNumPointsHost[i] < 0 || selectedNumPointsHost[i] > maxPointsNum)
        {
            std::cout << "max num points[" << i << "]: "
                      << selectedNumPointsHost[i] << std::endl;
        }
    }


    delete selectedNumPointsHost;


    free(qMap);
    freeCompPlan(&dcP);
    free(cM.mat);
    free(repIDsQ);
    free(distToRepsQ);
    free(groupCountQ);
}

void computePilots(const matrix q, const cl::Buffer &repsPilots,
                   const ocl_rbcStruct& rbcS, cl::Buffer &pilots)
{
    unint m = q.r;
    unint numReps = rbcS.dr.r;

    ocl_matrix dq;
    copyAndMove(&dq, &q);

    unint *repIDsQ;
    repIDsQ = (unint*)calloc(m, sizeof(*repIDsQ));
    real *distToRepsQ;
    distToRepsQ = (real*)calloc(m, sizeof(*distToRepsQ));
    unint *groupCountQ;
    groupCountQ = (unint*)calloc(PAD(numReps), sizeof(*groupCountQ));

    cl::Context& context = OclContextHolder::context;
    //cl::CommandQueue& queue = OclContextHolder::queue;

    int byte_size = dq.pr * sizeof(unint);
    cl_int err;
    cl::Buffer indexes(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    computeRepsNoHost(dq, rbcS.dr, indexes);

    bindPilots(indexes, repsPilots, pilots, m);

}

void bindPilots(const cl::Buffer &indexes, const cl::Buffer &repsPilots,
                cl::Buffer &pilots, int pilots_size)
{
    int kernel_size = 256;

    cl::NDRange local(kernel_size, 1);

    cl::NDRange global(((pilots_size + kernel_size - 1)
                        / kernel_size) * kernel_size, 1);

    cl::CommandQueue& queue = OclContextHolder::queue;
    cl::Kernel& bindPilotsKernel = OclContextHolder::bindPilotsKernel;

    bindPilotsKernel.setArg(0, indexes);
    bindPilotsKernel.setArg(1, repsPilots);
    bindPilotsKernel.setArg(2, pilots);
    bindPilotsKernel.setArg(3, pilots_size);

    queue.enqueueNDRangeKernel(bindPilotsKernel, cl::NullRange,
                               global, local);
}


/** Building rbc structure
 *  x - input points (database)
 *  rbcS - output rbc structure
 *  numReps - number of representatives
 *  s - number of points assigned to each representative.
 */
void buildRBC(const matrix x, ocl_rbcStruct *rbcS, unint numReps, unint s,
              cl::Buffer pilots, int threshold)
{
    bool computePilots = threshold != 0;

    unint n = x.pr;
    intMatrix xmap;

    setupReps(x, rbcS, numReps);
    copyAndMove(&rbcS->dx, &x);

    xmap.r = numReps;
    xmap.pr = PAD(numReps);
    xmap.c = s;
    xmap.pc = xmap.ld = PAD(s);
    xmap.mat = (unint*)calloc( xmap.pr*xmap.pc, sizeof(*xmap.mat) );
    copyAndMoveI(&rbcS->dxMap, &xmap);

    rbcS->groupCount = (unint*)calloc( PAD(numReps), sizeof(*rbcS->groupCount) );
  
  //Figure out how much fits into memory

  // CHECKING AVAILABLE MEMORY OMITED!

    size_t memFree, memTot;
//  cudaMemGetInfo(&memFree, &memTot);
//  memFree = (unint)(((float)memFree)*MEM_USABLE);
//  /* mem needed per rep:
//   *  n*sizeof(real) - dist mat
//   *  n*sizeof(char) - dir
//   *  n*sizeof(int)  - dSums
//   *  sizeof(real)   - dranges
//   *  sizeof(int)    - dCnts
//   *  MEM_USED_IN_SCAN - memory used internally
//   */

 // memFree = 1024 * 1024 * 1024;
 // memTot = 1024 * 1024 * 1024;

//    unint ptsAtOnce = 1024 * 64;//DPAD(memFree/((n+1)*sizeof(real) + n*sizeof(char) + (n+1)*sizeof(unint) + 2*MEM_USED_IN_SCAN(n)));
   unint ptsAtOnce = 512;
   // unint ptsAtOnce = 512;//DPAD(memFree/((n+1)*sizeof(real) + n*sizeof(char) + (n+1)*sizeof(unint) + 2*MEM_USED_IN_SCAN(n)));
    if(!ptsAtOnce)
    {
        fprintf(stderr,"error: %lu is not enough memory to build the RBC.. exiting\n", (unsigned long)memFree);
        exit(1);
    }

    cl::Context& context = OclContextHolder::context;
    cl::CommandQueue& queue = OclContextHolder::queue;
    cl_int err;

    //Now set everything up for the scans
    ocl_matrix dD;
    dD.pr = dD.r = ptsAtOnce;         /** IN MOST CASES NUM OF REPRESENTATIVES SHOULD BE USED */
    dD.c = rbcS->dx.r;
    dD.pc = rbcS->dx.pr;
    dD.ld = dD.pc;
    //checkErr( cudaMalloc( (void**)&dD.mat, dD.pr*dD.pc*sizeof(*dD.mat) ) );
    int byte_size = dD.pr*dD.pc*sizeof(real);
    dD.mat = cl::Buffer(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    //real *dranges;
    //checkErr( cudaMalloc( (void**)&dranges, ptsAtOnce*sizeof(real) ) );
    byte_size = ptsAtOnce*sizeof(real);
    cl::Buffer dranges(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    charMatrix ir;
    ir.r=dD.r; ir.pr=dD.pr; ir.c=dD.c; ir.pc=dD.pc; ir.ld=dD.ld;
    /** not necessary host memory allocation */
    //ir.mat = (char*)calloc( ir.pr*ir.pc, sizeof(*ir.mat) );
    ir.mat = NULL;

    ocl_charMatrix dir;
    copyAndMoveC(&dir, &ir);

    ocl_intMatrix dSums; //used to compute memory addresses.
    dSums.r=dir.r; dSums.pr=dir.pr; dSums.c=dir.c; dSums.pc=dir.pc; dSums.ld=dir.ld;
    //checkErr( cudaMalloc( (void**)&dSums.mat, dSums.pc*dSums.pr*sizeof(*dSums.mat) ) );
    byte_size = dSums.pc*dSums.pr*sizeof(int);
    dSums.mat = cl::Buffer(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    //  unint *dCnts;
    //  checkErr( cudaMalloc( (void**)&dCnts, ptsAtOnce*sizeof(*dCnts) ) );
    byte_size = ptsAtOnce*sizeof(unint);
    cl::Buffer dCnts(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    //Do the scans to build the dxMap
    unint numLeft = rbcS->dr.r; //points left to process
    unint row = 0; //base row for iteration of while loop
    unint pi, pip; //pi=pts per it, pip=pad(pi)

    while(numLeft > 0)
    {
        pi = MIN(ptsAtOnce, numLeft);  //points to do this iteration.
        pip = PAD(pi);

        dD.r = pi; dD.pr = pip;
        dir.r = pi; dir.pr = pip;
        dSums.r = pi; dSums.pr = pip;

        /** compute the distance matrix
         *  rbcS->dr - matrix of representatives (choosen from input data)
         *  rbcS->dx - matrix of input data
         *  dD - matrix of distances
         */
        distSubMat(rbcS->dr, rbcS->dx, dD, row, pip);


        DBG_DEVICE_MATRIX_WRITE(dD, "distances.txt");
        DBG_DEVICE_MATRIX_WRITE(rbcS->dr, "srbS_dr0.txt");

        /** find an appropriate range
         *  dD - matrix of distances
         *  dranges - vector of maximal value for each row
         *  s - desired number of values within a range
         */
        findRangeWrap(dD, dranges, s, 0);

        if(computePilots)
            findRangeWrap(dD, pilots, threshold, row);

        /** set binary vector for points in range
         *  dD - matrix of disances
         *  dranges - buffer of range upper bounds
         *  dir - matrix (the same size as dD) containing binary indication
         *        if corresponding value from dD belongs to the range
         */
        rangeSearchWrap(dD, dranges, dir);

        DBG_DEVICE_MATRIX_WRITE(dir, "bin_vec.txt");

        sumWrap(dir, dSums);  //This and the next call perform the parallel compaction.

        DBG_DEVICE_MATRIX_WRITE(dSums, "dSums.txt");

        buildMapWrap(rbcS->dxMap, dir, dSums, row);

        DBG_DEVICE_MATRIX_WRITE(rbcS->dxMap, "dxMap.txt");

        /** How many points are assigned to each rep?  It is not
          *exactly* s, which is why we need to compute this. */
        getCountsWrap(dCnts, dir, dSums);

        DBG_DEVICE_UINT_BUFF_WRITE(dCnts, pi, "dCnts.txt");

        err = queue.enqueueReadBuffer(dCnts, CL_TRUE, 0,
                                      pi * sizeof(unint), &(rbcS->groupCount[row]));

        checkErr(err);

        numLeft -= pi;
        row += pi;
    }
}


/** Choose representatives and move them to device */
void setupReps(matrix x, ocl_rbcStruct *rbcS, unint numReps)
{
    unint i;
    unint *randInds;
    randInds = (unint*)calloc(PAD(numReps), sizeof(*randInds));
    subRandPerm(numReps, x.r, randInds);

    matrix r;
    r.r = numReps; r.pr = PAD(numReps);
    r.c = x.c; r.pc = r.ld = PAD(r.c);
    r.mat = (real*)calloc( r.pr*r.pc, sizeof(*r.mat) );

    for(i = 0;i < numReps; i++)
    {
        copyVector(&r.mat[IDX(i,0,r.ld)],
                   &x.mat[IDX(randInds[i],0,x.ld)], x.c);
    }

    copyAndMove(&rbcS->dr, &r);

    free(randInds);
    free(r.mat);
}


/** Assign each point in dq to its nearest point in dr.
  *
  */
void computeReps(const ocl_matrix& dq, const ocl_matrix& dr,
                 unint *repIDs, real *distToReps)
{
    cl::Context& context = OclContextHolder::context;
    cl::CommandQueue& queue = OclContextHolder::queue;

    int byte_size = dq.pr*sizeof(real);
    cl_int err;

    cl::Buffer dMins(context, CL_TRUE, byte_size, 0, &err);
    checkErr(err);

    byte_size = dq.pr*sizeof(unint);

    cl::Buffer dMinIDs(context, CL_TRUE, byte_size, 0, &err);
    checkErr(err);

    nnWrap(dq, dr, dMins, dMinIDs);

    // POTENTIAL PERFORMANCE BOTTLENECK:

    byte_size = dq.r*sizeof(real);
    err = queue.enqueueReadBuffer(dMins, CL_TRUE, 0, byte_size, distToReps);
    checkErr(err);

    byte_size = dq.r*sizeof(unint);
    err = queue.enqueueReadBuffer(dMinIDs, CL_TRUE, 0, byte_size, repIDs);
    checkErr(err);
}


void computeRepsNoHost(const ocl_matrix& dq, const ocl_matrix& dr,
                       cl::Buffer& indexes)
{
    cl::Context& context = OclContextHolder::context;
    cl::CommandQueue& queue = OclContextHolder::queue;

    int byte_size = dq.pr*sizeof(real);
    cl_int err;

    /** this buffer in unnecessary, nnWrap version is needed */
    cl::Buffer dMins(context, CL_TRUE, byte_size, 0, &err);
    checkErr(err);

    nnWrap(dq, dr, dMins, indexes);
}



//Assumes radii is initialized to 0s
void computeRadii(unint *repIDs, real *distToReps, real *radii, unint n, unint numReps){
  unint i;

  for(i=0;i<n;i++)
    radii[repIDs[i]] = MAX(distToReps[i],radii[repIDs[i]]);
}


//Assumes groupCount is initialized to 0s
void computeCounts(unint *repIDs, unint n, unint *groupCount){
  unint i;
  
  for(i=0;i<n;i++)
    groupCount[repIDs[i]]++;
}


void buildQMap(matrix q, unint *qMap, unint *repIDs,
               unint numReps, unint *compLength)
{
    unint n=q.r;
    unint i;
    unint *gS; //groupSize

    gS = (unint*)calloc(numReps+1,sizeof(*gS));

    /** histogram */
    for(i = 0; i < n; i++)
        gS[repIDs[i]+1]++;

    /** padding */
    for(i = 0; i < numReps + 1; i++)
        gS[i] = PAD(gS[i]);

    /** exclusive prefix sum */
    for(i = 1; i < numReps + 1; i++)
        gS[i] = gS[i - 1] + gS[i];

    /** number of queries after padding */
    *compLength = gS[numReps];

    /** map initialization */
    for(i = 0; i < (*compLength); i++)
        qMap[i] = DUMMY_IDX;

    for(i = 0; i < n; i++)
    {
        qMap[gS[repIDs[i]]] = i;
        gS[repIDs[i]]++;
    }

    free(gS);
}


// Sets the computation matrix to the identity.  
void idIntersection(charMatrix cM)
{
    for(unint i = 0; i < cM.r; i++)
    {
        if(i < cM.c)
            cM.mat[IDX(i, i, cM.ld)] = 1;
    }
}


void fullIntersection(charMatrix cM)
{
    for(int i = 0; i < cM.r; i++)
    {
        for(int j = 0; j < cM.c; j++)
        {
            cM.mat[IDX(i,j,cM.ld)]=1;
        }
    }
}

//NEVER USED
/*
void computeNNs(matrix dx, intMatrix dxMap, matrix dq, unint *dqMap, compPlan dcP, unint *NNs, real *NNdists, unint compLength){
  real *dNNdists;
  unint *dMinIDs;
  
  checkErr( cudaMalloc((void**)&dNNdists,compLength*sizeof(*dNNdists)) );
  checkErr( cudaMalloc((void**)&dMinIDs,compLength*sizeof(*dMinIDs)) );

  planNNWrap(dq, dqMap, dx, dxMap, dNNdists, dMinIDs, dcP, compLength );
  cudaMemcpy( NNs, dMinIDs, dq.r*sizeof(*NNs), cudaMemcpyDeviceToHost );
  cudaMemcpy( NNdists, dNNdists, dq.r*sizeof(*dNNdists), cudaMemcpyDeviceToHost );

  cudaFree(dNNdists);
  cudaFree(dMinIDs);
}*/

void computeKNNs(const ocl_matrix& dx, const ocl_intMatrix& dxMap,
                 const ocl_matrix& dq, const cl::Buffer& dqMap,
                 const ocl_compPlan& dcP, intMatrix NNs,
                 matrix NNdists, unint compLength)
{
    ocl_matrix dNNdists;
    ocl_intMatrix dMinIDs;
    dNNdists.r = compLength; dNNdists.pr = compLength;
    dNNdists.c = KMAX; dNNdists.pc = KMAX;
    dNNdists.ld = dNNdists.pc;

    dMinIDs.r = compLength; dMinIDs.pr = compLength;
    dMinIDs.c = KMAX; dMinIDs.pc = KMAX; dMinIDs.ld = dMinIDs.pc;

    cl::Context& context = OclContextHolder::context;
    cl::CommandQueue& queue = OclContextHolder::queue;

    int byte_size = dNNdists.pr * dNNdists.pc * sizeof(real);
    cl_int err;

    dNNdists.mat = cl::Buffer(context, CL_MEM_READ_WRITE,
                              byte_size, 0, &err);
    checkErr(err);

    byte_size = dMinIDs.pr * dMinIDs.pc * sizeof(unint);

    dMinIDs.mat = cl::Buffer(context, CL_MEM_READ_WRITE,
                             byte_size, 0, &err);
    checkErr(err);

    planKNNWrap(dq, dqMap, dx, dxMap, dNNdists, dMinIDs, dcP, compLength);

    byte_size = dq.r * KMAX*sizeof(unint);

    err = queue.enqueueReadBuffer(dMinIDs.mat, CL_TRUE,
                                  0, byte_size, NNs.mat);
    checkErr(err);

    byte_size = dq.r*KMAX*sizeof(real);

    err = queue.enqueueReadBuffer(dNNdists.mat, CL_TRUE,
                                  0, byte_size, NNdists.mat);
    checkErr(err);
}


void meanshiftComputeKNNs(const ocl_matrix& dx, const ocl_intMatrix& dxMap,
                          const ocl_matrix& dq, const cl::Buffer& dqMap,
                          const ocl_compPlan& dcP,
                          cl::Buffer windows, cl::Buffer outputMeans,
                          cl::Buffer outputMeansNums, cl::Buffer newWindows,
                          int maxPointsNum, unint compLength)
{

    meanshiftPlanKNNWrap(dq, dqMap, dx, dxMap, dcP, windows, outputMeans,
                         outputMeansNums, newWindows, maxPointsNum, compLength);

}


//This calls the dist1Kernel wrapper, but has it compute only 
//a submatrix of the all-pairs distance matrix.  In particular,
//only distances from dr[start,:].. dr[start+length-1] to all of x
//are computed, resulting in a distance matrix of size 
//length by dx.pr.  It is assumed that length is padded.
//void distSubMat(matrix dr, matrix dx, matrix dD, unint start, unint length){
void distSubMat(ocl_matrix& dr, ocl_matrix& dx, ocl_matrix &dD,
                unint start, unint length)
{
    ocl_matrix dr_tmp = dr;
    dr_tmp.r = dr_tmp.pr = length;
    unint dr_offset = IDX(start, 0, dr.ld);

    dist1Wrap(dr_tmp, dx, dD, dr_offset);
}


void destroyRBC(ocl_rbcStruct *rbcS)
{
  free(rbcS->groupCount);
}


/* Danger: this function allocates memory that it does not free.  
 * Use freeCompPlan to clear mem.  
 * See the readme.txt file for a description of why this function is needed.
 */
void initCompPlan(ocl_compPlan *dcP, const charMatrix cM,
                  const unint *groupCountQ, const unint *groupCountX,
                  unint numReps)
{
    unint i, j, k;
    unint maxNumGroups = 0;
    compPlan cP;

    unint sNumGroups = numReps;
    cP.numGroups = (unint*)calloc(sNumGroups, sizeof(*cP.numGroups));

    /** for identity matrix cM, cp.numGroups is initialized to 1,
     *  and maxNumGroups is initialized to 1
     */
    for(i = 0; i < numReps; i++)
    {
        cP.numGroups[i] = 0;

        for(j = 0; j < numReps; j++)
        {
            cP.numGroups[i] += cM.mat[IDX(i, j, cM.ld)];
        }

        maxNumGroups = MAX(cP.numGroups[i], maxNumGroups);
    }

    cP.ld = maxNumGroups;
  
    unint sQToQGroup;

    for(i = 0, sQToQGroup = 0; i < numReps; i++)
    {
        sQToQGroup += PAD(groupCountQ[i]);
    }
  
    cP.qToQGroup = (unint*)calloc( sQToQGroup, sizeof(*cP.qToQGroup));

    for(i=0, k=0; i<numReps; i++)
    {
        for(j=0; j<PAD(groupCountQ[i]); j++)
        {
            cP.qToQGroup[k++] = i;
        }
    }
  
    unint sQGroupToXGroup = numReps*maxNumGroups;
    cP.qGroupToXGroup = (unint*)calloc(sQGroupToXGroup,
                                       sizeof(*cP.qGroupToXGroup));
    unint sGroupCountX = maxNumGroups*numReps;
    cP.groupCountX = (unint*)calloc(sGroupCountX, sizeof(*cP.groupCountX));
  
    for(i = 0; i<numReps; i++)
    {
        for(j = 0, k = 0; j < numReps; j++)
        {
            if(cM.mat[IDX(i, j, cM.ld )])
            {
                cP.qGroupToXGroup[IDX(i, k, cP.ld )] = j;
                cP.groupCountX[IDX(i, k++, cP.ld)] = groupCountX[j];
            }
        }
    }

    cl::Context& context = OclContextHolder::context;
    cl::CommandQueue& queue = OclContextHolder::queue;

    cl_int err;

    int byte_size = sNumGroups*sizeof(unint);
    dcP->numGroups = cl::Buffer(context, CL_MEM_READ_WRITE,
                                byte_size, 0, &err);
    checkErr(err);
    err = queue.enqueueWriteBuffer(dcP->numGroups, CL_TRUE,
                                   0, byte_size, cP.numGroups);
    checkErr(err);

    byte_size = sGroupCountX*sizeof(unint);
    dcP->groupCountX = cl::Buffer(context, CL_MEM_READ_WRITE,
                                  byte_size, 0, &err);
    checkErr(err);
    err = queue.enqueueWriteBuffer(dcP->groupCountX, CL_TRUE,
                                   0, byte_size, cP.groupCountX);
    checkErr(err);

    byte_size = sQToQGroup*sizeof(unint);
    dcP->qToQGroup = cl::Buffer(context, CL_MEM_READ_WRITE,
                                byte_size, 0, &err);
    checkErr(err);
    err = queue.enqueueWriteBuffer(dcP->qToQGroup, CL_TRUE,
                                   0, byte_size, cP.qToQGroup);
    checkErr(err);

    byte_size = sQGroupToXGroup*sizeof(unint);
    dcP->qGroupToXGroup = cl::Buffer(context, CL_MEM_READ_WRITE,
                                     byte_size, 0, &err);
    checkErr(err);
    err = queue.enqueueWriteBuffer(dcP->qGroupToXGroup, CL_TRUE,
                                   0, byte_size, cP.qGroupToXGroup);
    checkErr(err);

    dcP->ld = cP.ld;

    free(cP.numGroups);
    free(cP.groupCountX);
    free(cP.qToQGroup);
    free(cP.qGroupToXGroup);
}


//Frees memory allocated in initCompPlan.
void freeCompPlan(ocl_compPlan *dcP){

}

#endif
