/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef UTILSGPU_CU
#define UTILSGPU_CU

//#include<cuda.h>
#include<stdio.h>
#include "defs.h"
#include "utilsGPU.h"

#include <iostream>
#include <sstream>

cl::Context OclContextHolder::context;
cl::CommandQueue OclContextHolder::queue;

cl::Kernel OclContextHolder::dist1Kernel;
cl::Kernel OclContextHolder::findRangeKernel;
cl::Kernel OclContextHolder::rangeSearchKernel;
cl::Kernel OclContextHolder::sumKernel;
cl::Kernel OclContextHolder::sumKernelI;
cl::Kernel OclContextHolder::combineSumKernel;
cl::Kernel OclContextHolder::buildMapKernel;
cl::Kernel OclContextHolder::getCountsKernel;
cl::Kernel OclContextHolder::planKNNKernel;
cl::Kernel OclContextHolder::nnKernel;
cl::Kernel OclContextHolder::nn32Kernel;
cl::Kernel OclContextHolder::bindPilotsKernel;


extern const char* rbc;

void OclContextHolder::oclInit()
{
    std::vector<cl::Platform> platforms;
    cl_int err = CL_SUCCESS;

    cl_device_type type = CL_DEVICE_TYPE_ALL;

    cl::Platform::get(&platforms);

    if (!platforms.size())
    {
        std::cout << "Platform size 0" << std::endl;
    }
    else
    {
        std::cout << "Platforms size: " << platforms.size() << std::endl;
        std::string platform_name = platforms[0].getInfo<CL_PLATFORM_NAME>();

        std::cout << "Platform name: " << platform_name << std::endl;
        std::cout << "Platform name: "
                  << platforms[0].getInfo<CL_PLATFORM_NAME>() << std::endl;
    }

    cl_context_properties properties[] =
    {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};

    context = cl::Context(type, properties, NULL, NULL, &err);

    //std::cout << (err == CL_SUCCESS ? "true" : "false") << std::endl;

    int num_devices = context.getInfo<CL_CONTEXT_NUM_DEVICES>();

    std::cout << "num devices: " << num_devices << std::endl;

    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    for(int i = 0; i < devices.size(); ++i){

        cl::Device& device = devices[i];

        std::cout << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    }

    queue = cl::CommandQueue(context, devices[0]);


    // INITIALIZING KERNELS

    std::vector<std::string> source_codes;
    source_codes.push_back(std::string(rbc));

//    cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();

    std::stringstream stream;
//    stream << "-DX_DIM=" << (neuron_size_rounded / 2);
//    stream << " -DSOM_SIZE_X=" << d_data.x;
//    stream << " -DSOM_SIZE_Y=" << d_data.y;
//    stream << " -DSOM_SIZE_Z=" << d_data.z;
//    stream << " -DVECTOR_SIZE=" << neuron_size_rounded;

//#ifdef DEBUG_MODE
    stream << " -DDEBUG_MODE -Werror";
//#endif

    cl::Program::Sources sources;

    for(std::vector<std::string>::const_iterator it = source_codes.begin();
        it != source_codes.end(); ++it)
    {
        const char* code = it->c_str();
        sources.push_back(std::make_pair(code, strlen(code)));
    }

    cl::Program program = cl::Program(context, sources);

    std::vector<cl::Device> context_devices = context.getInfo<CL_CONTEXT_DEVICES>();

    try{
        err = program.build(context_devices, stream.str().c_str());
    }catch(cl::Error& err)
    {
        std::cout << std::endl;
        std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context_devices[0]);

        throw err;
    }

    if(err != CL_SUCCESS)
    {
        std::cout << std::endl;
        std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context_devices[0]);
    }

//    // Make kernels
    dist1Kernel = cl::Kernel(program, "dist1Kernel");
    findRangeKernel = cl::Kernel(program, "findRangeKernel");
    rangeSearchKernel = cl::Kernel(program, "rangeSearchKernel");
    sumKernel = cl::Kernel(program, "sumKernel");
    sumKernelI = cl::Kernel(program, "sumKernelI");
    combineSumKernel = cl::Kernel(program, "combineSumKernel");
    buildMapKernel = cl::Kernel(program, "buildMapKernel");
    getCountsKernel = cl::Kernel(program, "getCountsKernel");
    planKNNKernel = cl::Kernel(program, "planKNNKernel");
    nnKernel = cl::Kernel(program, "nnKernel");
    nn32Kernel = cl::Kernel(program, "nn32Kernel");
    bindPilotsKernel = cl::Kernel(program, "bindPilotsKernel");
}




template<typename DEVICE_MATRIX, typename HOST_MATRIX>
void genericCopyAndMove(DEVICE_MATRIX *dx, HOST_MATRIX *x)
{
    dx->r = x->r;
    dx->c = x->c;
    dx->pr = x->pr;
    dx->pc = x->pc;
    dx->ld = x->ld;

    cl::Context& context = OclContextHolder::context;
    cl::CommandQueue& queue = OclContextHolder::queue;

    cl_int err;


    int byte_size =  dx->pr*dx->pc*sizeof(*(x->mat));
    dx->mat = cl::Buffer(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    if(x->mat)
    {
        err = queue.enqueueWriteBuffer(dx->mat, CL_TRUE, 0, byte_size, x->mat);
        checkErr(err);
    }
}


void copyAndMove(ocl_matrix *dx, const matrix *x)
{
    genericCopyAndMove(dx, x);
}


void copyAndMoveI(ocl_intMatrix *dx, const intMatrix *x)
{
    genericCopyAndMove(dx, x);
}


void copyAndMoveC(ocl_charMatrix *dx, const charMatrix *x)
{
    genericCopyAndMove(dx, x);
}


void device_matrix_to_file(const ocl_matrix& mat, const char* filetxt)
{
    FILE *fp = fopen(filetxt,"w");
    if( !fp ){
      fprintf(stderr, "can't open output file\n");
      return;
    }

    int total_size = mat.pr * mat.pc;

    int total_size_byte = total_size * sizeof(float);

    float *mem = new float[total_size];

    cl::CommandQueue& queue = OclContextHolder::queue;

    queue.enqueueReadBuffer(mat.mat, CL_TRUE, 0, total_size_byte, mem);

    for(int i = 0; i < mat.r; ++i)
    {
        for(int j = 0; j < mat.c; ++j)
        {
            //int index = i * mat.pr + j;
            fprintf( fp, "%f ", mem[IDX(i,j, mat.ld)]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);

    delete[] mem;
}

void device_matrix_to_file(const ocl_intMatrix& mat, const char* filetxt)
{
    FILE *fp = fopen(filetxt,"w");
    if(!fp)
    {
        fprintf(stderr, "can't open output file\n");
        return;
    }

    int total_size = mat.pr * mat.pc;

    int total_size_byte = total_size * sizeof(unint);

    unint *mem = new unint[total_size];

    cl::CommandQueue& queue = OclContextHolder::queue;

    queue.enqueueReadBuffer(mat.mat, CL_TRUE, 0, total_size_byte, mem);

    for(int i = 0; i < mat.r; ++i)
    {
        for(int j = 0; j < mat.c; ++j)
        {
            //int index = i * mat.pr + j;
            fprintf( fp, "%u ", mem[IDX(i,j, mat.ld)]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    delete[] mem;
}

void device_matrix_to_file(const ocl_charMatrix& mat, const char* filetxt)
{
    FILE *fp = fopen(filetxt,"w");
    if( !fp ){
      fprintf(stderr, "can't open output file\n");
      return;
    }

    int total_size = mat.pr * mat.pc;

    int total_size_byte = total_size * sizeof(char);

    char *mem = new char[total_size];

    cl::CommandQueue& queue = OclContextHolder::queue;

    queue.enqueueReadBuffer(mat.mat, CL_TRUE, 0, total_size_byte, mem);

    for(int i = 0; i < mat.r; ++i)
    {
        for(int j = 0; j < mat.c; ++j)
        {
            //int index = i * mat.pr + j;
            fprintf( fp, "%u ", mem[IDX(i,j, mat.ld)]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);

    delete[] mem;
}

void matrix_to_file(const matrix& mat, const char* filetxt)
{
    FILE *fp = fopen(filetxt,"w");
    if( !fp ){
      fprintf(stderr, "can't open output file\n");
      return;
    }

    real *mem = mat.mat;

    for(int i = 0; i < mat.r; ++i)
    {
        for(int j = 0; j < mat.c; ++j)
        {
            //int index = i * mat.pr + j;
            fprintf( fp, "%f ", mem[IDX(i,j, mat.ld)]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

void matrix_to_file(const intMatrix& mat, const char* filetxt)
{
    FILE *fp = fopen(filetxt,"w");
    if( !fp ){
      fprintf(stderr, "can't open output file\n");
      return;
    }

    unint *mem = mat.mat;

    for(int i = 0; i < mat.r; ++i)
    {
        for(int j = 0; j < mat.c; ++j)
        {
            //int index = i * mat.pr + j;
            fprintf( fp, "%u ", mem[IDX(i,j, mat.ld)]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}


//void copyAndMove(matrix *dx, const matrix *x){
//  dx->r = x->r;
//  dx->c = x->c;
//  dx->pr = x->pr;
//  dx->pc = x->pc;
//  dx->ld = x->ld;

//  checkErr( cudaMalloc( (void**)&(dx->mat), dx->pr*dx->pc*sizeof(*(dx->mat)) ) );
//  cudaMemcpy( dx->mat, x->mat, dx->pr*dx->pc*sizeof(*(dx->mat)), cudaMemcpyHostToDevice );
  
//}


//void copyAndMoveI(intMatrix *dx, const intMatrix *x){
//  dx->r = x->r;
//  dx->c = x->c;
//  dx->pr = x->pr;
//  dx->pc = x->pc;
//  dx->ld = x->ld;

//  checkErr( cudaMalloc( (void**)&(dx->mat), dx->pr*dx->pc*sizeof(*(dx->mat)) ) );
//  cudaMemcpy( dx->mat, x->mat, dx->pr*dx->pc*sizeof(*(dx->mat)), cudaMemcpyHostToDevice );
  
//}


//void copyAndMoveC(charMatrix *dx, const charMatrix *x){
//  dx->r = x->r;
//  dx->c = x->c;
//  dx->pr = x->pr;
//  dx->pc = x->pc;
//  dx->ld = x->ld;

//  checkErr( cudaMalloc( (void**)&(dx->mat), dx->pr*dx->pc*sizeof(*(dx->mat)) ) );
//  cudaMemcpy( dx->mat, x->mat, dx->pr*dx->pc*sizeof(*(dx->mat)), cudaMemcpyHostToDevice );
  
//}


void checkErr(cl_int cError){
  if(cError != CL_SUCCESS){
  //  fprintf(stderr,"GPU-related error:\n\t%s \n", cudaGetErrorString(cError) );
    fprintf(stderr,"ocl error! ..\n");
    fprintf(stderr,"exiting ..\n");
    exit(1);
  }
}

void checkErr(char* loc, cl_int cError){
  printf("in %s:\n",loc);
  checkErr(cError);
}

#endif
