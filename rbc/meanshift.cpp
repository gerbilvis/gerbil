#include "meanshift.h"

#include <cmath>
#include <cassert>
#include <iostream>
#include <algorithm>

#include "kernelWrap.h"

void meanshift_rbc(matrix database, int numReps, int pointsPerRepresentative)
{
    int maxQuerySize = 1024;

    int database_size = database.r;

    if(numReps == 0)
        numReps = sqrt(database_size) * 5;


    cl::Context& context = OclContextHolder::context;
    cl::CommandQueue& queue = OclContextHolder::queue;

    cl_int err;
    cl::Buffer repsPilots(context, CL_MEM_READ_WRITE,
                          PAD(numReps) * sizeof(real), 0, &err);
    checkErr(err);

//    cl::Buffer newRepsPilots(context, CL_MEM_READ_WRITE,
//                             PAD(numReps) * sizeof(real), 0, &err);
//    checkErr(err);

    ocl_rbcStruct rbcS;

    std::cout << "building rbc" << std::endl;

    /** calculating RBC and pilots for representatives */
    buildRBC(database, &rbcS, numReps, pointsPerRepresentative,
             repsPilots, 512);


    cl::Buffer allPilots(context, CL_MEM_READ_WRITE,
                          PAD(database.pr) * sizeof(real), 0, &err);
    checkErr(err);

    std::cout << "computing pilot" << std::endl;

    /** calculating pilot for every point */
    computePilots(database, repsPilots, rbcS, allPilots);

    validate_pilots(database, allPilots);

//    ocl_matrix output_means;
//    copyAndMove(&output_means, &database);

//    std::cout << "querying" << std::endl;

    int byte_size = sizeof(unint) * database.pr * maxQuerySize;

    cl::Buffer selectedPoints(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    byte_size = sizeof(unint) * database.pr;

    cl::Buffer selectedPointsNum(context, CL_MEM_READ_WRITE,
                                 byte_size, 0, &err);
    checkErr(err);

    std::cout << "calculating meanshift query" << std::endl;

    meanshiftKQueryRBC(database, rbcS, allPilots,
                       selectedPoints, selectedPointsNum,
                       maxQuerySize);

    ocl_matrix database_ocl;
    copyAndMove(&database_ocl, &database);

    ocl_matrix output = database_ocl;
//    output.r = database.r;
//    output.c = database.c;
//    output.pr = database.pr;
//    output.pc = database.pc;
//    output.ld = database.ld;

    byte_size = output.pr * output.pc * sizeof(real);
    output.mat = cl::Buffer(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    std::cout << "calculating meanshift mean" << std::endl;

    meanshiftMeanWrap(database_ocl, selectedPoints, selectedPointsNum,
                      maxQuerySize, output);


    byte_size = output.pr * sizeof(real);
    cl::Buffer distances(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    std::cout << "calculating distances" << std::endl;

    simpleDistanceKernelWrap(database_ocl, output, distances);

    validate_query_and_mean(database, selectedPoints, selectedPointsNum, allPilots,
                   maxQuerySize, output, distances);
}

bool sort_function(real i, real j)
{
    return (i < j);
}

void validate_pilots(matrix database, cl::Buffer pilots)
{
    cl::CommandQueue& queue = OclContextHolder::queue;

    queue.finish();

    std::cout << "validating pilots..." << std::endl;

    int tests = 10;
    int db_size = database.r;

    real* pilots_host = new real[db_size];
    real* distances = new real[db_size];

    cl_int err;

    err = queue.enqueueReadBuffer(pilots, CL_TRUE, 0,
                                  db_size * sizeof(real), pilots_host);
    checkErr(err);

    for(int i = 0; i < tests; ++i)
    {
        int candidate = rand() % db_size;

        for(int j = 0; j < db_size; ++j)
        {
            real distance = 0.f;

            for(int k = 0; k < database.c; ++k)
            {
                real a = database.mat[IDX(j, k, database.ld)];
                real b = database.mat[IDX(candidate, k, database.ld)];

                distance += (a - b) * (a - b);
            }

            distances[j] = distance;
        }

        std::sort(distances, distances + db_size, sort_function);

        std::cout << "pilot host  : " << distances[512] << std::endl;
        std::cout << "pilot device: " << pilots_host[candidate] << std::endl;
    }

    delete[] pilots_host;
    delete[] distances;
}


void validate_query_and_mean(matrix database, cl::Buffer selectedPoints,
                    cl::Buffer selectedPointsNum, cl::Buffer pilots,
                    int maxQuerySize, ocl_matrix means,
                    cl::Buffer result_distances)
{
    srand(23);

    cl::CommandQueue& queue = OclContextHolder::queue;
    queue.finish();

    std::cout << "validating query..." << std::endl;

    int tests = 16;
    int db_size = database.r;

    int selPointsSize = db_size * maxQuerySize;
    int meansSize = means.pc * means.pr;

    unint* selectedPointsHost = new unint[selPointsSize];
    unint* selectedPointsNumHost = new unint[db_size];
    real* pilots_host = new real[db_size];
    real* distances = new real[db_size];
    real* means_host = new real[meansSize];
    real* mean = new real[database.c];
    real* result_distances_host = new real[database.r];

    cl_int err;

    std::cout << "downloading selected points" << std::endl;
    std::cout << (selPointsSize / (1024 * 1024) * sizeof(unint))
              << " MB" << std::endl;

    err = queue.enqueueReadBuffer(selectedPoints, CL_TRUE, 0,
                                  selPointsSize * sizeof(unint),
                                  selectedPointsHost);
    checkErr(err);

    std::cout << "downloading numbers of selected points" << std::endl;

    err = queue.enqueueReadBuffer(selectedPointsNum, CL_TRUE, 0,
                                  db_size * sizeof(unint),
                                  selectedPointsNumHost);
    checkErr(err);

    std::cout << "downloading pilots" << std::endl;

    err = queue.enqueueReadBuffer(pilots, CL_TRUE, 0,
                                  db_size * sizeof(real), pilots_host);
    checkErr(err);

    std::cout << "downloading means";

    err = queue.enqueueReadBuffer(means.mat, CL_TRUE, 0,
                                  meansSize * sizeof(real), means_host);
    checkErr(err);

    std::cout << "downloading distances";

    err = queue.enqueueReadBuffer(result_distances, CL_TRUE, 0,
                                  database.r * sizeof(real),
                                  result_distances_host);
    checkErr(err);


    for(int i = 0; i < tests; ++i)
    {
        int candidate = rand() % db_size;

        std::fill(mean, mean + database.c, 0.f);

        std::cout << "test: " << i << ", random candidate: "
                  << candidate << std::endl;

        /** calculating distances to given candidate */
        for(int j = 0; j < db_size; ++j)
        {
            real distance = 0.f;

            for(int k = 0; k < database.c; ++k)
            {
                real a = database.mat[IDX(j, k, database.ld)];
                real b = database.mat[IDX(candidate, k, database.ld)];

                distance += (a - b) * (a - b);
            }

            distances[j] = distance;
        }

        std::cout << "host:" << std::endl;

        int counter = 0;

        /** checking if distance satisfies inequality */
        for(int j = 0; j < db_size; ++j)
        {
            real threshold = pilots_host[j];

            if(distances[j] < threshold)
            {
                std::cout << j << " ";
                counter++;

                /** adding value to the mean */
                for(int k = 0; k < database.c; ++k)
                {
                    mean[k] += database.mat[IDX(j, k, database.ld)];
                }
            }
        }
        std::cout << std::endl;
        std::cout << "num points host: " << counter << std::endl;
        std::cout << "host sum:" << std::endl;

        for(int j = 0; j < database.c; ++j)
        {
            std::cout << mean[j] << ", ";
        }

        std::cout << std::endl;

        std::cout << "host mean:" << std::endl;

        for(int j = 0; j < database.c; ++j)
        {
            std::cout << mean[j] / counter << ", ";
        }

        std::cout << std::endl;

        std::cout << "device:" << std::endl;

        int deviceNum = selectedPointsNumHost[candidate];

        assert(deviceNum <= maxQuerySize);

        for(int j = 0; j < deviceNum; ++j)
        {
            std::cout << selectedPointsHost[candidate * maxQuerySize + j]
                    << " ";
        }

        std::cout << std::endl;
        std::cout << "num points device: " << deviceNum << std::endl;

        std::cout << "device mean:" << std::endl;

        for(int j = 0; j < database.c; ++j)
        {
            std::cout << means_host[candidate * means.ld + j] << ", ";
        }

        std::cout << std::endl;

        std::cout << "result distance: " << result_distances_host[candidate]
                     << std::endl;

        std::cout << std::endl;

        std::cout << std::endl;

        //std::sort(distances, distances + db_size, sort_function);
    }

    double distances_sum = 0.0;

    for(int i = 0; i < database.r; ++i)
    {
        distances_sum += result_distances_host[i];

        std::cout << "avg result distance: " << distances_sum / database.r
                     << std::endl;
    }

    std::cout << "avg result distance: " << distances_sum / database.r
                 << std::endl;

    delete[] selectedPointsHost;
    delete[] selectedPointsNumHost;
    delete[] pilots_host;
    delete[] distances;
    delete[] mean;
}


