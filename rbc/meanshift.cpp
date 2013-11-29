#include "meanshift.h"

#include <cmath>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <set>
#include <map>

#include <QImage>

#include "kernelWrap.h"

void meanshift_rbc(matrix database, int numReps, int pointsPerRepresentative)
{
    int maxQuerySize = 512;

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

    /** preparing memory for meanshift query */

    int byte_size = sizeof(unint) * database.pr * maxQuerySize;

    cl::Buffer selectedPoints(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    byte_size = sizeof(unint) * database.pr;

    cl::Buffer selectedPointsNum(context, CL_MEM_READ_WRITE,
                                 byte_size, 0, &err);
    checkErr(err);

    /** preparing memory for input and output means */

    ocl_matrix database_ocl;
    copyAndMove(&database_ocl, &database);

    ocl_matrix output_ocl = database_ocl;

    byte_size = output_ocl.pr * output_ocl.pc * sizeof(real);
    output_ocl.mat = cl::Buffer(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    /** preparing memory for distances between points
     *  from current and previous iteration */

    byte_size = output_ocl.pr * sizeof(real);
    cl::Buffer distances(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    int itersNum = 20;

    for(int i = 0; i < itersNum; ++i)
    {
        std::cout << "calculating meanshift query" << std::endl;

        meanshiftKQueryRBC(database_ocl, rbcS, allPilots,
                           selectedPoints, selectedPointsNum,
                           maxQuerySize);

        std::cout << "calculating meanshift mean" << std::endl;

        meanshiftMeanWrap(database_ocl, selectedPoints, selectedPointsNum,
                          maxQuerySize, output_ocl);

        std::cout << "calculating distances" << std::endl;

        simpleDistanceKernelWrap(database_ocl, output_ocl, distances);

        validate_query_and_mean(database, selectedPoints, selectedPointsNum, allPilots,
                       maxQuerySize, database_ocl, output_ocl, distances);

        /** switch input and output for the next iteration */
        if(i != itersNum - 1)
        {
            std::cout << "switching input-output" << std::endl;

            ocl_matrix tmp = database_ocl;
            database_ocl = output_ocl;
            output_ocl = tmp;
        }
    }


    write_modes(output_ocl, 512, 512);

//    validate_query_and_mean(database, selectedPoints, selectedPointsNum, allPilots,
//                   maxQuerySize, database_ocl, output_ocl, distances);
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
                    int maxQuerySize, ocl_matrix previous_means,
                    ocl_matrix means, cl::Buffer result_distances)
{
    srand(23);

    cl::CommandQueue& queue = OclContextHolder::queue;
    queue.finish();

    std::cout << "validating query..." << std::endl;

    int tests = 0;
    int db_size = database.r;

    int selPointsSize = db_size * maxQuerySize;
    int meansSize = means.pc * means.pr;

    unint* selectedPointsHost = new unint[selPointsSize];
    unint* selectedPointsNumHost = new unint[db_size];
    real* pilots_host = new real[db_size];
    real* distances = new real[db_size];
    real* previous_means_host = new real[meansSize];
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

    std::cout << "downloading means (1)" << std::endl;

    err = queue.enqueueReadBuffer(previous_means.mat, CL_TRUE, 0,
                                  meansSize * sizeof(real),
                                  previous_means_host);
    checkErr(err);

    std::cout << "downloading means (2)" << std::endl;

    err = queue.enqueueReadBuffer(means.mat, CL_TRUE, 0,
                                  meansSize * sizeof(real), means_host);
    checkErr(err);

    std::cout << "downloading distances" << std::endl;

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

        real dist = 0;
        for(int i = 0; i < database.c; ++i)
        {
            real p1 = database.mat[candidate * database.pc + i];
            real p2 = means_host[candidate * database.pc + i];

            dist += (p1 - p2) * (p1 - p2);
        }

        std::cout << "host result distance: " << dist << std::endl;

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

        std::cout << "device result distance: "
                  << result_distances_host[candidate] << std::endl;

        std::cout << std::endl;

        std::cout << std::endl;

        //std::sort(distances, distances + db_size, sort_function);
    }

    int zeros = 0;

    for(int i = 0; i < database.r; ++i)
    {
        if(!selectedPointsHost[i])
        {
            zeros++;
        }
    }

    std::cout << zeros
              << " points has no points to calculate mean" << std::endl;

    for(int i = 0; i < database.pr * database.pc; ++i)
    {
        if(std::isnan(means_host[i]))
        {
            int row = i / database.pc;
            int col = i % database.pc;

            std::cout << "row: " << row << std::endl;
            std::cout << "col: " << col << std::endl;
            std::cout << "num points: "
                      << selectedPointsNumHost[row] << std::endl;
            assert(false);
        }
    }


    double distances_sum = 0.0;
    int zero_distances = 0;
    int nearly_zero_distances = 0;

    for(int i = 0; i < database.r; ++i)
    {
        float dist = result_distances_host[i];

        assert(!std::isnan(dist));

        if(dist == 0.f)
        {
            ++zero_distances;
        }
        else if(dist < 1.0e-10)
        {
            ++nearly_zero_distances;
        }

        distances_sum += dist;
    }

    std::cout << "result distances sum: " << distances_sum << std::endl;
    std::cout << "avg result distance: " << distances_sum / database.r
                 << std::endl;

    std::cout << "zero distances: "
              << (((float)zero_distances) / database.r) * 100
              << "%" << std::endl;
    std::cout << "nearly zero distances: "
              << (((float)nearly_zero_distances) / database.r) * 100
              << "%" << std::endl;

    delete[] selectedPointsHost;
    delete[] selectedPointsNumHost;
    delete[] pilots_host;
    delete[] distances;
    delete[] previous_means_host;
    delete[] means_host;
    delete[] mean;
    delete[] result_distances_host;
}

void write_modes(ocl_matrix modes, int img_width, int img_height)
{
    int modesSize = modes.pc * modes.pr;

    real* modes_host = new real[modesSize];

    cl_int err;

    cl::CommandQueue& queue = OclContextHolder::queue;

    err = queue.enqueueReadBuffer(modes.mat, CL_TRUE, 0,
                                  modesSize * sizeof(real),
                                  modes_host);
    checkErr(err);

    real* min_values = new real[modes.c];
    real* max_values = new real[modes.c];

    std::fill(min_values, min_values + modes.c,
              std::numeric_limits<real>::max());

    std::fill(max_values, max_values + modes.c,
              -std::numeric_limits<real>::max());

    /** find min/max */

    for(int i = 0; i < img_height; ++i)
    {
        int row_idx = i * img_width;

        for(int j = 0; j < img_width; ++j)
        {
            int linear_idx = row_idx + j;

            real* mode_ptr = modes_host + linear_idx * modes.pc;

            for(int k = 0; k < modes.c; ++k)
            {
                min_values[k] = std::min(mode_ptr[k], min_values[k]);
                max_values[k] = std::max(mode_ptr[k], max_values[k]);
            }
        }
    }

    std::cout << "min values:" << std::endl;

    for(int i = 0; i < modes.c; ++i)
    {
        std::cout << min_values[i] << ", ";
    }

    std::cout << std::endl;

    std::cout << "max values:" << std::endl;

    for(int i = 0; i < modes.c; ++i)
    {
        std::cout << max_values[i] << ", ";
    }

    std::cout << std::endl;

    /** normalize */

    for(int i = 0; i < img_height; ++i)
    {
        int row_idx = i * img_width;

        for(int j = 0; j < img_width; ++j)
        {
            int linear_idx = row_idx + j;

            real* mode_ptr = modes_host + linear_idx * modes.pc;

            for(int k = 0; k < modes.c; ++k)
            {
                mode_ptr[k] = 255 * (mode_ptr[k] - min_values[k])
                                             / (max_values[k] - min_values[k]);
            }
        }
    }

    /** writing images */

    for(int m = 0; m < modes.c; ++m)
    {
        QImage img(img_width, img_height, QImage::Format_ARGB32);

        std::set<real> distinct_values;

        for(int i = 0; i < img_height; ++i)
        {
            int row_idx = i * img_width;

            for(int j = 0; j < img_width; ++j)
            {
                int linear_idx = row_idx + j;

                real* mode_ptr = modes_host + linear_idx * modes.pc;

                real val = mode_ptr[m];

                distinct_values.insert(val);

                int value = (int)val;

                img.setPixel(j, i, qRgb(value, value, value));
            }
        }

        std::cout << "distinct values: "
                  << distinct_values.size() << std::endl;

        QString filename = QString("mode_%1.png").arg(m);
        img.save(filename);

        std::map<real, unsigned char> mapping;
        int counter = 0;
        int step = 255/distinct_values.size();

        for(std::set<real>::iterator it = distinct_values.begin();
            it != distinct_values.end(); ++it)
        {
            mapping[*it] = counter * step;
            ++counter;
        }

        QImage img2(img_width, img_height, QImage::Format_ARGB32);

        for(int i = 0; i < img_height; ++i)
        {
            int row_idx = i * img_width;

            for(int j = 0; j < img_width; ++j)
            {
                int linear_idx = row_idx + j;

                real* mode_ptr = modes_host + linear_idx * modes.pc;

                real val = mode_ptr[m];

                int value = mapping[val];

                img2.setPixel(j, i, qRgb(value, value, value));
            }
        }

        QString filename2 = QString("mode_mapped_%1.png").arg(m);
        img2.save(filename2);

    }

    delete[] modes_host;
    delete[] min_values;
    delete[] max_values;
}
