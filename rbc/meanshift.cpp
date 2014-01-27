#include "meanshift.h"

#include <cmath>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <set>
#include <map>

#include <QImage>
#include <QElapsedTimer>

#include "kernelWrap.h"

void meanshift_rbc(vole::RBCConfig config, matrix database,
                   int img_width, int img_height,
                   unsigned short* final_modes, unsigned int* final_hmodes)
{
    int maxQuerySize = config.maxQuerySize;
    int numReps = config.numReps;
    int pointsPerRepresentative = config.pointsPerRepr;
    int pilotsThreshold = config.pilotsThreshold;

    int database_size = database.r;

    if(numReps == 0)
        numReps = sqrt(database_size) * 5;


    cl::Context& context = OclContextHolder::context;
    cl::CommandQueue& queue = OclContextHolder::queue;

    cl_int err;
    cl::Buffer repsPilots(context, CL_MEM_READ_WRITE,
                          PAD(numReps) * sizeof(real), 0, &err);

    cl::Buffer repsWeights(context, CL_MEM_READ_WRITE,
                           PAD(numReps) * sizeof(double), 0, &err);

    checkErr(err);

//    cl::Buffer newRepsPilots(context, CL_MEM_READ_WRITE,
//                             PAD(numReps) * sizeof(real), 0, &err);
//    checkErr(err);

    ocl_rbcStruct rbcS;

    std::cout << "building rbc" << std::endl;

    /** calculating RBC and pilots for representatives */
    buildRBC(database, &rbcS, numReps, pointsPerRepresentative,
             repsPilots, pilotsThreshold);

    meanshiftWeightsWrap(repsPilots, repsWeights, numReps, database.c);


    cl::Buffer allPilots(context, CL_MEM_READ_WRITE,
                          PAD(database.pr) * sizeof(real), 0, &err);
    checkErr(err);

    cl::Buffer allWeights(context, CL_MEM_READ_WRITE,
                          PAD(database.pr) * sizeof(double), 0, &err);
    checkErr(err);

    std::cout << "computing pilot" << std::endl;

    /** calculating pilot for every point */
    computePilotsAndWeights(database, repsPilots, repsWeights,
                            rbcS, allPilots, allWeights);

    validate_pilots(database, allPilots);


    /** preparing memory for meanshift query */

    unint partsNum = 1;
    unint pointsPerPart = database.pr / partsNum;

    assert(database.pr % partsNum == 0);

    int byte_size = sizeof(unint) * pointsPerPart * maxQuerySize;

    cl::Buffer selectedPoints(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    byte_size = sizeof(real) * pointsPerPart * maxQuerySize;

    cl::Buffer selectedDistances(context, CL_MEM_READ_WRITE,
                                 byte_size, 0, &err);
    checkErr(err);

    byte_size = sizeof(unint) * pointsPerPart;

    cl::Buffer selectedPointsNum(context, CL_MEM_READ_WRITE,
                                 byte_size, 0, &err);
    checkErr(err);

    byte_size = sizeof(real) * pointsPerPart;

    cl::Buffer hmodes(context, CL_MEM_READ_WRITE, byte_size, 0, &err);

    checkErr(err);

    byte_size = sizeof(real) * database.pr;

    cl::Buffer hmodes_total(context, CL_MEM_READ_WRITE, byte_size, 0, &err);

    checkErr(err);

    /** preparing memory for input and output means */

    ocl_matrix database_ocl; /** original data for calculating mean */
    copyAndMove(&database_ocl, &database);

    ocl_matrix input_ocl; /** input for query */
    copyAndMove(&input_ocl, &database);
    
//    ocl_matrix database_ocl_2;
//    copyAndMove(&database_ocl_2, &database);

    ocl_matrix output_ocl = database_ocl; /** result mean */
    ocl_matrix next_input_ocl = database_ocl; /** next input */
    ocl_matrix final_modes_ocl = database_ocl; /** final modes */

    byte_size = output_ocl.pr * output_ocl.pc * sizeof(real);
    output_ocl.mat = cl::Buffer(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    next_input_ocl.mat = cl::Buffer(context, CL_MEM_READ_WRITE,
                                    byte_size, 0, &err);
    checkErr(err);

    final_modes_ocl.mat = cl::Buffer(context, CL_MEM_READ_WRITE,
                                     byte_size, 0, &err);

    clearRealKernelWrap(final_modes_ocl.mat, database.pr * database.pc);

    checkErr(err);

    byte_size = output_ocl.pr * sizeof(unint);

    cl::Buffer curr_indexes(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    cl::Buffer new_indexes(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    initIndexesKernelWrap(curr_indexes, output_ocl.pr);


    /** preparing memory for distances between points
     *  from current and previous iteration */

    byte_size = output_ocl.pr * sizeof(real);
    cl::Buffer distances(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    byte_size = output_ocl.pr * sizeof(unint);
    cl::Buffer iteration_map(context, CL_MEM_READ_WRITE, byte_size, 0, &err);
    checkErr(err);

    clearIntKernelWrap(iteration_map, output_ocl.pr);


    int itersNum = 250;

    int data_size = database.pr;

    QElapsedTimer timer;
    timer.start();

    for(int i = 0; i < itersNum; ++i)
    {
        std::cout << "iteration: " << i << std::endl;

        for(int j = 0; j < partsNum; ++j)
        {
            std::cout << "part: " << j << std::endl;

            if(j * pointsPerPart >= input_ocl.pr)
            {
                break;
            }

            int todo = std::min(pointsPerPart, (input_ocl.pr - j * pointsPerPart));

            todo = ((todo + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;


            cl_buffer_region region;
            region.origin = j * pointsPerPart * input_ocl.pc * sizeof(real);
//            region.size = std::min(pointsPerPart * input_ocl.pc * sizeof(real),
//                                   (input_ocl.pr - j * pointsPerPart) * input_ocl.pc * sizeof(real));

            region.size = todo * input_ocl.pc * sizeof(real);


            cl::Buffer input_subBuffer = input_ocl.mat.createSubBuffer(
                        CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                        &region, &err);

            checkErr(err);

            cl::Buffer output_subBuffer = output_ocl.mat.createSubBuffer(
                        CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                        &region, &err);

            checkErr(err);

            ocl_matrix sub_input_matrix = input_ocl;
            sub_input_matrix.mat = input_subBuffer;
            sub_input_matrix.r = todo;
            sub_input_matrix.pr = todo;

            ocl_matrix sub_output_matrix = output_ocl;
            sub_output_matrix.mat = output_subBuffer;
            sub_output_matrix.r = todo;
            sub_output_matrix.pr = todo;

            std::cout << "calculating meanshift query" << std::endl;

            meanshiftKQueryRBC(sub_input_matrix, rbcS, allPilots, allWeights,
                               selectedPoints, selectedDistances,
                               selectedPointsNum, hmodes, maxQuerySize);

         //   queue.finish();

            std::cout << "calculating meanshift mean" << std::endl;

            meanshiftMeanWrap(database_ocl, selectedPoints,
                              selectedDistances, selectedPointsNum, allPilots,
                              allWeights, maxQuerySize, sub_output_matrix);

        //    queue.finish();
//            meanshiftMeanWrap(database_ocl, selectedPoints,
//                              selectedDistances, selectedPointsNum, allPilots,
//                              allWeights, maxQuerySize, sub_output_matrix);
        }

        unint result_size = 0;

        meanshiftPackKernelWrap(input_ocl, output_ocl, next_input_ocl,
                                final_modes_ocl, curr_indexes, new_indexes,
                                data_size, result_size, iteration_map,
                                hmodes, hmodes_total, i + 1);

        //validate_indexes(curr_indexes, new_indexes, data_size, result_size);

        std::cout << "result size: " << result_size << std::endl;


//        std::cout << "calculating distances" << std::endl;
//        simpleDistanceKernelWrap(prev_input_ocl, output_ocl, distances);
  //      validate_distances(database, distances);

      //  validate_query_and_mean(database, selectedPoints, selectedPointsNum, allPilots,
        //               maxQuerySize, database_ocl, output_ocl, distances);

       // write_modes(output_ocl, img_width, img_height);

      //  write_modes(final_modes_ocl, img_width, img_height);
        write_iteration_map(iteration_map, img_width, img_height);

        /** switch input and output for the next iteration */
        if(i != itersNum - 1)
        {
            std::cout << "switching input-output" << std::endl;

            std::swap(input_ocl, next_input_ocl);
            std::swap(curr_indexes, new_indexes);

            unint result_size_rounded = ((result_size + BLOCK_SIZE - 1)
                                        / BLOCK_SIZE) * BLOCK_SIZE;

            input_ocl.pr = result_size_rounded;
            input_ocl.r = result_size;

            output_ocl.pr = result_size_rounded;
            output_ocl.r = result_size;

            data_size = result_size;
        }

        qint64 time = timer.elapsed();

        std::cout << "elapsed time: " << time/1000.f << " [s]" << std::endl;
    }

    /** reading final distances */

    byte_size = database.pr * database.pc * sizeof(real);

    real* final_modes_host = new real[database.pr * database.pc];

    err = queue.enqueueReadBuffer(final_modes_ocl.mat, CL_TRUE, 0, byte_size,
                                  final_modes_host);
    checkErr(err);

    for(int i = 0; i < database.r; ++i)
    {
        real* row_src = final_modes_host + database.pc * i;
        unsigned short* row_dst = final_modes + database.c * i;

        for(int j = 0; j < database.c; ++j)
        {
            row_dst[j] = row_src[j];
        }
    }

    delete[] final_modes_host;

    /** reading final hmodes */

    byte_size = database.pr * sizeof(real);

    real* hmodes_host = new real[database.pr];

    err = queue.enqueueReadBuffer(hmodes_total, CL_TRUE, 0, byte_size, hmodes_host);
    checkErr(err);

    for(int i = 0; i < database.r; ++ i)
    {
        final_hmodes[i] = hmodes_host[i];
    }

    delete[] hmodes_host;




    //write_modes(output_ocl, img_width, img_height);

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

    int tests = 5;
    int db_size = database.r;

    real* pilots_host = new real[db_size];
    real* distances = new real[db_size];

    cl_int err;

    err = queue.enqueueReadBuffer(pilots, CL_TRUE, 0,
                                  db_size * sizeof(real), pilots_host);

    real p_max = 0.f;
    real p_min = std::numeric_limits<real>::max();

    for(int i = 0; i < db_size; ++i)
    {
        real p = pilots_host[i];

        assert(p > 0.f);
        assert(((int)p) > 0);

        p_max = std::max(p_max, p);
        p_min = std::min(p_min, p);
    }

    std::cout << "pilot max: " << p_max << std::endl;
    std::cout << "pilot min: " << p_min << std::endl;

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

//                distance += (a - b) * (a - b);
                distance += std::abs(a - b);
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

void validate_distances(matrix database, cl::Buffer result_distances)
{
    cl_int err;

    cl::CommandQueue& queue = OclContextHolder::queue;
    queue.finish();

    std::cout << "validating distances..." << std::endl;

    int db_size = database.r;

    real* result_distances_host = new real[db_size];


    std::cout << "downloading distances" << std::endl;

    err = queue.enqueueReadBuffer(result_distances, CL_TRUE, 0,
                                  database.r * sizeof(real),
                                  result_distances_host);
    checkErr(err);

    double distances_sum = 0.0;
    int zero_distances = 0;
    int nearly_zero_distances = 0;

    static std::set<int> all_zeros_so_far; // temporary

    for(int i = 0; i < database.r; ++i)
    {
        float dist = result_distances_host[i];

        assert(!std::isnan(dist));

        if(dist == 0.f)
        {
            ++zero_distances;
            all_zeros_so_far.insert(i);
        }
        else if(dist < 1.0e-10)
        {
            ++nearly_zero_distances;
        }

        distances_sum += dist;
    }

    std::cout << "all_zeros_so_far: "
              << (((float)all_zeros_so_far.size()) / database.r) * 100
              << "%" << std::endl;

    std::cout << "result distances sum: " << distances_sum << std::endl;
    std::cout << "avg result distance: " << distances_sum / database.r
                 << std::endl;

    std::cout << "zero distances: "
              << (((float)zero_distances) / database.r) * 100
              << "%" << std::endl;
    std::cout << "nearly zero distances: "
              << (((float)nearly_zero_distances) / database.r) * 100
              << "%" << std::endl;

    delete[] result_distances_host;

}

void validate_indexes(cl::Buffer oldIndexes, cl::Buffer newIndexes,
                      int old_size, int new_size)
{
    unint* oldIndexesHost = new unint[old_size];
    unint* newIndexesHost = new unint[new_size];

    cl::CommandQueue& queue = OclContextHolder::queue;

    cl_int err;
    int byte_size = old_size * sizeof(unint);

    err = queue.enqueueReadBuffer(oldIndexes, CL_TRUE, 0, byte_size,
                                  oldIndexesHost, 0, 0);
    checkErr(err);

    byte_size = new_size * sizeof(unint);
    err = queue.enqueueReadBuffer(newIndexes, CL_TRUE, 0, byte_size,
                                  newIndexesHost, 0, 0);
    checkErr(err);

    int size_1 = std::min(100, std::min(old_size, new_size));

    for(int i = 0; i < size_1; ++i)
    {
        std::cout << "idx old: " << oldIndexesHost[i]
                     << ", new: " << newIndexesHost[i] << std::endl;
    }


    delete[] oldIndexesHost;
    delete[] newIndexesHost;
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

    std::cout << "first dim only!" << std::endl;

//    for(int m = 0; m < modes.c; ++m)
    for(int m = 0; m < 1; ++m)
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

        QString filename = QString("mode_%1.bmp").arg(m);
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

        QString filename2 = QString("mode_mapped_%1.bmp").arg(m);
        img2.save(filename2);

    }

    delete[] modes_host;
    delete[] min_values;
    delete[] max_values;
}

void write_iteration_map(cl::Buffer map, int img_width, int img_height)
{
    int img_size = img_width * img_height;

    unint* map_host = new unint[img_size];

    cl_int err;

    cl::CommandQueue& queue = OclContextHolder::queue;

    err = queue.enqueueReadBuffer(map, CL_TRUE, 0,
                                  img_size * sizeof(unint),
                                  map_host);
    checkErr(err);


    QImage img(img_width, img_height, QImage::Format_ARGB32);

    for(int i = 0; i < img_height; ++i)
    {
        int row_idx = i * img_width;

        for(int j = 0; j < img_width; ++j)
        {
            unint val = map_host[row_idx + j];

//            val *= 2;

            if(val == 0)
                img.setPixel(j, i, qRgb(255, 0, 0));
            else
                img.setPixel(j, i, qRgb(val, val, val));
        }
    }

    QString filename = "iteration_map.bmp";
    img.save(filename);
}

