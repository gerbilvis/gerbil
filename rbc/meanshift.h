#ifndef MEANSHIFT_H
#define MEANSHIFT_H

#include "rbc_include.h"

/** numReps - number of representatives for rbc or 0 for 5 * sqrt(n)
 *
 */
void meanshift_rbc(matrix database, int img_width, int img_height,
                   unsigned short* final_modes = 0,
                   unsigned int* final_hmodes = 0,
                   int numReps = 0, int pointsPerRepresentative = 16*1024);

void validate_pilots(matrix database, cl::Buffer pilots);
void validate_query_and_mean(matrix database, cl::Buffer selectedPoints,
                    cl::Buffer selectedPointsNum, cl::Buffer pilots,
                    int maxQuerySize, ocl_matrix previous_means,
                    ocl_matrix means, cl::Buffer result_distances);

void validate_distances(matrix database, cl::Buffer result_distances);
void validate_indexes(cl::Buffer oldIndexes, cl::Buffer newIndexes,
                      int old_size, int new_size);

void write_modes(ocl_matrix modes, int img_width, int img_height);
void write_iteration_map(cl::Buffer map, int img_width, int img_height);

#endif
