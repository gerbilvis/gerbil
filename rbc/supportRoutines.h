#ifndef SUPPORT_ROUTINES
#define SUPPROT_ROUTINES

#include "defs.h"

void parseInput(int argc, char **argv);
void readData(char *dataFile, matrix x);
void readDataText(char *dataFile, matrix x);
void writeNeighbs(const char *file, const char *filetxt,
                  intMatrix NNs, matrix dNNs);
void evalKNNerror(matrix x, matrix q, intMatrix NNs);

int old_main(int argc, char**argv);

#endif
