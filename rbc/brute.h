/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef BRUTE_H
#define BRUTE_H

#include "defs.h"

void bruteRangeCount(matrix,matrix,real*,unint*);

//NEVER USED
//void bruteSearch(matrix,matrix,unint*);


void bruteCPU(matrix,matrix,unint*);
void bruteK(matrix,matrix,intMatrix,matrix);
//void bruteKCPU(matrix,matrix,intMatrix);
#endif
