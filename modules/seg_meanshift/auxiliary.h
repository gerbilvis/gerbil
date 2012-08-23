#ifndef AUXILIARY_H
#define AUXILIARY_H

#ifndef __max
#define __max(a, b)    (a > b ? a : b)
#define __min(a, b)    (a < b ? a : b)
#endif

// aux functions
void bgLog(const char *varStr, ...);

void timer_start();
double timer_stop();
double timer_elapsed(int prnt);

#endif
