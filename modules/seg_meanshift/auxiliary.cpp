#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <ctime>
#include "auxiliary.h" // for fun

using namespace std;

void bgLog(const char *varStr, ...) {
	//obtain argument list using ANSI standard...
	va_list argList;
	va_start(argList, varStr);

	//print the output string to stderr using
	vfprintf(stderr, varStr, argList);
	va_end(argList);
	fflush(stderr);
}

time_t timestart, timeend;

void timer_start() {
	timestart = clock();
}

double timer_stop() {
	timeend = clock();
	unsigned long seconds, milliseconds;
	seconds      = (timeend - timestart) / CLOCKS_PER_SEC;
	milliseconds =
		((1000 * (timeend - timestart)) / CLOCKS_PER_SEC) - 1000 * seconds;
	return seconds + milliseconds / 1000.0;
}

double timer_elapsed(int prnt) {
	timeend = clock();
	unsigned long hours = 0, minutes = 0, seconds = 0, milliseconds = 0;
	seconds      = (timeend - timestart) / CLOCKS_PER_SEC;
	milliseconds =
		((1000 * (timeend - timestart)) / CLOCKS_PER_SEC) - 1000 * seconds;
	minutes = seconds / 60;
	if (minutes == 0) {
		if (prnt) {
			printf("elapsed %lu.%03lu seconds. \n", seconds, milliseconds);
		}
	} else {
		hours   = minutes / 60;
		seconds = seconds - minutes * 60;
		if (hours == 0) {
			if (prnt)
				printf("elapsed %lum%lus%lums\n", minutes,
					   seconds,
					   milliseconds);
		} else {
			minutes = minutes - hours * 60;
			if (prnt)
				printf("elapsed %luh%lum%lus%lums\n", hours,
					   minutes, seconds,
					   milliseconds);
		}
	}
	return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0;
}
