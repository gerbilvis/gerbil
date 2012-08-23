#ifndef SORTING_H
#define SORTING_H

namespace powerwaterseg {

void TriRapideStochastique(int * A, int *I, int p, int r);
void TriRapideStochastique_dec(unsigned int * A, int *I, int p, int r);
void TriRapideStochastique_inc(unsigned int * A, int *I, int p, int r);

struct Score
{
	float value;
	int index;
	bool operator<(const Score& score) const;
};

void sortRange(float *F, int *Es, int M, bool reverse = false);

}

#endif
