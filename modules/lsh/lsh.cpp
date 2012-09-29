/*
	Copyright(c) 2011 Daniel Danner,
	Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/
#include <cstdlib>
#include <string.h>
#include <algorithm>
#include <assert.h>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include "lsh.h"

//#define DEBUG
//#define DEBUG_VERBOSE
//#define VERBOSE_RANDOM

LSH::LSH(data_t *data, int npoints, int dims, int K, int L, bool dataDrivenPartitions, const vector<unsigned int> &subSet) :
		data(data),
		npoints(npoints),
		dims(dims),
		K(K),
		L(L),
		dataDrivenPartitions(dataDrivenPartitions),
		subSet(subSet),

		/// variety of hashes should depend solely on dims and K
		/// (original implementation used 3 * npoints * L / 256, but has fixed bucket lengths)
		nbuckets(GetPrime(dims * K))
{
#ifdef DEBUG
	fprintf(stderr, "nbuckets=%d, bucketSize=%d\n", nbuckets, bucketSize);
#endif // DEBUG

	/// sanity checks
	assert(K > 0);
	assert(L > 0);

	/// initialize L hash tables, with nbuckets each
	tables.assign(L, Htable(nbuckets));

	/// initialize hash coefficients
	for (int i = 0; i < max(K, L); i++)
		hashCoeffs.push_back(rand());

	makeCuts();

	fillTable();
}

int LSH::GetPrime(int minp) {
	int i, j;
	for (i = minp % 2 == 0 ? minp + 1 : minp;; i += 2) {
		int sqt = (int)std::sqrt((float)i);
		if (i % 2 == 0)
			continue;
		for (j = 3; j < sqt; j += 2) {
			if (i % j == 0)
				break;
		}
		if (j >= sqt)
			return i;
	}
	return -1;
};

void LSH::makeCuts()
{
	partitions.assign(L, vector<cut_t>(K));

	/// for each partition...
	for (int l = 0; l < L; l++) {
		partition_t &cuts = partitions[l];
		int ncuts = 0;

		/// every dimension gets K/dims cuts (that's the average)
		int cutsPerDim = K / dims;
		for (int d = 0; d < dims; d++) {
			for (int i = 0; i < cutsPerDim; i++) {
				cuts[ncuts++] = randomCut(d);
			}
		}

		/// randomly choose dimensions for the remainder of K/d,
		std::vector<char> used(dims, 0);
		while (ncuts < K) {
			int d;
			do {
				d = random(dims - 1);
			} while(used[d]);
			used[d] = 1;

			cuts[ncuts++] = randomCut(d);
		}
#ifdef VERBOSE_RANDOM
		fprintf(stderr, "LSH: cuts for l=%d: ", l);
		for (int i = 0; i < ncuts; i++)
			fprintf(stderr, "(%d,%d) ", cuts[i].dim, cuts[i].pos);
		fprintf(stderr, "\n");
#endif // VERBOSE_RANDOM
	}
}

int LSH::random(int max) const {
	return min((int) ((double) rand() / RAND_MAX * (max)), max);
}

LSH::cut_t LSH::randomCut(int dim) const
{
	cut_t ret;

	if (dataDrivenPartitions) {
		int p;
		if (subSet.empty()) {
			p = random(npoints - 1);
		} else {
			p = random(subSet.size() - 1);
			p = subSet[p];
		}
#ifdef VERBOSE_RANDOM
		fprintf(stderr, "LSH: rand: -> %d\n", p);
#endif // VERBOSE_RANDOM
		ret.dim = dim;
		ret.pos = data[p * dims + dim];
	} else {
		ret.dim = dim;
		/// assuming data_t is unsigned, this should yield the maximum value
		double maxval = (data_t) -1;
		ret.pos = min((int) ((double) rand() / RAND_MAX * (maxval)), (int) maxval);
	}

	return ret;
}

void LSH::fillTable()
{
	/// for each partition
	for (int l = 0; l < L; l++) {
		Htable &table = tables[l];
		/// for each point...
		int n = subSet.empty() ? npoints : subSet.size();
		for (int p_i = 0; p_i < n; p_i++) {
			int p = subSet.empty() ? p_i : subSet[p_i];
			std::vector<bool> boolVec = getBoolVec(p, partitions[l]);
			pair<int, int> hashes = hashFunc(boolVec, l);
			int primaryHash = abs(hashes.first) % nbuckets;

#ifdef DEBUG_VERBOSE
			fprintf(stderr, "LSH::hashFunc point=%d, l=%d -> hashes.second=%d\n", p, l, hashes.second);
			fprintf(stderr, "LSH::fillTable() Putting point %i into bucket %i (hashes.second=%d)\n", p, primaryHash, hashes.second);
#endif // DEBUG_VERBOSE

			/// insert new entry
			table[primaryHash].push_back(Entry(p, hashes.second));;
		}
	}
}

std::vector<bool> LSH::getBoolVec(const data_t *point, const partition_t &part) const
{
	std::vector<bool> ret(K);
	for (int k = 0; k < K; k++)
		ret[k] = point[part[k].dim] >= part[k].pos;
	return ret;
}

std::vector<bool> LSH::getBoolVec(unsigned int point, const partition_t &part) const
{
	return getBoolVec(&data[point * dims], part);
}

pair<int, int> LSH::hashFunc(const std::vector<bool>& boolVec, int partIdx) const
{
	int primary = partIdx;
	int secondary = partIdx;
	for (int i = 1; i < K; i++) {
		if (boolVec[i]) {
			primary += hashCoeffs[i];
			if (i != 0) /// secondary skips first bool
				secondary += hashCoeffs[i - 1];
		}
	}

	return make_pair(primary, secondary);
}

vector< vector<unsigned int> > LSH::getLargestBuckets(double p) const
{
	vector< vector<unsigned int> > ret;
	unsigned int minCount = (int)p * npoints;
	for (int l = 0; l < L; ++l) {
		const Htable &table = tables[l];
		for (int k = 0; k < nbuckets; ++k) {
			const vector<Entry> &bucket = table[k];
			if (bucket.size() > minCount) {
				ret.push_back(vector<unsigned int>());
				vector<Entry>::const_iterator it;
				for (it = bucket.begin(); it != bucket.end(); ++it) {
					ret.back().push_back(it->point);
				}
			}
		}
	}
	return ret;
}
