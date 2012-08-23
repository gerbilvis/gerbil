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
		nbuckets(GetPrime(dims * K)),

		/// metadata array is initialized to 0, so first query gets tag 1
		queryTag(1)
{
#ifdef DEBUG
	fprintf(stderr, "nbuckets=%d, bucketSize=%d\n", nbuckets, bucketSize);
#endif // DEBUG

	/// sanity checks
	assert(K > 0);
	assert(L > 0);

	/// initialize metadata array
	queryTags.assign(npoints, 0);

	/// initialize result state
	result.valid = false;

	/// initialize shortcut table
	/// loosely based on original implementation, but should actually use nsel instead of npoints
	shortcutTableSize = GetPrime(npoints * 4);
	shortcutTable.assign(shortcutTableSize, vector<ShortcutEntry>());

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

/// perform query on given coordinates
/// (expects array with dims elements)
const void* LSH::query(const data_t *point, const void *endResult)
{
	vector<vector<bool> > boolVecs(L);
	vector<int> primaryHashes(L);
	vector<int> secondaryHashes(L);

	/// determine boolean vector for all partitions
	for (int l = 0; l < L; l++) {
		boolVecs[l] = getBoolVec(point, partitions[l]);
		std::pair<int, int> hashes = hashFunc(boolVecs[l], l);
		primaryHashes[l] = hashes.first;
		secondaryHashes[l] = hashes.second;
	}

	/// result caching for early trajectory termination
	if (endResult != NULL) {
		/// compute two hashes of all primary hashes
		int shortcutHash1 = 0;
		int shortcutHash2 = 0;
		for (int l = 0; l < L; l++) {
			shortcutHash1 += primaryHashes[l] * hashCoeffs[l];
		}
		for (int l = 0; l < L / 2; l++) {
			shortcutHash2 += primaryHashes[l + L/2] * hashCoeffs[l];
		}

		shortcutHash1 = abs(shortcutHash1) % shortcutTableSize;

		/// find match in result cache
		vector<ShortcutEntry> &bucket = shortcutTable.at(shortcutHash1);
		vector<ShortcutEntry>::const_iterator bucketIt = bucket.begin();

		const void *match = NULL;
		for (; bucketIt != bucket.end(); ++bucketIt) {
			const ShortcutEntry &entry = *bucketIt;
			if (entry.secondaryHash == shortcutHash2) {
				match = entry.p;
				break;
			}
		}

		if (match != NULL) {
			return match;
		} else {
			/// no match, insert into result cache
			ShortcutEntry newentry;
			newentry.p = endResult;
			newentry.secondaryHash = shortcutHash2;
			shortcutTable.at(shortcutHash1).push_back(newentry);
		}
	}

	/// compare with vectors from previous query
	if (result.valid && primaryHashes == result.primaryHashes) {
#ifdef DEBUG_VERBOSE
		fprintf(stderr, "LSH::query() cache hit! (%d points)\n", (int) result.points.size());
#endif // DEBUG_VERBOSE
		/// cache hit, keep result unchanged
		return NULL;
	}

	/// mark result valid
	result.valid = true;
	result.primaryHashes = primaryHashes;

	/// clear and prepare result vectors
	result.points.clear();
	result.numByPartition.clear();
	result.numByPartition.reserve(L);

	/// for each partition...
	for (int l = 0; l < L; l++) {
		int primaryHash = abs(primaryHashes[l]) % nbuckets;
		int secondaryHash = secondaryHashes[l];

#ifdef DEBUG_VERBOSE
		fprintf(stderr, "LSH::query() l=%d, hash=%d, hash2=%d\n", l, primaryHash, hashes.second);
#endif // DEBUG_VERBOSE

		/// inspect all entries in bucket
		vector<Entry> &bucket = tables[l][primaryHash];
		vector<Entry>::iterator bucketIt;

		for (bucketIt = bucket.begin(); bucketIt != bucket.end(); ++bucketIt) {
			Entry &entry = *bucketIt;
			int p = entry.point;
			if (queryTags[p] == queryTag)
				continue; /// already in result
			if (entry.secondaryHash != secondaryHash)
				continue; /// no match

			/// mark point
			queryTags[p] = queryTag;

			/// add to result
#ifdef DEBUG_VERBOSE
			std::cerr << "LSH::query() push_back: " << p << std::endl;
#endif // DEBUG_VERBOSE
			result.points.push_back(p);
		}

		result.numByPartition.push_back(result.points.size());
	}

#ifdef DEBUG_VERBOSE
	std::cerr << "LSH::query() returned " << result.points.size() << " points"  << std::endl;
#endif // DEBUG_VERBOSE

	queryTag++;

	return NULL;
}

void LSH::query(unsigned int point)
{
	query(&data[point * dims], NULL);
}

const std::vector<unsigned int>& LSH::getResult() const
{
	return result.points;
}

const std::vector<int>& LSH::getNumByPartition() const
{
	return result.numByPartition;
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
