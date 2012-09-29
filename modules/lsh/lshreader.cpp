#include "lshreader.h"
#include <cmath>
#include <algorithm>

LSHReader::LSHReader(const LSH& master)
	: lsh(master),
	  /// metadata array is initialized to 0, so first query gets tag 1
	  queryTag(1)
{
	/// initialize metadata array
	queryTags.assign(lsh.npoints, 0);

	/// initialize result state
	result.valid = false;

	/// initialize shortcut table
	/// loosely based on original implementation, but should actually use nsel instead of npoints
	shortcutTableSize = lsh.GetPrime(lsh.npoints * 4);
	shortcutTable.assign(shortcutTableSize, vector<ShortcutEntry>());
}

/// perform query on given coordinates
/// (expects array with dims elements)
const void* LSHReader::query(const LSH::data_t *point, const void *endResult)
{
	vector<vector<bool> > boolVecs(lsh.L);
	vector<int> primaryHashes(lsh.L);
	vector<int> secondaryHashes(lsh.L);

	/// determine boolean vector for all partitions
	for (int l = 0; l < lsh.L; l++) {
		boolVecs[l] = lsh.getBoolVec(point, lsh.partitions[l]);
		std::pair<int, int> hashes = lsh.hashFunc(boolVecs[l], l);
		primaryHashes[l] = hashes.first;
		secondaryHashes[l] = hashes.second;
	}

	/// result caching for early trajectory termination
	if (endResult != NULL) {
		/// compute two hashes of all primary hashes
		int shortcutHash1 = 0;
		int shortcutHash2 = 0;
		for (int l = 0; l < lsh.L; l++) {
			shortcutHash1 += primaryHashes[l] * lsh.hashCoeffs[l];
		}
		for (int l = 0; l < lsh.L / 2; l++) {
			shortcutHash2 += primaryHashes[l + lsh.L/2] * lsh.hashCoeffs[l];
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
	result.numByPartition.reserve(lsh.L);

	/// for each partition...
	for (int l = 0; l < lsh.L; l++) {
		int primaryHash = abs(primaryHashes[l]) % lsh.nbuckets;
		int secondaryHash = secondaryHashes[l];

#ifdef DEBUG_VERBOSE
		fprintf(stderr, "LSH::query() l=%d, hash=%d, hash2=%d\n", l, primaryHash, hashes.second);
#endif // DEBUG_VERBOSE

		/// inspect all entries in bucket
		const vector<LSH::Entry> &bucket = lsh.tables[l][primaryHash];
		vector<LSH::Entry>::const_iterator bucketIt;

		for (bucketIt = bucket.begin(); bucketIt != bucket.end(); ++bucketIt) {
			const LSH::Entry &entry = *bucketIt;
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

void LSHReader::query(unsigned int point)
{
	query(&lsh.data[point * lsh.dims], NULL);
}

const std::vector<unsigned int>& LSHReader::getResult() const
{
	return result.points;
}

const std::vector<int>& LSHReader::getNumByPartition() const
{
	return result.numByPartition;
}
