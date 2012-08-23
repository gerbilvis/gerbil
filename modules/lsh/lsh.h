/*	
	Copyright(c) 2011 Daniel Danner,
	Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/
#ifndef LSH_H
#define LSH_H

#include <vector>
#include <map>

/// fixed size for partition data type
#define K_MAX 70

using std::vector;
using std::min;
using std::max;
using std::pair;
using std::make_pair;

class LSH
{
	typedef unsigned short data_t;

	typedef struct Entry {
		/// data point index
		unsigned int point;
		/// secondary hash for more accurate results
		int secondaryHash;

		Entry(unsigned int point, int secondaryHash)
			: point(point), secondaryHash(secondaryHash) {}
	} Entry;

	typedef struct cut {
		int dim;
		data_t pos;
	} cut_t;

	typedef vector<cut_t> partition_t;

	typedef vector< vector<Entry> > Htable;

	typedef struct ShortcutEntry {
		/// secondary hash for more accurate results
		int secondaryHash;

		/// result pointer
		const void *p;
	} ShortcutEntry;

public:
	LSH(data_t *data, int npoints, int dims, int K, int L, bool dataDrivenPartitions = true, const vector<unsigned int> &subSet = vector<unsigned int>());

	~LSH() {};

	/// Perform query on given coordinates.
	/// If endResult is not NULL, the queried point will be associated
	/// with its value in an extra hash table. Any further queries to
	/// the same intersection (i.e. same boolean vectors) will return
	/// the pointer's value instead of NULL. The actual result will be empty.
	/// This can serve as shortcut to the calling algorithm's final result.
	const void *query(const data_t *point, const void *endResult = NULL);

	/// perform query on existing data point
	void query(unsigned int point);

	/// return result for previous query
	const vector<unsigned int>& getResult() const;

	/// maps partition number to result size for previous query
	const vector<int>& getNumByPartition() const;

	/// return contents of buckets containing more than p*npoints items
	vector< vector<unsigned int> > getLargestBuckets(double p) const;
private:
	/// members:

	/// interleaved data points
	const data_t *data;

	/// number of data points
	const int npoints;

	/// number of dimensions
	const int dims;

	/// query tag for each data point
	vector<unsigned int> queryTags;

	/// number of cuts per partition
	const int K;

	/// number of partitions
	const int L;

	/// use data-driven partitions
	const bool dataDrivenPartitions;

	/// subset of indices to use (optional)
	const vector<unsigned int> &subSet;

	/// number of hash table buckets
	const int nbuckets;

	/// hash tables
	vector<Htable> tables;

	vector< vector<cut_t> > partitions;
	vector<int> hashCoeffs;

	/// shortcut hash table (maps queried points to a given pointer)
	int shortcutTableSize;
	vector< vector<ShortcutEntry> > shortcutTable;

	/// current query tag,  used to prevent duplicates in result
	unsigned int queryTag;

	/// return random number in [0;size)
	int random(int max) const;

	/// return random cut chosen from data set
	cut_t randomCut(int dim) const;

	/// contains the latest query's boolean vectors and yielded result
	/// (used as cache for similar queries)
	struct result {
		bool valid;
		vector<int> primaryHashes;
		vector<unsigned int> points;
		vector<int> numByPartition;
	} result;

	/// methods:

	void makeCuts();

	void fillTable();

	/// determine boolean vector for given coordinates in a certain partition
	std::vector<bool> getBoolVec(const data_t *point, const partition_t &part) const;

	/// determine boolean vector for an existing point in a certain partition
	std::vector<bool> getBoolVec(unsigned int point, const partition_t &part) const;

	/// calculate primary and secondary hash
	std::pair<int, int> hashFunc(const std::vector<bool>& boolVec, int partIdx) const;

	/// return the smallest prime number greater than a given value
	static int GetPrime(int minp);

};

#endif // LSH_H
