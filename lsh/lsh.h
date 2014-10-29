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

class LSHReader;

class LSH
{
	friend class LSHReader;

	typedef unsigned short data_t;

	struct Entry {
		Entry(unsigned int point, int secondaryHash)
			: point(point), secondaryHash(secondaryHash) {}

		/// data point index
		unsigned int point;
		/// secondary hash for more accurate results
		int secondaryHash;
	};

	struct cut_t {
		int dim;
		data_t pos;
	};

	typedef vector<cut_t> partition_t;

	typedef vector< vector<Entry> > Htable;

public:
	LSH(const vector<vector<data_t> > &data, int K, int L,
		bool dataDrivenPartitions = true,
		const vector<unsigned int> &subSet = vector<unsigned int>());

	~LSH() {}

	/// return contents of buckets containing more than p*npoints items
	vector< vector<unsigned int> > getLargestBuckets(double p) const;

private:
	/// members:

	/// interleaved data points
	const vector<vector<data_t> > &data;

	/// number of dimensions
	const int dims;

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

	/// return random number in [0;size)
	int random(int max) const;

	/// return random cut chosen from data set
	cut_t randomCut(int dim) const;

	/// methods:

	void makeCuts();

	void fillTable();

	/// determine boolean vector for given coordinates in a certain partition
	std::vector<bool> getBoolVec(const std::vector<data_t> &point,
								 const partition_t &part) const;

	/// determine boolean vector for an existing point in a certain partition
	std::vector<bool> getBoolVec(unsigned int point, const partition_t &part) const;

	/// calculate primary and secondary hash
	std::pair<int, int> hashFunc(const std::vector<bool>& boolVec, int partIdx) const;

	/// return the smallest prime number greater than a given value
	static int GetPrime(int minp);

};

#endif // LSH_H
