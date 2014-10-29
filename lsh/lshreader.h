#ifndef LSHREADER_H
#define LSHREADER_H

#include "lsh.h"

class LSHReader
{
	struct ShortcutEntry {
		/// secondary hash for more accurate results
		int secondaryHash;

		/// result pointer
		const void *p;
	};

public:
	LSHReader(const LSH& master);

	/// Perform query on given coordinates.
	/// If endResult is not NULL, the queried point will be associated
	/// with its value in an extra hash table. Any further queries to
	/// the same intersection (i.e. same boolean vectors) will return
	/// the pointer's value instead of NULL. The actual result will be empty.
	/// This can serve as shortcut to the calling algorithm's final result.
	const void *query(const vector<LSH::data_t> &point,
					  const void *endResult = 0);

	/// perform query on existing data point
	void query(unsigned int point);

	/// return result for previous query
	const vector<unsigned int>& getResult() const;

	/// maps partition number to result size for previous query
	const vector<int>& getNumByPartition() const;

	const LSH& lsh;

private:

	/// contains the latest query's boolean vectors and yielded result
	/// (used as cache for similar queries)
	struct result {
		bool valid;
		vector<int> primaryHashes;
		vector<unsigned int> points;
		vector<int> numByPartition;
	} result;

	/// shortcut hash table (maps queried points to a given pointer)
	int shortcutTableSize;
	vector< vector<ShortcutEntry> > shortcutTable;

	/// query tag for each data point
	vector<unsigned int> queryTags;

	/// current query tag, used to prevent duplicates in result
	unsigned int queryTag;
};

#endif // LSHREADER_H
