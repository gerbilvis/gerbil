#ifndef DISTVIEWCOMPUTE_H
#define DISTVIEWCOMPUTE_H

#include "../model/representation.h"

#include <multi_img.h>
#include <shared_data.h>

#include <QGLBuffer>
#include <QGLFramebufferObject>

#include <tbb/atomic.h>
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/task.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/partitioner.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/tbb_allocator.h>
#include <boost/functional/hash.hpp>

#include <limits>
#include <algorithm>
#include <functional>

/* N: number of bands,
 * D: number of bins per band (discretization steps)
 */
/* a Bin is an entry in our N-dimensional sparse histogram
 * it holds a representative vector and is identified by its
 * hash key (the hash key is not part of the Bin class
 */
struct Bin {
	Bin() : weight(0.f) {}
	Bin(const multi_img::Pixel& initial_means)
		: weight(1.f), means(initial_means) {} //, points(initial_means.size()) {}

	/* we store the mean/avg. of all pixel vectors represented by this bin
	 * the mean is not normalized during filling the bin, only afterwards
	 */
	inline void add(const multi_img::Pixel& p) {
		/* weight holds the number of pixels this bin represents
		 */
		weight += 1.f;
		if (means.empty())
			means.resize(p.size(), 0.f);
		std::transform(means.begin(), means.end(), p.begin(), means.begin(),
					   std::plus<multi_img::Value>());
	}

	/* in incremental update of our BinSet, we can also remove pixels from a bin */
	inline void sub(const multi_img::Pixel& p) {
		weight -= 1.f;
		assert(!means.empty());
		std::transform(means.begin(), means.end(), p.begin(), means.begin(),
					   std::minus<multi_img::Value>());
	}

	float weight;
	std::vector<multi_img::Value> means;
	/* each bin can have a color calculated for the mean vector
	 */
	QColor rgb;
};

struct BinSet {
	BinSet(const QColor &c, int size)
		: label(c), boundary(size, std::make_pair((int)255, (int)0))
	{ totalweight = 0; }

	// Hash function for tbb::concurrent_hash_map
	struct vector_char_hash_compare {
		size_t hash( const  std::vector<unsigned char> & a ) const
		{
			// large random init
			size_t seed = 1878709926690269970;
			boost::hash_range(seed, a.begin(), a.end());
			return seed;
		}
		// compare vectors by value
		bool equal( std::vector<unsigned char> const &a,
					std::vector<unsigned char> const &b ) const
		{
			return a == b;
		}
	};


	/* each BinSet represents a label and has the label color
	 */
	QColor label;
	/* each entry is a N-dimensional vector, discretized by one char per band
	 * this means that we can have at most D = 256
	 */
	typedef std::vector<unsigned char> HashKey;
	/* the hash map holds all representative vectors (of size N)
	 * the hash realizes a sparse histogram
	 */
	typedef tbb::concurrent_hash_map<HashKey, Bin, vector_char_hash_compare>
			HashMap;
	HashMap bins;
	/* to set opacity value we normalize by total weight == sum of bin weights
	 * this is atomic to allow multi-threaded adding of vectors into the hash
	 */
	tbb::atomic<int> totalweight;
	/* the boundary is used for limiter mode initialization by label
	 * it is of length N and holds min, max bin indices occupied in each dimension
	 */
	std::vector<std::pair<int, int> > boundary;
};

typedef boost::shared_ptr<SharedData<std::vector<BinSet> > > sets_ptr;
typedef tbb::concurrent_vector<std::pair<int, BinSet::HashKey> > binindex;

struct ViewportCtx {
	representation::t type;

	// true if viewport has freshly computed data, but not on GPU yet
	tbb::atomic<int> wait;
	// true if viewport needs a full reset
	tbb::atomic<int> reset;

	/* metadata depending on image data */
	size_t dimensionality;
	std::vector<multi_img::BandDesc> meta;
	std::vector<QString> xlabels; 	// x-axis labels
	bool ignoreLabels;
	multi_img::Value binsize;
	multi_img::Value minval;
	multi_img::Value maxval;
	// true if metadata reflects current image information
	bool valid;

	/* metadata depending on display configuration */
	int nbins;
};

typedef boost::shared_ptr<SharedData<ViewportCtx> > vpctx_ptr;

class Compute
{
public:

	/* translate image value to value in binning coordinate system */
	static multi_img::Value curpos(const multi_img::Value &val, int dim,
								   const multi_img::Value &minval,
								   const multi_img::Value &binsize,
							  const std::vector<multi_img::Value> &illuminant
								   = std::vector<multi_img::Value>());

	/* method and helper class to preprocess bins before vertex generation */
	static void preparePolylines(const ViewportCtx &context,
								 std::vector<BinSet> &sets, binindex &index);

	class PreprocessBins {
	public:
		PreprocessBins(int label, size_t dimensionality, multi_img::Value maxval,
			const std::vector<multi_img::BandDesc> &meta,
			binindex &index)
			: label(label), dimensionality(dimensionality), maxval(maxval), meta(meta),
			index(index), ranges(dimensionality, std::pair<int, int>(INT_MAX, INT_MIN)) {}
		PreprocessBins(PreprocessBins &toSplit, tbb::split)
			: label(toSplit.label), dimensionality(toSplit.dimensionality),
			maxval(toSplit.maxval), meta(toSplit.meta),
			index(toSplit.index), ranges(dimensionality, std::pair<int, int>(INT_MAX, INT_MIN)) {}
		void operator()(const BinSet::HashMap::range_type &r);
		void join(PreprocessBins &toJoin);
		std::vector<std::pair<int, int> > GetRanges() { return ranges; }
	private:
		int label;
		size_t dimensionality;
		multi_img::Value maxval;
		const std::vector<multi_img::BandDesc> &meta;
		// pair of label index and hash-key within label's bin set
		binindex &index;
		std::vector<std::pair<int, int> > ranges;
	};

	/* method and helper class to extract and store vertice data from
	 * preprocessed bins */
	static void storeVertices(const ViewportCtx &context,
							 const std::vector<BinSet> &sets,
							 const binindex& index, QGLBuffer &vb,
							 bool drawMeans,
							 const std::vector<multi_img::Value> &illuminant);

	class GenerateVertices {
	public:
		GenerateVertices(bool drawMeans, size_t dimensionality, multi_img::Value minval, multi_img::Value binsize,
			const std::vector<multi_img::Value> &illuminant, const std::vector<BinSet> &sets,
			const binindex &index, GLfloat *varr)
			: drawMeans(drawMeans), dimensionality(dimensionality), minval(minval), binsize(binsize),
			illuminant(illuminant), sets(sets),
			index(index), varr(varr) {}
		void operator()(const tbb::blocked_range<size_t> &r) const;
	private:
		bool drawMeans;
		size_t dimensionality;
		multi_img::Value minval;
		multi_img::Value binsize;
		const std::vector<multi_img::Value> &illuminant;
		const std::vector<BinSet> &sets;
		const binindex &index;
		GLfloat *varr;
	};
};

#endif // DISTVIEWCOMPUTE_H
