#ifndef GENSOM_H
#define GENSOM_H

#include <opencv2/core/core.hpp>

#include <sm_config.h>
#include <iosfwd>

#include "som_neuron.h"
#include "som_config.h"

namespace vole {
	class ProgressObserver;
}

struct DistIndexPair {
	typedef Neuron::value_type value_type; // TODO: inconsistent usage
	DistIndexPair()
		: dist(std::numeric_limits<value_type>::infinity()), index(0)
	{}
	DistIndexPair(value_type dist, size_t index)
		: dist(dist), index(index)
	{}
	/** Distance of the best matching unit to query vector. */
	value_type dist;
	/** Linear index into SOM neuron storage. */
	size_t index;
};

/** Compare function to sort DistIndexPair s by distance. */
inline bool cmpDist(
		DistIndexPair& a,
		DistIndexPair& b)
{
	return (a.dist < b.dist);
}

/** Compute geometric series neuron weights.
 * w_i = 2 w_(i+1)
 * where w_0 is the weight of the best matching neuron.
 * @param w std::vector<T> of size n, with n being the number of best
 *			matching neurons.
 */
template <typename T>
void neuronWeightsGeometric(std::vector<T> & w)
{
	const int n = w.size();
	if (0 == n) {
		throw std::runtime_error("neuronWeightsGeometric(): w.size() == 0");
	} else if (1 == n) {
		w[0] = 1.0;
		return;
	}

	// geometric series sum sn = 2^n - 1
	const T sn = std::ldexp(1.0, n) - 1;
	const T rsn = 1.0/sn;

	w[n-1] = rsn;
	double ws = rsn;
	for(int i=n-2; 0 <= i; --i) {
		w[i] = 2*w[i+1];
		ws += w[i];
	}
	assert(std::abs(ws-1.0) < std::numeric_limits<T>::epsilon() * 100.0);
}

/** Abstract n-dimensional SOM class.
 *
 * This abstract class implements neuron (aka. unit) storage and stores
 * SOMConfig.
 */
class GenSOM
{
public:
	typedef Neuron::value_type value_type;

	virtual ~GenSOM();

	/** Factory: Create a SOM by loading a binary file or by training on multi_img.
	 *
	 * If SOMConfig::somFile is a non-empty string and the file exists, the
	 * SOM data is loaded from this file. Otherwise, the SOM is trained on the
	 * multi_img img. In the latter case, the SOM is also written to
	 * SOMConfig::somFile if SOMConfig::writeSOM is true.
	 *
	 * @param img multi_img to train the SOM on.
	 */
	static GenSOM* create(const SOMConfig& conf, const multi_img& img,
						  vole::ProgressObserver *po = 0);

	/** Train SOM on multi_img.
	*/
	void train(const multi_img & input, vole::ProgressObserver *po = 0);

	size_t size() const { return neurons.size(); }
	virtual cv::Size size2D() const = 0;

	SOMConfig const& getConfig() const {
		return config;
	}

	/** Return neuron at linear index idx. */
	Neuron const& neuron(size_t idx) const {
		return neurons.at(idx);
	}

	/** Find best matching unit for inputVec.
	 *
	 * @return Pair of distance and linear index into the SOM's neuron array.
	 */
	DistIndexPair findBMU(const multi_img::Pixel &inputVec) const;

	/** Find closest n neurons for inputVec.
	 *
	 * Also known as k-nearest neighbours (kNN).
	 * The result vector is sorted by ascending distance.
	 */
	std::vector<DistIndexPair> findClosestN(const multi_img::Pixel &p,
											size_t n) const
	{
		std::vector<DistIndexPair> ret(n);
		findClosestN(p, ret.begin(), ret.end());
		return ret;
	}

	/** Find closest n neurons for inputVec - iterator implementation.
	 *
	 * The result is stored in the range [dfirst, dlast) where 
	 * dlast - dfirst = n. The range [dfirst, dlast) is sorted by ascending
	 * distance.  
	 *
	 * T must meet the requirements of the RandomAccessIterator concept.
	 */
	template<typename T>
	void findClosestN(const multi_img::Pixel &inputVec,
					  T dfirst, T dlast) const;

	/** Return a higher-dimensional coordinate for a neuron at index idx,
	 * @param normalize return a coordinate in [0,1]
	 * @see vec2Point3
	 */
	virtual std::vector<float>
	getCoord(size_t idx, bool normalize = true) const = 0;

	/** Return a two-dimensional coordinate for a neuron at index idx.
	 * This is helpful for visualizing any data associated with the SOM in 2D.
	 * Depending on the SOM structure, it might be pretty, or in the worst case
	 * just one long line.
	 */
	virtual cv::Point getCoord2D(size_t idx) const = 0;

	/** Write SOM in gerbil binary SOM format.
	 *
	 * @param os Output stream in std::ios::bin mode. */
	void saveFile(std::ostream &os) const;
	void saveFile(const std::string &fileName) const;

	/** Load SOM data from gerbil binary SOM file.
	 * @param is Input stream in std::ios::bin mode. */
	static GenSOM* loadFile(std::istream &is, const SOMConfig& config);
	static GenSOM* loadFile(const std::string &fileName, const SOMConfig& config);

	/** Compose multi_img based on SOM values ordered in a 2D structure */
	multi_img img(const std::vector<multi_img_base::BandDesc> &meta,
				  const multi_img_base::Range &range);

	/** Compute RGB representation of SOM in 2D (useful for debugging) */
	cv::Mat3f bgr(const std::vector<multi_img::BandDesc> &meta,
				  multi_img::Value maxval);

protected:
	/// Store config, allocate SimilarityMeasure. Does not initialize neurons.
	GenSOM(const SOMConfig& config);

	/** Reserve neuron storage
	 * This function is typically called by constructors of derived classes
	 * @param randomize If true, fill neurons with uniform random values
	 * from [0,1].
	 */
	void init(size_t nbands, size_t nneurons, bool randomize);

	virtual int updateNeighborhood(size_t index,
								   const multi_img::Pixel &input,
								   double sigma, double learnRate) = 0;
	// helper to train()
	int trainSingle(const multi_img::Pixel &input, int iter, int max);
	// helper to updateNeighborhood()
	double gaussWeight(double distance, double sigma, double learnRate);
	// is called before feeding
	virtual void notifyTrainingStart() {}
	// is called after feeding
	virtual void notifyTrainingEnd() {}
	/** Factory: Create un-trained and un-initialized SOM with parameters from
	 * SOMConfig and allocated storage.
	 *
	 * This is the protected counter-part to the public create(), used by
	 * loadFile().
	 *
	 * @param nbands Number of bands of each neuron.
	 * @param randomize If true, fill neurons with uniform randomized values
	 * from [0,1].
	*/
	static GenSOM* create(const SOMConfig& conf, size_t nbands, bool randomize);

	SOMConfig config;

	// Flat storage of n-dimensional SOM neuron structure.
	std::vector<Neuron> neurons;

	vole::SimilarityMeasure<value_type> *distfun;

private:
	GenSOM(); // undefined
	GenSOM(const GenSOM& other); // undefined
	GenSOM& operator=(const GenSOM& other); // undefined
};

template<typename T> // T is iterator to a container of DistIndexPairs
void GenSOM::findClosestN(const multi_img::Pixel &inputVec,
						  T dfirst, T dlast) const
{
	// initialize heap with infinity distances
	std::fill(dfirst,
			  dlast,
			  DistIndexPair());

	for (std::vector<Neuron>::const_iterator it = neurons.begin();
		 it != neurons.end();
		 ++it)
	{
		value_type dist = distfun->getSimilarity(*it, inputVec);

		if (dist < dfirst->dist) {
			// remove max. value in heap
			std::pop_heap(dfirst, dlast, cmpDist);

			// max element is now on position "back" and should be popped
			// instead we overwrite it directly with the new element
			DistIndexPair &back = *(dlast-1);
			back = DistIndexPair(dist,                  // distance
								 it - neurons.begin()); // index into neurons
			std::push_heap(dfirst, dlast, cmpDist);
		}
	}
	std::sort_heap(dfirst, dlast, cmpDist); // sort ascending
}

/** Build cv::Point3 from vector v with 1 <= v.size() <= 3.
 *
 * Useful for converting GenSOM::getCoord() result.
 */
template<typename T>
inline cv::Point3_<T> vec2Point3(std::vector<T> const& v)
{
	assert(1 <= v.size() && v.size() <= 3);
	cv::Point3_<T> p(0,0,0);
	switch(v.size()) {
	case 3:
		p.z = v[2];
	case 2:
		p.y = v[1];
	case 1:
		p.x = v[0];
	default:
		;
	}
	return p;
}
#endif // GENSOM_H
