#ifndef SOM_H
#define SOM_H

#include "edge_detection_config.h"
#include "neuron.h"

#include <similarity_measure.h>
#include <multi_img.h>

#include <iterator>
#include <opencv2/core/core.hpp>
#include <vector>

class SOM
{
public:
	class neighbourIterator;

protected:
	// This iterator can be extended in subclasses to support
	// the different neuron structures.
	class IteratorBase
	{
	public:
		IteratorBase() {}
		virtual ~IteratorBase() {}

		virtual void operator++() = 0;
		virtual Neuron &operator*() = 0;
		virtual IteratorBase *clone() = 0;

		// only use this function, if you are sure, that the dynamic
		// types of *this and *source are the same and source != NULL
		virtual void assignWith(const IteratorBase *source) = 0;

		bool operator==(const IteratorBase &other) const {
			return typeid(*this) == typeid(other) && equal(other);
		}

		// returns the coordinates of the neurons in the multi_img
		// that is created by export_2d
		virtual cv::Point get2dCoordinates() = 0;

		virtual cv::Point3d get3dPos() = 0;

		// iterates over all neurons if radius <= 0
		virtual neighbourIterator neighboursBegin(double radius) = 0;
		virtual neighbourIterator neighboursEnd(double radius) = 0;

	protected:
		 virtual bool equal(const IteratorBase &other) const = 0;
	};

	// This iterator can be extended in subclasses to support
	// the different grid structures.
	class NeighbourIteratorBase
	{
	public:
		NeighbourIteratorBase() {}
		virtual ~NeighbourIteratorBase() {}

		virtual void operator++() = 0;
		virtual Neuron &operator*() = 0;
		virtual NeighbourIteratorBase *clone() = 0;

		bool operator==(const NeighbourIteratorBase &other) const {
			return typeid(*this) == typeid(other) && equal(other);
		}

		virtual double getDistance() const = 0;
		virtual double getDistanceSquared() const = 0;
		virtual double getFakeGaussianWeight(double sigma) const = 0;

	protected:
		 virtual bool equal(const NeighbourIteratorBase &other) const = 0;
	};

public:
	// Iterator is returned by the functions of the SOM-interface.
	// It forwards the calls to IteratorBase, which can be extended
	// to the different grid structures in the subclasses
	// (we can only have pointers to IteratorBase because it is virtual)
	class iterator : public std::iterator<std::forward_iterator_tag, Neuron>
	{
	public:
		iterator(IteratorBase *baseIterator) : itr(baseIterator) { }
		iterator(const iterator &other) : itr(other.itr->clone()) { }
		// TODO: uncomment, when c++11 is enabled, dito move assignment operator
		// iterator(iterator &&other) : itr(other.itr) { other.itr = 0; }
		~iterator() { delete itr; }

		iterator &operator=(const iterator &other)
		{
			// catch assignment to self (would segfault on itr->clone after itr was deleted)
			if (this == &other) return *this;

			// if the types of the referenced iterators are equal, we do not
			// need to delete and allocate a new iterator object
			// we can just copy the content
			if (itr != NULL && other.itr != NULL &&
			    typeid(*itr) == typeid(*other.itr))
			{
				itr->assignWith(other.itr);
			}
			else
			{
				delete itr;
				itr = other.itr->clone();
			}
			return *this;
		}

		/*iterator &operator=(iterator &&other)
		{
			if (this == &other) return *this;

			delete itr;
			itr = other.itr;

			other.itr = NULL;

			return *this;
		}*/

		iterator &operator++()
		{
			++(*itr);
			return *this;
		}

		Neuron &operator*()
		{
			return *(*itr);
		}

		bool operator==(const iterator &other) const {
			return *itr == *other.itr;
		}
		bool operator!=(const iterator &other) { return !(*this==other); }

		// returns the coordinates of the neurons in the multi_img
		// that is created by export_2d
		cv::Point get2dCoordinates() { return itr->get2dCoordinates(); }

		cv::Point3d get3dPos() { return itr->get3dPos(); }

		// Returns a pointer to the IteratorBase, that is stored in itr
		// ! Keep in mind that this pointer is deleted in ~Iterator !
		IteratorBase *getBase() const { return itr; }

		// iterates over all neurons if radius <= 0
		neighbourIterator neighboursBegin(double radius) {
			return itr->neighboursBegin(radius);
		}
		neighbourIterator neighboursEnd(double radius) {
			return itr->neighboursEnd(radius);
		}

	protected:
		IteratorBase *itr;
	};

	class neighbourIterator : public std::iterator<std::forward_iterator_tag, Neuron>
	{
	public:
		neighbourIterator(NeighbourIteratorBase *baseIterator) : itr(baseIterator) {}
		neighbourIterator(const neighbourIterator &other) : itr(other.itr->clone()) {}
		~neighbourIterator() { delete itr; }

		neighbourIterator &operator=(const neighbourIterator &other)
		{
			// catch assignment to self (would segfault itr->clone after itr was deleted)
			if (itr == other.itr) return *this;

			delete itr;
			itr = other.itr->clone();
			return *this;
		}

		neighbourIterator &operator++()
		{
			++(*itr);
			return *this;
		}

		Neuron &operator*()
		{
			return *(*itr);
		}

		// speed optimization: this function is undefined for Iterators, that
		// were created with different parameters (e.g. different radii / bases)
		bool operator==(const neighbourIterator &other) const {
			return *itr == *other.itr;
		}
		bool operator!=(const neighbourIterator &other) { return !(*this==other); }

		double getDistance() const { return itr->getDistance(); }
		double getDistanceSquared() const { return itr->getDistanceSquared(); }
		double getFakeGaussianWeight(double sigma) const
				{ return itr->getFakeGaussianWeight(sigma); }

	protected:
		NeighbourIteratorBase *itr;
	};

	class Cache
	{
	public:
		Cache()	: preloaded(false) { }

		virtual void preload(const multi_img &image) = 0;

		// returns the distance between the positions of the winner
		// neurons at the given image position
		virtual double getDistance(const multi_img::Pixel &v1,
		                           const multi_img::Pixel &v2,
		                           const cv::Point &c1,
		                           const cv::Point &c2) = 0;

		// the following functions are only useable when preload was called

		// returns the similarity of the winner neurons of the given image positions
		virtual SOM::iterator getWinnerNeuron(cv::Point p) = 0;
		virtual double getSimilarity(
			vole::SimilarityMeasure<multi_img::Value> *distfun,
			const cv::Point &p1,
			const cv::Point &p2) = 0;
		virtual double getSobelX(int x, int y) = 0;
		virtual double getSobelY(int x, int y) = 0;

	protected:
		bool preloaded;
	};

protected:
	SOM(const vole::EdgeDetectionConfig &conf, int dimension,
		std::vector<multi_img_base::BandDesc> meta);

public:
	virtual ~SOM();

	// Factories
	static SOM *createSOM(const vole::EdgeDetectionConfig &conf,
						  int dimensions,
						  std::vector<multi_img_base::BandDesc> meta
						  = std::vector<multi_img_base::BandDesc>());
	static SOM *createSOM(const vole::EdgeDetectionConfig &conf,
						  const multi_img &data,
						  std::vector<multi_img_base::BandDesc> meta
						  = std::vector<multi_img_base::BandDesc>());

	/**
	  @arg img_height height of the pixel cache (image size, not som size)
	  @arg img_width width of the pixel cache
	  */
	virtual SOM::Cache *createCache(int img_height, int img_width) = 0;

	// Amount of neurons in the SOM
	virtual int size() const = 0;

	// bijective mapping of the neurons to a 2d plane
	virtual int get2dWidth() const = 0;
	virtual int get2dHeight() const = 0;
	multi_img export_2d();

	virtual iterator identifyWinnerNeuron(const multi_img::Pixel &inputVec);

	// writes a list of (distance, iterator) tuples to coords,
	// that is sorted by distance, N = coords.size()
	void closestN(const multi_img::Pixel &inputVec,
		std::vector<std::pair<double, SOM::iterator> > &coords);

	// returns the amount of updated neurons
	virtual int updateNeighborhood(iterator &neuron,
							const multi_img::Pixel &input,
							double sigma, double learnRate);

	virtual double getDistanceBetweenWinners(const multi_img::Pixel &v1,
	                                         const multi_img::Pixel &v2) = 0;

	virtual cv::Vec3f getColor(cv::Point3d pos) = 0;

	// short info about type, shape and #neurons of the SOM
	virtual std::string description() = 0;

	// Iterators for iterating over all neurons of the SOM
	virtual iterator begin() = 0;
	const iterator &end() const { return theEnd; }

	inline double getSimilarity(const Neuron &n1, const Neuron &n2) {
		return distfun->getSimilarity(n1, n2);
	}

	void getEdge(const multi_img &image,
				 cv::Mat1d &dx, cv::Mat1d &dy);

	void getNeuronDistancePlot(vole::SimilarityMeasure<multi_img::Value> *distMetric,
	                       std::vector<double> &xVals,
	                       std::vector<double> &yVals);

protected:
	const vole::EdgeDetectionConfig &config;
	const std::vector<multi_img_base::BandDesc> origImgMeta;
	const int dim;		///< Dimension of each neuron / the SOM
	iterator theEnd;

private:
	vole::SimilarityMeasure<multi_img::Value> *distfun;
};

#endif // SOM_H
