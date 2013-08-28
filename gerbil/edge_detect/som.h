#ifndef SOM_H
#define SOM_H

#include "edge_detection_config.h"
#include "neuron.h"

#include <similarity_measure.h>
#include <multi_img.h>

#include <iterator>
#include <opencv2/core/core.hpp>
#include <vector>

// === Major TODOs ===

// remove fixed seed and verbosity in falsecolor.cpp

// updateNeighbourhood of SOMCone

// somcone: force granularity to be <= 0.25, test small cone sizes

// meanshift shell testen

// copy meta information from orig image?
// also min and max value would be interesting for initialization and export_2d

// === Minor TODOs / code structure ===

// SOM3d -> SOMCube, SOM2d -> SOM...
// dito iteratoren

// end() methods in iterators are not constant. this means one malloc per iteration
//  \-> call it before the loop

// for all som-subclasses, constructor with data-multi_img:
// provide some checks, if the provided multi_img has the correct form

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
		~iterator() { delete itr; }

		iterator &operator=(const iterator &other)
		{
			delete itr;
			itr = other.itr->clone();
			return *this;
		}

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
		double getFakeGaussianWeight(double sigma) const { itr->getFakeGaussianWeight(sigma); }

	protected:
		NeighbourIteratorBase *itr;
	};

	class Cache
	{
	public:
		Cache() : preloaded(false) { }

		virtual void preload(const multi_img &image) = 0;

		virtual double getDistance(const multi_img::Pixel &v1,
								   const multi_img::Pixel &v2,
								   const cv::Point &c1,
								   const cv::Point &c2) = 0;

		// only useable when preload was called, fails otherwise
		virtual double getSobelX(int x, int y) = 0;
		virtual double getSobelY(int x, int y) = 0;

	protected:
		bool preloaded;
	};

protected:
	SOM(const vole::EdgeDetectionConfig &conf, int dimension);

public:
	virtual ~SOM();

	// Factories
	static SOM *createSOM(const vole::EdgeDetectionConfig &conf,
						  int dimensions);
	static SOM *createSOM(const vole::EdgeDetectionConfig &conf,
						  const multi_img &data);

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

	// returns a list of (distance, iterator) tuples, that is sorted by distance
	std::vector<std::pair<double, SOM::iterator> >
	closestN(const multi_img::Pixel &inputVec, unsigned int N);

	// returns the amount of updated neurons
	virtual int updateNeighborhood(iterator &neuron,
							const multi_img::Pixel &input,
							double sigma, double learnRate);

	virtual double getDistanceBetweenWinners(const multi_img::Pixel &v1,
											 const multi_img::Pixel &v2) = 0;

	// short info about type, shape and #neurons of the SOM
	virtual std::string toString() = 0;

	// Iterators for iterating over all neurons of the SOM
	virtual iterator begin() = 0;
	virtual iterator end() = 0;

	inline double getSimilarity(const Neuron &n1, const Neuron &n2) {
		distfun->getSimilarity(n1, n2);
	}

	void getEdge(const multi_img &image,
				 cv::Mat1d &dx, cv::Mat1d &dy);

protected:
	const vole::EdgeDetectionConfig &config;
	const int dim;		///< Dimension of each neuron / the SOM

private:
	vole::SimilarityMeasure<multi_img::Value> *distfun;
};

#endif // SOM_H
