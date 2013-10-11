#include "som.h"

// for factory methods
#include "som2d.h"
#include "som3d.h"

#include <sm_factory.h>

#include <fstream>

SOM::SOM(const vole::EdgeDetectionConfig &conf, int dimension,
         std::vector<multi_img_base::BandDesc> meta)
	: config(conf), dim(dimension), origImgMeta(meta),
	theEnd(SOM::iterator(NULL)) // shall be overwritten in child constructors
{
	/// Create similarity measure
	distfun = vole::SMFactory<multi_img::Value>::spawn(config.similarity);
	assert(distfun);
}

SOM::~SOM()
{
	delete distfun;
}

SOM* SOM::createSOM(const vole::EdgeDetectionConfig &conf,
					int dimensions,
					std::vector<multi_img_base::BandDesc> meta)
{
	switch (conf.type)
	{
	case vole::SOM_LINE:
		// SOM2d constructor will create 1d SOM
	case vole::SOM_SQUARE:
		return new SOM2d(conf, dimensions, meta);
		return new SOM2d(conf, dimensions, meta);
	case vole::SOM_CUBE:
		return new SOM3d(conf, dimensions, meta);
	default:
		return 0;
	}
}

SOM* SOM::createSOM(const vole::EdgeDetectionConfig &conf,
					const multi_img &data,
					std::vector<multi_img_base::BandDesc> meta)
{
	switch (conf.type)
	{
	case vole::SOM_LINE:
		// SOM2d constructor will create 1d SOM
	case vole::SOM_SQUARE:
		return new SOM2d(conf, data, meta);
	case vole::SOM_CUBE:
		return new SOM3d(conf, data, meta);
	default:
		return 0;
	}
}

multi_img SOM::export_2d()
{
	multi_img ret = multi_img(get2dHeight(), get2dWidth(), dim);

	if (!origImgMeta.empty())
		ret.meta = origImgMeta;

	multi_img::Range r = ret.data_range();
	ret.maxval = r.max;
	ret.minval = r.min;

	for (SOM::iterator n = begin(); n != end(); ++n)
	{
		cv::Point p = n.get2dCoordinates();
		ret.setPixel(p.y, p.x, *n);
	}
	return ret;
}

SOM::iterator SOM::identifyWinnerNeuron(const multi_img::Pixel &inputVec)
{
	// initialize with maximum value
	double closestDistance = std::numeric_limits<double>::max();

	// init winner with non existent value
	SOM::iterator winner = end();

	// find closest Neuron to inputVec in the SOM
	// -> iterate over all neurons in grid
	for (SOM::iterator neuron = begin(); neuron != end(); ++neuron) {
		double dist = getSimilarity(*neuron, inputVec);

		// compare current distance with minimal found distance
		if (dist < closestDistance) {
			// set new minimal distance and winner position
			closestDistance = dist;
			winner = neuron;
		}
	}
	assert(winner != end());
	return winner;
}

// comparer for heap in closestN()
static bool sortpair(std::pair<double, SOM::iterator> i,
					 std::pair<double, SOM::iterator> j) {
	return (i.first < j.first);
}

void SOM::closestN(const multi_img::Pixel &inputVec,
	std::vector<std::pair<double, SOM::iterator> > &heap)
{
	// initialize with maximum values
	for (int i = 0; i < heap.size(); ++i)
		heap[i].first = std::numeric_limits<double>::max();

	// find closest Neurons to inputVec in the SOM
	// iterate over all neurons in grid
	for (SOM::iterator neuron = begin(); neuron != end(); ++neuron) {
		double dist = distfun->getSimilarity(*neuron, inputVec);
		/* compare current distance with the maximum of the N shortest
		 * found distances */
		if (dist < heap[0].first) {
			// remove max. value in heap
			std::pop_heap(heap.begin(), heap.end(), sortpair);

			/* max element is now on position "back" and should be popped
			 * instead we overwrite it directly with the new element */
			std::pair<double, SOM::iterator> &back = heap.back();
			back.first = dist;
			back.second = neuron;

			std::push_heap(heap.begin(), heap.end(), sortpair);
		}
	}

	assert(heap[0].first != std::numeric_limits<double>::max());
	std::sort_heap(heap.begin(), heap.end(), sortpair); // sort ascending
}

/* fallback method that should be overwritten, but can be used as reference
 * calculates the exp function for all neurons -> slow */
int SOM::updateNeighborhood(SOM::iterator &neuron,
							 const multi_img::Pixel &input,
							 double sigma, double learnRate)
{
	const double INF = std::numeric_limits<double>::infinity();
	SOM::neighbourIterator theEnd = neuron.neighboursEnd(INF);

	// Generic method that calculates the weight for all neurons
	int updates = 0;
	for (SOM::neighbourIterator neighbour = neuron.neighboursBegin(INF);
		 neighbour != theEnd; ++neighbour)
	{
		double weight = learnRate * neighbour.getFakeGaussianWeight(sigma);
		if (weight >= 0.01) // for consistency, doesn't save us much here
		{
			(*neighbour).update(input, weight);
			++updates;
		}
	}
	return updates;
}

void SOM::getEdge(const multi_img &image, cv::Mat1d &dx, cv::Mat1d &dy)
{
	std::cout << "Calculating derivatives (dx, dy)" << std::endl;

	Cache *cache = createCache(image.height, image.width);
	cache->preload(image);

	dx = cv::Mat::zeros(image.height, image.width, CV_64F);
	dy = cv::Mat::zeros(image.height, image.width, CV_64F);

	double maxIntensity = 0.0;

	for (int y = 1; y < image.height-1; y++) {
		double valx, valy, valxAbs, valyAbs;

		for (int x = 1; x < image.width-1; x++) {
			// x-direction
			valx = cache->getSobelX(x, y);
			valxAbs = std::fabs(valx);
			if (valxAbs > maxIntensity)
				maxIntensity = valx;
			dx[y][x] = valx;

			// y-direction
			valy = cache->getSobelY(x, y);
			valyAbs = std::fabs(valy);
			if (valyAbs > maxIntensity)
				maxIntensity = valyAbs;
			dy[y][x] = valy;
		}
	}

	delete cache;

	// normalization
	cv::MatIterator_<double> ix, iy;
	for (ix = dx.begin(), iy = dy.begin(); ix != dx.end(); ++ix, ++iy) {
		// [-X .. X] --( /2X )--> [-0.5 .. 0.5] --( +0.5 )--> [0 .. 1]
		*ix = (*ix / (2*maxIntensity)) + 0.5;
		*iy = (*iy / (2*maxIntensity)) + 0.5;
	}
}

void SOM::getNeuronDistancePlot(vole::SimilarityMeasure<multi_img_base::Value> *distMetric,
                            std::vector<double> &xVals,
                            std::vector<double> &yVals)
{
	xVals.reserve(size() * size());
	yVals.reserve(size() * size());
	SOM::iterator theEnd = end();
	for (SOM::iterator i = begin(); i != theEnd; ++i)
	{
		for (SOM::iterator j = begin(); j != theEnd; ++j)
		{
			if (i != j)
			{
				// distance between the neurons in the SOM
				cv::Point3d d = i.get3dPos() - j.get3dPos();
				double positionDistance = std::sqrt(d.dot(d));
				xVals.push_back(positionDistance);

				// distance between the neuron vectors
				double valueDistance = distMetric->getSimilarity(*i, *j);
				yVals.push_back(valueDistance);
			}
		}
	}
}
