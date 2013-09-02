#include "som_cone.h"

#include <sstream>

SOMCone::SOMCone(const vole::EdgeDetectionConfig &conf, int dimension,
				 std::vector<multi_img_base::BandDesc> meta)
	: SOM(conf, dimension, meta), coordinates(std::vector<cv::Point3d>()),
	  slicePtr(std::vector<int>()), sliceZ(std::vector<double>()),
	  granularity(conf.granularity)
{
	initCoordinates(dimension);

	/// Uniformly randomizes each neuron
	// TODO: given interval [0..1] sure? purpose? it will not fit anyway
	cv::RNG rng(config.seed);

	SOM::iterator theEnd = end(); // only needs to be called once
	for (SOM::iterator n = begin(); n != theEnd; ++n) {
		(*n).randomize(rng, 0., 1.);
	}
}

SOMCone::SOMCone(const vole::EdgeDetectionConfig &conf, const multi_img &data,
				 std::vector<multi_img_base::BandDesc> meta)
	: SOM(conf, data.size(), meta), coordinates(std::vector<cv::Point3d>()),
	  slicePtr(std::vector<int>()), sliceZ(std::vector<double>()),
	  granularity(conf.granularity)
{
	initCoordinates(data.size());

	// check format
	if (data.width != size() || data.height != 1) {
		std::cerr << "SOM image has wrong dimensions!" << std::endl;
		assert(false);
		return; // somdata will be empty
	}

	/// Read SOM from multi_img
	SOM::iterator theEnd = end(); // only needs to be called once
	for (SOM::iterator n = begin(); n != theEnd; ++n) {
		cv::Point pos = n.get2dCoordinates();
		*n = data(pos.y, pos.x);
	}
}

void SOMCone::initCoordinates(int dimension)
{
	// init neuron coordinates
	int height = (int)(1.0 / config.granularity); // steps in z direction (+1)
	int radius = (int)(0.5 / config.granularity); // steps in x / y direction (+1)
	double radDivHeight = (double)radius / (double)height;

	for (int z = 0; z <= height; z++)
	{
		double currZ = z * config.granularity;
		double currRadius = radDivHeight * currZ; // equiangular triangle
		double currRadiusSquared = currRadius * currRadius;

		slicePtr.push_back(coordinates.size());
		sliceZ.push_back(currZ);

		// TODO: use currRadius in a way, that it doesn't affect the result - would save some work
		for (int y = -radius; y <= radius; y++)
		{
			double currY = ((double)y / (double)(2 *radius));
			double currYSquare = currY * currY;

			for (int x = -radius; x <= radius; x++)
			{
				double currX = ((double)x / (double)(2 * radius));
				if (currYSquare + currX * currX <= currRadiusSquared)
				{
					// this prints the coordinates of all existing neurons
					//std::cout << "currZ=" << currZ << " currY=" << currY << " currX=" << currX << " currRadius=" << currRadius << std::endl;
					coordinates.push_back(cv::Point3d(currX, currY, currZ));
				}
			}
		}
	}
	slices = slicePtr.size();
	slicePtr.push_back(coordinates.size()); // (slicePtr[slices] == #neurons)

	neurons = std::vector<Neuron>(slicePtr[slices], Neuron(dimension));
}

SOMCone::CacheCone *SOMCone::createCache(int img_height, int img_width)
{
	return new CacheCone(this, img_height, img_width);
}

// no virtual function call overhead inside the loop(s)
SOM::iterator SOMCone::identifyWinnerNeuron(const multi_img::Pixel &inputVec)
{
	// initialize with maximum / invalid value
	double closestDistance = std::numeric_limits<double>::max();
	int winnerIdx = -1;

	// find closest Neuron to inputVec in the SOM
	// -> iterate over all neurons in grid
	for (int idx = 0; idx < size(); ++idx) {
		double dist = getSimilarity(neurons[idx], inputVec);

		// compare current distance with minimal found distance
		if (dist < closestDistance) {
			// set new minimal distance and winner position
			closestDistance = dist;
			winnerIdx = idx;
		}
	}
	assert(winnerIdx != -1);
	return SOM::iterator(new IteratorCone(this, winnerIdx));
}

// buggy
/*int SOMCone::updateNeighborhood(SOM::iterator &neuron,
								 const multi_img::Pixel &input,
								 double sigma, double learnRate)
{
	IteratorCone *it = static_cast<IteratorCone *>(neuron.getBase());
	int winnerIdx = it->getIdx();
	double sigmaSquare = sigma * sigma;

	// find first slice, that has neurons with at least the minimal weight
	int nextSlice = slices;
	for (int sliceIdx = 0; sliceIdx < slices; ++sliceIdx)
	{
		// one could throw the fabs out, if the storage direction along the z axis
		// is fixed. if it is changed later anyways, this bug will be hard to find
		double dist = std::fabs(coordinates[winnerIdx].z - sliceZ[nextSlice]);
		if (learnRate * exp(-(dist)/(2.0*sigmaSquare)) >= 0.01)
		{
			nextSlice = sliceIdx + 1;
			break;
		}
	}

	// iterate over pixels within |neuron + other|^2 <= radius, cache aligned
	// starts at the first relevant slice, stops at the first slice outside
	int updates = 0;
	double maximum = 0;
	for (int idx = slicePtr[nextSlice - 1];; ++idx)
	{
		// we reached the end of the current slice
		if (idx == slicePtr[nextSlice])
		{
			// we reached the end of the som
			if (idx == size()) break;

			// check, if we left the range of slices that can contain relevant neurons
			double dist = std::fabs(coordinates[winnerIdx].z - sliceZ[nextSlice]);
			if (learnRate * exp(-(dist)/(2.0*sigmaSquare)) < 0.01) break;

			// go on with the next slice
			++nextSlice;
		}

		double dist = getDistanceSquared(winnerIdx, idx);
		double weight = learnRate * exp(-(dist)/(2.0*sigmaSquare));
		if (weight > maximum) maximum = weight;
		if (weight >= 0.01)
		{
			++updates;
			neurons[idx].update(input, weight);
		}
	}
	return updates;
}*/

double SOMCone::getDistanceBetweenWinners(const multi_img::Pixel &v1,
										  const multi_img::Pixel &v2)
{
	// This is a SOMCone, so identifyWinnerNeuron returns IteratorCones
	// Is it better to override identifyWinnerNeuron in the subclasses?
	// unneccessary code duplication, but would remove the somewhat dangerous cast
	SOM::iterator iter1 = identifyWinnerNeuron(v1);
	IteratorCone *it1 = static_cast<IteratorCone *>(iter1.getBase());
	cv::Point3d &p1 = coordinates[it1->getIdx()];

	SOM::iterator iter2 = identifyWinnerNeuron(v2);
	IteratorCone *it2 = static_cast<IteratorCone *>(iter2.getBase());
	cv::Point3d &p2 = coordinates[it2->getIdx()];

	getDistance(p1, p2);
}

cv::Vec3f SOMCone::getColor(cv::Point3d pos)
{
	const double PI = 3.141592653589793;

	// calculate hsv values
	double h, s, v;
	h = std::atan2(pos.y, pos.x) + PI; // only x and y affect hue, h has range 0 - 2*PI
	s = std::sqrt(pos.x * pos.x + pos.y * pos.y); // only x and y affect saturation
	if (pos.z > 0) s /= 0.5 * pos.z; // normalize by radius at height avg.z of the cone
	else s = 0;
	v = pos.z; // the cone has a height of 1

	// convert hsv2rgb - see german wikipedia article
	double f, p, q, t;
	int h_i = (int)(h / (PI / 3));
	f = (h / (PI / 3)) - h_i;
	p = v * (1 - s);
	q = v * (1 - s * f);
	t = v * (1 - s * (1-f));

	cv::Vec3f pixel;
	switch (h_i)
	{
	case 0:
	case 6:
		pixel = cv::Vec3f(v, t, p);
		break;
	case 1:
		pixel = cv::Vec3f(q, v, p);
		break;
	case 2:
		pixel = cv::Vec3f(p, v, t);
		break;
	case 3:
		pixel = cv::Vec3f(p, q, v);
		break;
	case 4:
		pixel = cv::Vec3f(t, p, v);
		break;
	case 5:
		pixel = cv::Vec3f(v, p, q);
		break;
	}

	if (pixel[0] < 0) pixel[0] = 0;
	else if (pixel[0] > 1) pixel[0] = 1;

	if (pixel[1] < 0) pixel[1] = 0;
	else if (pixel[1] > 1) pixel[1] = 1;

	if (pixel[2] < 0) pixel[2] = 0;
	else if (pixel[2] > 1) pixel[2] = 1;

	return pixel;
}

std::string SOMCone::description()
{
	std::stringstream s;
	s << "SOM of type cone, granularity " << granularity;
	s << ", with " << size() << " neurons of dimension " << dim;
	s << ", seed=" << config.seed;
	return s.str();
}

SOM::iterator SOMCone::begin()
{
	return SOM::iterator(new IteratorCone(this, 0));
}

SOM::iterator SOMCone::end()
{
	return SOM::iterator(new IteratorCone(this, size()));
}

double SOMCone::getDistance(int idx1, int idx2)
{
	assert(idx1 < size() && idx2 < size());
	cv::Point3d &p1 = coordinates[idx1];
	cv::Point3d &p2 = coordinates[idx2];
	return getDistance(p1, p2);
}

double SOMCone::getDistance(cv::Point3d &p1, cv::Point3d &p2)
{
	cv::Point3d d = p1 - p2;
	return std::sqrt(d.dot(d)); // TODO: IEEE inv-sqrt approximation should be enough...
}

double SOMCone::getDistanceSquared(int idx1, int idx2)
{
	assert(idx1 < size() && idx2 < size());
	cv::Point3d &p1 = coordinates[idx1];
	cv::Point3d &p2 = coordinates[idx2];
	return getDistanceSquared(p1, p2);
}

double SOMCone::getDistanceSquared(cv::Point3d &p1, cv::Point3d &p2)
{
	cv::Point3d d = p1 - p2;
	return (d.dot(d));
}

void SOMCone::IteratorCone::operator++()
{
	++idx;
}

bool SOMCone::IteratorCone::equal(const SOM::IteratorBase &other) const
{
	// Types were checked in the super class
	const IteratorCone &o = static_cast<const IteratorCone &>(other);
	return this->base == o.base && this->idx == o.idx;
}

SOM::neighbourIterator SOMCone::IteratorCone::neighboursBegin(double radius)
{
	assert(radius >= 0);
	return SOM::neighbourIterator(
				new NeighbourIteratorCone(base, idx, radius));
}

SOM::neighbourIterator SOMCone::IteratorCone::neighboursEnd(double radius)
{
	return SOM::neighbourIterator(
				new NeighbourIteratorCone(base, idx, radius, true));
}

SOMCone::NeighbourIteratorCone::NeighbourIteratorCone(SOMCone *base,
							 int neuronIdx, double radius, bool end)
	: SOM::NeighbourIteratorBase(), base(base),
	  neuron(base->coordinates[neuronIdx]), radius(radius)
{
	if (end)
	{
		idx = base->size();
		nextSlice = 0; // not relevant in comparison
		return;
	}

	// find start index (find first slice, that is in radius, use its first idx)
	for (int sliceIdx = 0; sliceIdx < base->slices; ++sliceIdx)
	{
		// one could throw the fabs out, if the storage direction along the z axis
		// is fixed. if it is changed later anyways, this bug will be hard to find
		if (std::fabs(neuron.z - base->sliceZ[sliceIdx]) <= radius)
		{
			idx = base->slicePtr[sliceIdx];
			nextSlice = sliceIdx + 1;
			return;
		}
	}
	assert(false);
}

void SOMCone::NeighbourIteratorCone::operator++()
{
	// iterate over pixels within |neuron + other|^2 <= radius, cache aligned
	// starts at the first relevant slice, stops at the first slice outside
	do
	{
		++idx;

		if (idx == base->slicePtr[nextSlice])
		{
			// check if we left the range of possible slices
			if (idx >= base->size() ||
				std::fabs(neuron.z - base->sliceZ[nextSlice]) > radius)
			{
				idx = base->size();
				return;
			}
			++nextSlice;
		}
	}
	while (getDistanceSquared() > (radius * radius));
}

double SOMCone::NeighbourIteratorCone::getDistance() const
{
	cv::Point3d d = neuron - base->coordinates[idx];
	return std::sqrt(d.dot(d)); // TODO: IEEE inv-sqrt approximation should be enough...
}

double SOMCone::NeighbourIteratorCone::getDistanceSquared() const
{
	cv::Point3d d = neuron - base->coordinates[idx];
	return (d.dot(d));
}

double SOMCone::NeighbourIteratorCone::getFakeGaussianWeight(double sigma) const
{
	double dist = getDistanceSquared();
	return exp(-(dist)/(2.0*sigma));
}

bool SOMCone::NeighbourIteratorCone::equal(const NeighbourIteratorBase &other) const
{
	// Types were checked in the super class
	const NeighbourIteratorCone &o =
			static_cast<const NeighbourIteratorCone &>(other);
	// speed optimization: only compare mutable values
	// this means that iterators with different radii / bases could be equal...
	// don't compare sliceIdx here, as it directly depends on idx anyways
	// this way, the calculation of end() is much easier
	// ( O(1) vs O(slices) with distance calculations... )
	return this->idx == o.idx;
			//&& this->base == o.base && this->radiusSquared == o.radiusSquared;
}

void SOMCone::CacheCone::preload(const multi_img &image)
{
	for (int y = 0; y < image.height; y++)
	{
		for (int x = 0; x < image.width; x++)
		{
			SOM::iterator iter = som->identifyWinnerNeuron(image(y, x));
			IteratorCone *it = static_cast<IteratorCone *>(iter.getBase());
			data[y][x] = it->getIdx();
		}
	}
	preloaded = true;
}

double SOMCone::CacheCone::getDistance(const multi_img::Pixel &v1,
									   const multi_img::Pixel &v2,
									   const cv::Point &c1,
									   const cv::Point &c2)
{
	// reference! important for storing the value in the cache, not in a local var!
	int &idx1 = data[c1.y][c1.x];
	int &idx2 = data[c2.y][c2.x];

	if (!preloaded)
	{
		if (idx1 == -1)
		{
			SOM::iterator iter = som->identifyWinnerNeuron(v1);
			IteratorCone *it = static_cast<IteratorCone *>(iter.getBase());
			idx1 = it->getIdx();
		}

		if (idx2 == -1)
		{
			SOM::iterator iter = som->identifyWinnerNeuron(v2);
			IteratorCone *it = static_cast<IteratorCone *>(iter.getBase());
			idx2 = it->getIdx();
		}
	}

	return som->getDistance(idx1, idx2);
}

double SOMCone::CacheCone::getSobelX(int x, int y)
{
	if (!preloaded)
	{
		assert(preloaded); // fail
		return 0.0;        // in case that assertions are disabled
	}

	cv::Point3d u = (coords(y-1, x-1) + 2*coords(y-1, x) + coords(y-1, x+1)) * 0.25;
	cv::Point3d d = (coords(y+1, x-1) + 2*coords(y+1, x) + coords(y+1, x+1)) * 0.25;

	double valy = som->getDistance(u, d);

	if (u.dot(u) < d.dot(d)) // TODO: hier war bisher kein .z drin. war das absicht?
		valy = -valy;

	return valy;
}

double SOMCone::CacheCone::getSobelY(int x, int y)
{
	if (!preloaded)
	{
		assert(preloaded); // fail
		return 0.0;        // in case that assertions are disabled
	}

	cv::Point3d u = (coords(y-1, x-1) + 2*coords(y, x-1) + coords(y+1, x-1)) * 0.25;
	cv::Point3d d = (coords(y-1, x+1) + 2*coords(y, x+1) + coords(y+1, x+1)) * 0.25;

	double valx = som->getDistance(u, d);

	if (u.dot(u) < d.dot(d)) // TODO: hier war bisher kein .z drin. war das absicht?
		valx = -valx;

	return valx;
}
