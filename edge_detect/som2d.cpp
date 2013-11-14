#include "som2d.h"

#include <sstream>

SOM2d::SOM2d(const vole::EdgeDetectionConfig &conf, int dimension,
			 std::vector<multi_img_base::BandDesc> meta)
	: SOM(conf, dimension, meta), width(conf.sidelength),
	  height(conf.type == vole::SOM_SQUARE ? conf.sidelength : 1), // case 1d SOM
	  neurons(Field(conf.sidelength, Row(conf.sidelength, Neuron(dimension))))
{
	/// Uniformly randomizes each neuron
	// TODO: given interval [0..1] sure? purpose? it will not fit anyway
	cv::RNG rng(config.seed);

	theEnd = SOM::iterator(new Iterator2d(this, 0, height));

	for (SOM::iterator n = begin(); n != end(); ++n) {
		(*n).randomize(rng, 0., 1.);
	}
}

SOM2d::SOM2d(const vole::EdgeDetectionConfig &conf, const multi_img &data,
			 std::vector<multi_img_base::BandDesc> meta)
	: SOM(conf, data.size(), meta), width(conf.sidelength),
	  height(conf.type == vole::SOM_SQUARE ? conf.sidelength : 1), // case 1d SOM
	  neurons(Field(conf.sidelength, Row(conf.sidelength, Neuron(data.size()))))
{
	// check format
	if (data.width != width || data.height != height) {
		std::cerr << "SOM image has wrong dimensions!" << std::endl;
		assert(false);
		return; // somdata will be empty
	}

	theEnd = SOM::iterator(new Iterator2d(this, 0, height));

	/// Read SOM from multi_img
	for (SOM::iterator n = begin(); n != end(); ++n) {
		cv::Point pos = n.get2dCoordinates();
		*n = data(pos.y, pos.x);
	}
}

SOM2d::Cache2d *SOM2d::createCache(int img_height, int img_width)
{
	return new Cache2d(this, img_height, img_width);
}

// no virtual function call overhead inside the loop(s)
SOM::iterator SOM2d::identifyWinnerNeuron(const multi_img::Pixel &inputVec)
{
	// initialize with maximum / invalid value
	double closestDistance = std::numeric_limits<double>::max();
	cv::Point winner(-1, -1);

	// find closest Neuron to inputVec in the SOM
	// -> iterate over all neurons in grid
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			double dist = getSimilarity(neurons[y][x], inputVec);

			// compare current distance with minimal found distance
			if (dist < closestDistance) {
				// set new minimal distance and winner position
				closestDistance = dist;
				winner = cv::Point(x, y);
			}
		}
	}
	assert(closestDistance < std::numeric_limits<double>::max());
	return SOM::iterator(new Iterator2d(this, winner.x, winner.y));
}

int SOM2d::updateNeighborhood(SOM::iterator &neuron,
							   const multi_img::Pixel &input,
							   double sigma, double learnRate)
{
	// Get position of winner neuron in the 2d grid
	Iterator2d *it = static_cast<Iterator2d *>(neuron.getBase());
	cv::Point pos = it->getId();
	double sigmaSquare = sigma * sigma;

	int updates = 0;

	// update winner neuron. only one neuron has distance = 0
	{
		// dist = 0  ->  exp(0) = 1
		//double dist = getDistanceSquared(pos, pos + cv::Point(0, 0));
		//double fakeGaussian = exp(-(dist)/(2.0*sigmaSquare));
		double weight = learnRate * 1; // * fakeGaussian;
		if (weight >= 0.01) {
			neurons[pos.y + 0][pos.x + 0].update(input, weight);
			++updates;
		}
	}

	int maxDist; // maximum distance of updates _along one axis_

	// update neurons on horizontal and vertical axis with equal distance
	// (mirror over the axis that is non-zero, swap the variables and mirror again => 2*2=4 updates)
	{
		int i;
		for (i = 1;; i++)
		{
			bool posX = pos.x + i < width;
			bool negX = pos.x - i >= 0;
			bool posY = pos.y + i < height;
			bool negY = pos.y - i >= 0;
			if ( !(posX | negX | posY | negY) ) break; // we're done already

			double dist = getDistanceSquared(pos, pos + cv::Point(i, 0));
			double fakeGaussian = exp(-(dist)/(2.0*sigmaSquare));
			double weight = learnRate * fakeGaussian;
			if (weight < 0.01) break;

			// x axis
			if (posX) { ++updates; neurons[pos.y + 0][pos.x + i].update(input, weight); }
			if (negX) { ++updates; neurons[pos.y + 0][pos.x - i].update(input, weight); }
			// y axis
			if (negY) { ++updates; neurons[pos.y - i][pos.x + 0].update(input, weight); }
			if (posY) { ++updates; neurons[pos.y + i][pos.x + 0].update(input, weight); }
		}
		maxDist = i;
	}

	// update neurons on diagonal directions with equal distance
	// (mirror over all axis, swap has no effect here => 2*2=4 updates per weight)
	{
		for (int i = 1; i < maxDist; i++)
		{
			bool posX = pos.x + i < width;
			bool negX = pos.x - i >= 0;
			bool posY = pos.y + i < height;
			bool negY = pos.y - i >= 0;
			if (!((posX | negX) & (posY | negY))) break; // we're done already

			double dist = getDistanceSquared(pos, pos + cv::Point(i, i));
			double fakeGaussian = exp(-(dist)/(2.0*sigmaSquare));
			double weight = learnRate * fakeGaussian;
			if (weight < 0.01) break;

			if (posY)
			{
				if (posX) { ++updates; neurons[pos.y + i][pos.x + i].update(input, weight); } //  first quadrant
				if (negX) { ++updates; neurons[pos.y + i][pos.x - i].update(input, weight); } // second quadrant
			}
			if (negY)
			{
				if (negX) { ++updates; neurons[pos.y - i][pos.x - i].update(input, weight); } //  third quadrant
				if (posX) { ++updates; neurons[pos.y - i][pos.x + i].update(input, weight); } // fourth quadrant
			}
		}
	}

	// update remaining neurons
	// (mirror over all axis and swap => 2*2*2=8 updates per weight)
	// x is always greater then y. so if (y,y) was outside the L2-circle,
	// (x,y) will be outside even further
	for (int y = 1; y < maxDist; y++)
	{
		for (int x = y + 1; x < maxDist; x++)
		{
			bool posXX = pos.x + x < width;
			bool negXX = pos.x - x >= 0;
			bool posXY = pos.x + y < width;
			bool negXY = pos.x - y >= 0;
			bool posYX = pos.y + x < height;
			bool negYX = pos.y - x >= 0;
			bool posYY = pos.y + y < height;
			bool negYY = pos.y - y >= 0;

			if ( !(
					((posYY | negYY) & (posXX | negXX)) // one of the first  four is updated
				 || ((posYX | negYX) & (posXY | negXY)) // one of the second four is updated
				 ) // nothing is updated
			   ) break;

			double dist = getDistanceSquared(pos, pos + cv::Point(x, y));
			double fakeGaussian = exp(-(dist)/(2.0*sigmaSquare));
			double weight = learnRate * fakeGaussian;
			if (weight < 0.01) break;

			if (posYY && posXX) { ++updates; neurons[pos.y + y][pos.x + x].update(input, weight); } //  first quadrant
			if (posYY && negXX) { ++updates; neurons[pos.y + y][pos.x - x].update(input, weight); } // second quadrant
			if (negYY && negXX) { ++updates; neurons[pos.y - y][pos.x - x].update(input, weight); } //  third quadrant
			if (negYY && posXX) { ++updates; neurons[pos.y - y][pos.x + x].update(input, weight); } // fourth quadrant
			// the swap of x and y mirrors over the diagonal of the quadrant
			if (posYX && posXY) { ++updates; neurons[pos.y + x][pos.x + y].update(input, weight); } //  first quadrant
			if (posYX && negXY) { ++updates; neurons[pos.y + x][pos.x - y].update(input, weight); } // second quadrant
			if (negYX && negXY) { ++updates; neurons[pos.y - x][pos.x - y].update(input, weight); } //  third quadrant
			if (negYX && posXY) { ++updates; neurons[pos.y - x][pos.x + y].update(input, weight); } // fourth quadrant
		}
	}

	return updates;
}

double SOM2d::getDistanceBetweenWinners(const multi_img::Pixel &v1,
										const multi_img::Pixel &v2)
{
	// This is a SOM2d, so identifyWinnerNeuron returns Iterator2ds
	// Is it better to override identifyWinnerNeuron in the subclasses?
	// unneccessary code duplication, but would remove the somewhat dangerous cast
	SOM::iterator iter1 = identifyWinnerNeuron(v1);
	Iterator2d *p1 = static_cast<Iterator2d *>(iter1.getBase());
	SOM::iterator iter2 = identifyWinnerNeuron(v2);
	Iterator2d *p2 = static_cast<Iterator2d *>(iter2.getBase());

	return getDistance(p1->getId(), p2->getId());
}

cv::Vec3f SOM2d::getColor(cv::Point3d pos)
{
	cv::Vec3f pixel;
	if (height > 1)
	{
		// don't use blue channel, because r and g should be more differentiable
		pixel[0] = (float)(pos.z); // should always be 0
		pixel[1] = (float)(pos.y);
		pixel[2] = (float)(pos.x);
	}
	else
	{
		pixel[0] = (float)(pos.x);
		pixel[1] = (float)(pos.x);
		pixel[2] = (float)(pos.x);
	}

	// normalize color by sidelength
	// use temporary to workaround OpenCV 2.3 bug
	const double div = 1.0 / config.sidelength;
	return pixel * div;
}

std::string SOM2d::description()
{
	std::stringstream s;
	if (height == 1)
		s << "SOM of type line, size " << width;
	else
		s << "SOM of type square, size " << width << "x" << height;
	s << ", with " << size() << " neurons of dimension " << dim;
	s << ", seed=" << config.seed;
	return s.str();
}

SOM::iterator SOM2d::begin()
{
	return SOM::iterator(new Iterator2d(this, 0, 0));
}

double SOM2d::getDistance(const cv::Point &p1, const cv::Point &p2)
{
	cv::Point d = p1 - p2;
	return std::sqrt(d.ddot(d)); // TODO: IEEE inv-sqrt approximation should be enough...
}

double SOM2d::getDistance(const cv::Point2d &p1, const cv::Point2d &p2)
{
	cv::Point2d d = p1 - p2;
	return std::sqrt(d.dot(d)); // TODO: IEEE inv-sqrt approximation should be enough...
}

double SOM2d::getDistanceSquared(const cv::Point &p1, const cv::Point &p2)
{
	cv::Point d = p1 - p2;
	return (d.ddot(d));
}

double SOM2d::getDistanceSquared(const cv::Point2d &p1, const cv::Point2d &p2)
{
	cv::Point2d d = p1 - p2;
	return (d.dot(d));
}

void SOM2d::Iterator2d::operator++()
{
	++x;
	y += x / base->width;
	x %= base->width;
}

bool SOM2d::Iterator2d::equal(const SOM::IteratorBase &other) const
{
	// Types were checked in the super class
	const Iterator2d &o = static_cast<const Iterator2d &>(other);
	return this->base == o.base && this->x == o.x && this->y == o.y;
}

SOM::neighbourIterator SOM2d::Iterator2d::neighboursBegin(double radius)
{
	assert(radius >= 0);
	return SOM::neighbourIterator(
				new NeighbourIterator2d(base, getId(), radius));
}

SOM::neighbourIterator SOM2d::Iterator2d::neighboursEnd(double radius)
{
	return SOM::neighbourIterator(
				new NeighbourIterator2d(base, getId(), radius, true));
}

SOM2d::NeighbourIterator2d::NeighbourIterator2d(SOM2d *base,
							 cv::Point neuron,
							 double radius,
							 bool end)
	: SOM::NeighbourIteratorBase(), base(base),
	  neuron(neuron), radiusSquared(
		radius > 0 ? radius * radius : std::numeric_limits<double>::infinity()),
	  fromX(std::max(neuron.x - (int)radius, 0)),
	  toX  (std::min(neuron.x + (int)radius + 1, base->width )), // +1 exclusive
	  toY  (std::min(neuron.y + (int)radius + 1, base->height))
{
	x = fromX;
	y = end ? toY : std::max(neuron.y - (int)radius, 0); // "fromY"
}

void SOM2d::NeighbourIterator2d::operator++()
{
	// iterate over pixels within |neuron + other|^2 <= radius, cache aligned
	do
	{
		++x;
		if (x == toX)
		{
			x = fromX;
			//toX = -1;
			++y;
		}
	}
	while (getDistanceSquared() > radiusSquared && y < toY);
	// this skips the pixels on the right (positive x direction) side outside the circle
	//if (toX == -1) toX = (neuron.x << 1) - x; // neuron.x + (neuron.x - x)
	// with a cache of length height, one could save the whole lower half of
	// outside pixels because of the symmetry - not sure if it's worth it, though
}

double SOM2d::NeighbourIterator2d::getDistance() const
{
	double dx = (double)(neuron.x - x);
	double dy = (double)(neuron.y - y);
	return std::sqrt(dx * dx + dy * dy); // TODO: IEEE inv-sqrt approximation should be enough...
}

double SOM2d::NeighbourIterator2d::getDistanceSquared() const
{
	double dx = (double)(neuron.x - x);
	double dy = (double)(neuron.y - y);
	return (dx * dx + dy * dy);
}

double SOM2d::NeighbourIterator2d::getFakeGaussianWeight(double sigma) const
{
	double dist = getDistanceSquared();
	return exp(-(dist)/(2.0*sigma*sigma));
}

bool SOM2d::NeighbourIterator2d::equal(const NeighbourIteratorBase &other) const
{
	// Types were checked in the super class
	const NeighbourIterator2d &o =
			static_cast<const NeighbourIterator2d &>(other);
	// speed optimization: only compare mutable values
	// this means that iterators with different radii / bases could be equal...
	return this->x == o.x && this->y == o.y;
			//&& this->base == o.base && this->radiusSquared == o.radiusSquared;
}

void SOM2d::Cache2d::PreloadTBB::operator()(const tbb::blocked_range<int>& r) const
{
	for (int i = r.begin(); i != r.end(); ++i)
	{
		int x = i % image.width;
		int y = i / image.width;

		SOM::iterator iter = som->identifyWinnerNeuron(image(y, x));
		SOM2d::Iterator2d *it = static_cast<SOM2d::Iterator2d *>(iter.getBase());
		data[y][x] = it->getId();
	}
}

void SOM2d::Cache2d::preload(const multi_img &image)
{
	tbb::parallel_for(tbb::blocked_range<int>(0, image.height*image.width),
	                  PreloadTBB(image, som, data));
	
	preloaded = true;
}

double SOM2d::Cache2d::getDistance(const multi_img::Pixel &v1,
								   const multi_img::Pixel &v2,
								   const cv::Point &c1,
								   const cv::Point &c2)
{
	cv::Point &p1 = data[c1.y][c1.x];
	cv::Point &p2 = data[c2.y][c2.x];

	if (!preloaded)
	{
		if (p1 == cv::Point(-1, -1))
		{
			SOM::iterator iter = som->identifyWinnerNeuron(v1);
			Iterator2d *it = static_cast<Iterator2d *>(iter.getBase());
			p1 = it->getId();
		}

		if (p2 == cv::Point(-1, -1))
		{
			SOM::iterator iter = som->identifyWinnerNeuron(v2);
			Iterator2d *it = static_cast<Iterator2d *>(iter.getBase());
			p2 = it->getId();
		}
	}

	return som->getDistance(p1, p2);
}

SOM::iterator SOM2d::Cache2d::getWinnerNeuron(cv::Point c)
{
	if (!preloaded)
	{
		assert(preloaded);
		return som->end();
	}

	const cv::Point &p = data[c.y][c.x];
	return SOM::iterator(new Iterator2d(som, p.x, p.y));
}

double SOM2d::Cache2d::getSimilarity(vole::SimilarityMeasure<multi_img::Value> *distfun,
                                     const cv::Point &c1,
                                     const cv::Point &c2)
{
	if (!preloaded)
	{
		assert(preloaded);
		return 0.0; // in case that assertions are disabled
	}

	// "transform" image coordinates to neuron coordinates
	const cv::Point &p1 = data[c1.y][c1.x];
	const cv::Point &p2 = data[c2.y][c2.x];

	// "transform" neuron coordinates to neuron values
	const Neuron &n1 = som->neurons[p1.y][p1.x];
	const Neuron &n2 = som->neurons[p2.y][p2.x];

	return distfun->getSimilarity(n1, n2);
}

double SOM2d::Cache2d::getSobelX(int x, int y)
{
	if (!preloaded)
	{
		assert(preloaded); // fail
		return 0.0;        // in case that assertions are disabled
	}

	cv::Point2d u = (data[y-1][x-1] + 2*data[y-1][x] + data[y-1][x+1]) * 0.25;
	cv::Point2d d = (data[y+1][x-1] + 2*data[y+1][x] + data[y+1][x+1]) * 0.25;

	double valy = som->getDistance(u, d);

	if (u.dot(u) < d.dot(d))
		valy = -valy;

	return valy;
}

double SOM2d::Cache2d::getSobelY(int x, int y)
{
	if (!preloaded)
	{
		assert(preloaded); // fail
		return 0.0;        // in case that assertions are disabled
	}

	cv::Point2d u = (data[y-1][x-1] + 2*data[y][x-1] + data[y+1][x-1]) * 0.25;
	cv::Point2d d = (data[y-1][x+1] + 2*data[y][x+1] + data[y+1][x+1]) * 0.25;

	double valx = som->getDistance(u, d);

	if (u.dot(u) < d.dot(d))
		valx = -valx;

	return valx;
}
