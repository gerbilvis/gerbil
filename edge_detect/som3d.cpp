#include "som3d.h"

#include <sstream>

SOM3d::SOM3d(const vole::EdgeDetectionConfig &conf, int dimension,
			 std::vector<multi_img_base::BandDesc> meta)
	: SOM(conf, dimension, meta), width(conf.sidelength), height(conf.sidelength),
	  depth(conf.sidelength), neurons(Cube(conf.sidelength,
										   Field(conf.sidelength,
												 Row(conf.sidelength,
													 Neuron(dimension)))))
{
	/// Uniformly randomizes each neuron
	// TODO: given interval [0..1] sure? purpose? it will not fit anyway
	cv::RNG rng(config.seed);

	theEnd = SOM::iterator(new Iterator3d(this, 0, 0, depth));

	for (SOM::iterator n = begin(); n != end(); ++n) {
		(*n).randomize(rng, 0., 1.);
	}
}

SOM3d::SOM3d(const vole::EdgeDetectionConfig &conf, const multi_img &data,
			 std::vector<multi_img_base::BandDesc> meta)
	: SOM(conf, data.size(), meta), width(conf.sidelength), height(conf.sidelength),
	  depth(conf.sidelength), neurons(Cube(conf.sidelength,
										   Field(conf.sidelength,
												 Row(conf.sidelength,
													 Neuron(data.size())))))
{
	// check format
	if (data.width != width * height || data.height != depth) {
		std::cerr << "SOM image has wrong dimensions!" << std::endl;
		assert(false);
		return; // somdata will be empty
	}

	theEnd = SOM::iterator(new Iterator3d(this, 0, 0, depth));

	/// Read SOM from multi_img
	for (SOM::iterator n = begin(); n != end(); ++n) {
		cv::Point pos = n.get2dCoordinates();
		*n = data(pos.y, pos.x);
	}
}

SOM3d::Cache3d *SOM3d::createCache(int img_height, int img_width)
{
	return new Cache3d(this, img_height, img_width);
}

// no virtual function call overhead inside the loop(s)
SOM::iterator SOM3d::identifyWinnerNeuron(const multi_img::Pixel &inputVec)
{
	// initialize with maximum / invalid value
	double closestDistance = std::numeric_limits<double>::max();
	cv::Point3i winner(-1, -1, -1);

	// find closest Neuron to inputVec in the SOM
	// -> iterate over all neurons in grid
	for (int z = 0; z < depth; ++z) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				double dist = getSimilarity(neurons[z][y][x], inputVec);

				// compare current distance with minimal found distance
				if (dist < closestDistance) {
					// set new minimal distance and winner position
					closestDistance = dist;
					winner = cv::Point3i(x, y, z);
				}
			}
		}
	}
	assert(closestDistance < std::numeric_limits<double>::max());
	return SOM::iterator(new Iterator3d(this, winner.x, winner.y, winner.z));
}

// this is basically a copy of the 2d version, just shifted by +-deltaZ
template<bool mirrorZ>
int SOM3d::updateSquare(const multi_img::Pixel &input,
						 const cv::Point3i &pos,
						 int deltaZ,
						 double sigmaSquare,
						 double learnRate)
{
	int updates = 0;
	// update winner neuron (except deltaZ == 0, dist is greater than 0)
	{
		double dist = getDistanceSquared(pos, pos + cv::Point3i(0, 0, deltaZ));
		double fakeGaussian = exp(-(dist)/(2.0*sigmaSquare));
		double weight = learnRate * fakeGaussian;

		// this point (no delta in x and y direction) is the closest point
		// to the winner neuron. if this one is not updated, no point will
		// be updated at all and we are finished with all updates
		if (weight < 0.01)
			return updates;

		// if dZ is zero, only one iteration with dZ == 0
		// else two iterations, with values dZ and -dZ
		int dZ = deltaZ;
		int iterations = mirrorZ ? 2 : 1;
		for (int i = 0; i < iterations; i++)
		{
			if (pos.z + dZ >= 0 && pos.z + dZ < depth)
			{
				neurons[pos.z + dZ][pos.y + 0][pos.x + 0].update(input, weight);
				++updates;
			}
			dZ = -dZ; // swap sign
		}
	}

	int maxDist;

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

			double dist = getDistanceSquared(pos, pos + cv::Point3i(i, 0, deltaZ));
			double fakeGaussian = exp(-(dist)/(2.0*sigmaSquare));
			double weight = learnRate * fakeGaussian;
			if (weight < 0.01) break;

			// if dZ is zero, only one iteration with dZ == 0
			// else two iterations, with values dZ and -dZ
			int dZ = deltaZ;
			int iterations = mirrorZ ? 2 : 1;
			for (int j = 0; j < iterations; j++)
			{
				if (pos.z + dZ >= 0 && pos.z + dZ < depth)
				{
					// x axis
					if (posX) { ++updates; neurons[pos.z + dZ][pos.y + 0][pos.x + i].update(input, weight); }
					if (negX) { ++updates; neurons[pos.z + dZ][pos.y + 0][pos.x - i].update(input, weight); }
					// y axis
					if (negY) { ++updates; neurons[pos.z + dZ][pos.y - i][pos.x + 0].update(input, weight); }
					if (posY) { ++updates; neurons[pos.z + dZ][pos.y + i][pos.x + 0].update(input, weight); }
				}
				dZ = -dZ; // swap sign
			}
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

			double dist = getDistanceSquared(pos, pos + cv::Point3i(i, i, deltaZ));
			double fakeGaussian = exp(-(dist)/(2.0*sigmaSquare));
			double weight = learnRate * fakeGaussian;
			if (weight < 0.01) break;

			// if dZ is zero, only one iteration with dZ == 0
			// else two iterations, with values dZ and -dZ
			int dZ = deltaZ;
			int iterations = mirrorZ ? 2 : 1;
			for (int j = 0; j < iterations; j++)
			{
				if (pos.z + dZ >= 0 && pos.z + dZ < depth)
				{
					if (posY)
					{
						if (posX) { ++updates; neurons[pos.z + dZ][pos.y + i][pos.x + i].update(input, weight); } //  first quadrant
						if (negX) { ++updates; neurons[pos.z + dZ][pos.y + i][pos.x - i].update(input, weight); } // second quadrant
					}
					if (negY)
					{
						if (negX) { ++updates; neurons[pos.z + dZ][pos.y - i][pos.x - i].update(input, weight); } //  third quadrant
						if (posX) { ++updates; neurons[pos.z + dZ][pos.y - i][pos.x + i].update(input, weight); } // fourth quadrant
					}
				}
				dZ = -dZ; // swap sign
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

			double dist = getDistanceSquared(pos, pos + cv::Point3i(x, y, deltaZ));
			double fakeGaussian = exp(-(dist)/(2.0*sigmaSquare));
			double weight = learnRate * fakeGaussian;
			if (weight < 0.01) break;

			// if dZ is zero, only one iteration with dZ == 0
			// else two iterations, with values dZ and -dZ
			int dZ = deltaZ;
			int iterations = mirrorZ ? 2 : 1;
			for (int j = 0; j < iterations; j++)
			{
				if (pos.z + dZ >= 0 && pos.z + dZ < depth)
				{
					if (posYY && posXX) { ++updates; neurons[pos.z + dZ][pos.y + y][pos.x + x].update(input, weight); } //  first quadrant
					if (posYY && negXX) { ++updates; neurons[pos.z + dZ][pos.y + y][pos.x - x].update(input, weight); } // second quadrant
					if (negYY && negXX) { ++updates; neurons[pos.z + dZ][pos.y - y][pos.x - x].update(input, weight); } //  third quadrant
					if (negYY && posXX) { ++updates; neurons[pos.z + dZ][pos.y - y][pos.x + x].update(input, weight); } // fourth quadrant
					// the swap of x and y mirrors over the diagonal of the quadrant
					if (posYX && posXY) { ++updates; neurons[pos.z + dZ][pos.y + x][pos.x + y].update(input, weight); } //  first quadrant
					if (posYX && negXY) { ++updates; neurons[pos.z + dZ][pos.y + x][pos.x - y].update(input, weight); } // second quadrant
					if (negYX && negXY) { ++updates; neurons[pos.z + dZ][pos.y - x][pos.x - y].update(input, weight); } //  third quadrant
					if (negYX && posXY) { ++updates; neurons[pos.z + dZ][pos.y - x][pos.x + y].update(input, weight); } // fourth quadrant
				}
				dZ = -dZ; // swap sign
			}
		}
	}

	return updates;
}

int SOM3d::updateNeighborhood(SOM::iterator &neuron,
							   const multi_img::Pixel &input,
							   double sigma, double learnRate)
{
	// Get position of winner neuron in the 3d grid
	Iterator3d *it = static_cast<Iterator3d *>(neuron.getBase());
	cv::Point3i pos = it->getId();

	int updates, totalUpdates;

	// if deltaZ == 0, we only update one plane of the cube
	totalUpdates = updates = updateSquare<false>(input, pos, 0, sigma * sigma, learnRate);

	for (int deltaZ = 1; updates > 0; ++deltaZ)
	{
		// make sure we get updates in at least one z direction
		if (pos.z - deltaZ < 0 && pos.z + deltaZ >= depth) break;

		updates = updateSquare<true>(input, pos, deltaZ, sigma * sigma, learnRate);
		totalUpdates += updates;
	}
	return totalUpdates;
}

double SOM3d::getDistanceBetweenWinners(const multi_img::Pixel &v1,
										const multi_img::Pixel &v2)
{
	// This is a SOM3d, so identifyWinnerNeuron returns Iterator3ds
	// Is it better to override identifyWinnerNeuron in the subclasses?
	// unneccessary code duplication, but would remove the somewhat dangerous cast
	SOM::iterator iter1 = identifyWinnerNeuron(v1);
	Iterator3d *p1 = static_cast<Iterator3d *>(iter1.getBase());
	SOM::iterator iter2 = identifyWinnerNeuron(v2);
	Iterator3d *p2 = static_cast<Iterator3d *>(iter2.getBase());

	return getDistance(p1->getId(), p2->getId());
}

cv::Vec3f SOM3d::getColor(cv::Point3d pos)
{
	cv::Vec3f pixel;
	pixel[0] = (float)(pos.x);
	pixel[1] = (float)(pos.y);
	pixel[2] = (float)(pos.z);

	// normalize color by sidelength
	// use temporary to workaround OpenCV 2.3 bug
	const double div = 1.0 / config.sidelength;
	return pixel * div;
}

std::string SOM3d::description()
{
	std::stringstream s;
	s << "SOM of type cube, size " << width << "x" << height << "x" << depth;
	s << ", with " << size() << " neurons of dimension " << dim;
	s << ", seed=" << config.seed;
	return s.str();
}

SOM::iterator SOM3d::begin()
{
	return SOM::iterator(new Iterator3d(this, 0, 0, 0));
}

double SOM3d::getDistance(const cv::Point3i &p1, const cv::Point3i &p2)
{
	cv::Point3i d = p1 - p2;
	return std::sqrt(d.ddot(d)); // TODO: IEEE inv-sqrt approximation should be enough...
}

double SOM3d::getDistance(const cv::Point3d &p1, const cv::Point3d &p2)
{
	cv::Point3d d = p1 - p2;
	return std::sqrt(d.dot(d)); // TODO: IEEE inv-sqrt approximation should be enough...
}

double SOM3d::getDistanceSquared(const cv::Point3i &p1, const cv::Point3i &p2)
{
	cv::Point3i d = p1 - p2;
	return (d.ddot(d));
}

double SOM3d::getDistanceSquared(const cv::Point3d &p1, const cv::Point3d &p2)
{
	cv::Point3d d = p1 - p2;
	return (d.dot(d));
}

void SOM3d::Iterator3d::operator++()
{
	++x;
	y += x / base->width;
	z += y / base->height;
	y %= base->height;
	x %= base->width;
}

bool SOM3d::Iterator3d::equal(const SOM::IteratorBase &other) const
{
	// Types were checked in the super class
	const Iterator3d &o = static_cast<const Iterator3d &>(other);
	return this->base == o.base
			&& this->x == o.x && this->y == o.y && this->z == o.z;
}

SOM::neighbourIterator SOM3d::Iterator3d::neighboursBegin(double radius)
{
	return SOM::neighbourIterator(new NeighbourIterator3d(
									  base, getId(), radius));
}

SOM::neighbourIterator SOM3d::Iterator3d::neighboursEnd(double radius)
{
	return SOM::neighbourIterator(new NeighbourIterator3d(
									  base, getId(), radius, true));
}

SOM3d::NeighbourIterator3d::NeighbourIterator3d(
		SOM3d *base, cv::Point3i neuron, double radius, bool end)
	: SOM::NeighbourIteratorBase(), neuron(neuron), base(base),
	  radiusSquared(
		radius > 0 ? radius * radius : std::numeric_limits<double>::infinity()),
	  fromX(std::max(neuron.x - (int)radius, 0)),
	  toX  (std::min(neuron.x + (int)radius + 1, base->width )), // +1 exclusive
	  fromY(std::max(neuron.y - (int)radius, 0)),
	  toY  (std::min(neuron.y + (int)radius + 1, base->height)),
	  toZ  (std::min(neuron.z + (int)radius + 1, base->depth ))
{
	x = fromX;
	y = fromY;
	z = end ? toZ : std::max(neuron.z - (int)radius, 0); // "fromZ"
}

void SOM3d::NeighbourIterator3d::operator++()
{
	// iterate over pixels within |neuron + other|^2 <= radius, cache aligned
	do
	{
		++x;
		if (x == toX)
		{
			x = fromX;
			++y;
			if (y == toY)
			{
				y = fromY;
				++z;
			}
		}
	}
	while (getDistanceSquared() > radiusSquared && z < toZ);
}

double SOM3d::NeighbourIterator3d::getDistance() const
{
	double dx = (double)(neuron.x - x);
	double dy = (double)(neuron.y - y);
	double dz = (double)(neuron.z - z);
	return std::sqrt(dx * dx + dy * dy + dz * dz); // TODO: IEEE inv-sqrt approximation should be enough...
}

double SOM3d::NeighbourIterator3d::getDistanceSquared() const
{
	double dx = (double)(neuron.x - x);
	double dy = (double)(neuron.y - y);
	double dz = (double)(neuron.z - z);
	return (dx * dx + dy * dy + dz * dz);
}

double SOM3d::NeighbourIterator3d::getFakeGaussianWeight(double sigma) const
{
	double dist = getDistanceSquared();
	return exp(-(dist)/(2.0*sigma*sigma));
}

bool SOM3d::NeighbourIterator3d::equal(const NeighbourIteratorBase &other) const
{
	// Types were checked in the super class
	const NeighbourIterator3d &o =
			static_cast<const NeighbourIterator3d &>(other);
	// speed optimization: only compare mutable values
	// this means that iterators with different radii / bases could be equal...
	return this->x == o.x && this->y == o.y && this->z == o.z;
			//&& this->base == o.base && this->radiusSquared == o.radiusSquared;
}

void SOM3d::Cache3d::PreloadTBB::operator()(const tbb::blocked_range<int>& r) const
{
	for (int i = r.begin(); i != r.end(); ++i)
	{
		int x = i % image.width;
		int y = i / image.width;

		SOM::iterator iter = som->identifyWinnerNeuron(image(y, x));
		SOM3d::Iterator3d *it = static_cast<SOM3d::Iterator3d *>(iter.getBase());
		data[y][x] = it->getId();
	}
}

void SOM3d::Cache3d::preload(const multi_img &image)
{
        tbb::parallel_for(tbb::blocked_range<int>(0, image.height*image.width),
                          PreloadTBB(image, som, data));

        preloaded = true;
}

double SOM3d::Cache3d::getDistance(const multi_img::Pixel &v1,
                                   const multi_img::Pixel &v2,
                                   const cv::Point &c1,
                                   const cv::Point &c2)
{
	cv::Point3i &p1 = data[c1.y][c1.x];
	cv::Point3i &p2 = data[c2.y][c2.x];

	if (!preloaded)
	{
		if (p1 == cv::Point3i(-1, -1, -1))
		{
			SOM::iterator iter = som->identifyWinnerNeuron(v1);
			Iterator3d *it = static_cast<Iterator3d *>(iter.getBase());
			p1 = it->getId();
		}

		if (p2 == cv::Point3i(-1, -1, -1))
		{
			SOM::iterator iter = som->identifyWinnerNeuron(v2);
			Iterator3d *it = static_cast<Iterator3d *>(iter.getBase());
			p2 = it->getId();
		}
	}

	return som->getDistance(p1, p2);
}

SOM::iterator SOM3d::Cache3d::getWinnerNeuron(cv::Point c)
{
	if (!preloaded)
	{
		assert(preloaded);
		return som->end();
	}

	const cv::Point3i &p = data[c.y][c.x];
	return SOM::iterator(new Iterator3d(som, p.x, p.y, p.z));
}

double SOM3d::Cache3d::getSimilarity(vole::SimilarityMeasure<multi_img::Value> *distfun,
                                     const cv::Point &c1,
                                     const cv::Point &c2)
{
	if (!preloaded)
	{
		assert(preloaded);
		return 0.0; // in case that assertions are disabled
	}

	const cv::Point3i &p1 = data[c1.y][c1.x];
	const cv::Point3i &p2 = data[c2.y][c2.x];

	const Neuron &n1 = som->neurons[p1.z][p1.y][p1.x];
	const Neuron &n2 = som->neurons[p2.z][p2.y][p2.x];

	return distfun->getSimilarity(n1, n2);
}

double SOM3d::Cache3d::getSobelX(int x, int y)
{
	if (!preloaded)
	{
		assert(preloaded); // fail
		return 0.0;        // in case that assertions are disabled
	}

	cv::Point3d u = (data[y-1][x-1] + 2*data[y-1][x] + data[y-1][x+1]) * 0.25;
	cv::Point3d d = (data[y+1][x-1] + 2*data[y+1][x] + data[y+1][x+1]) * 0.25;

	double valy = som->getDistance(u, d);

	if (u.dot(u) < d.dot(d))
		valy = -valy;

	return valy;
}

double SOM3d::Cache3d::getSobelY(int x, int y)
{
	if (!preloaded)
	{
		assert(preloaded); // fail
		return 0.0;        // in case that assertions are disabled
	}

	cv::Point3d u = (data[y-1][x-1] + 2*data[y][x-1] + data[y+1][x-1]) * 0.25;
	cv::Point3d d = (data[y-1][x+1] + 2*data[y][x+1] + data[y+1][x+1]) * 0.25;

	double valx = som->getDistance(u, d);

	if (u.dot(u) < d.dot(d))
		valx = -valx;

	return valx;
}
