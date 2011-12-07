#include "self_organizing_map.h"
#include <sm_factory.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm> 

SOM::SOM(const vole::EdgeDetectionConfig &conf, int dimension)
	: dim(dimension), width(conf.som_width), height(conf.som_height),
	  config(conf),
      neurons(Field(height, Row(width, Neuron(dim))))
{
	/// Create similarity measure
	distfun = vole::SMFactory<multi_img::Value>::spawn(config.similarity);
	assert(distfun);

	/// Uniformly randomizes each neuron
	// TODO: given interval [0..1] sure? purpose? it will not fit anyway
	cv::RNG rng;
	if (config.fixedSeed)	// TODO: get seed from outside
		rng = cv::RNG (19.0);
	else
		rng = cv::RNG(cv::getTickCount());
	
	for (int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			neurons[y][x].randomize(rng, 0., 1.);
		}
	}
}

SOM::~SOM()
{
	delete distfun;
}

cv::Point SOM::identifyWinnerNeuron(const multi_img::Pixel &inputVec) const
{
	// initialize with maximum value
	double closestDistance = DBL_MAX;
	double dist;
	// init grid position
	cv::Point winner(-1, -1);

	// find closest Neuron to inputVec in the SOM
	// iterate over all neurons in grid
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
       		dist = distfun->getSimilarity(neurons[y][x], inputVec);
			// compare current distance with minimal found distance
			if (dist < closestDistance) {
				// set new minimal distance and winner position
				closestDistance = dist;
				winner = cv::Point(x, y);
			}
		}
	}
	assert(winner.x >= 0);
	return winner;
}

multi_img SOM::export_2d()
{
	multi_img ret(height, width, dim);
	ret.maxval = 1.;
	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			ret.setPixel(y, x, neurons[y][x]);
		}
	}
	return ret;
}

void SOM::updateSingle(const cv::Point &pos, const multi_img::Pixel &input, double weight)
{
  // find neuron
	Neuron *currentNeuron = getNeuron(pos.x, pos.y);

	// update neuron using openCV
	currentNeuron->update(input, weight);
}

void SOM::updateSingle(const cv::Point3i &pos, const multi_img::Pixel &input, double weight)
{
	// find neuron
	Neuron *currentNeuron = getNeuron(pos.x, pos.y * width + pos.z);

	// update neuron using openCV
	currentNeuron->update(input, weight);
}

void SOM::updateNeighborhood(const cv::Point &pos, const multi_img::Pixel &input, double radius, double learnRate)
{
	if (config.hack3d) {
		updateNeighborhood3(pos, input, radius, learnRate);
		return;
	}

	double rad = (height == 1 ? radius : radius*radius);

	int minX = 0; int minY = 0;
	int maxX = height - 1;
	int maxY = width - 1;

	bool finished = false;
	int y = pos.y, x;
	while (1) { // y loop
		x = pos.x;
		while (1) { // x loop
			// squared distance(topological) between winning and current neuron
			double dist = (pos.x - x) * (pos.x - x) + (pos.y - y) * (pos.y - y);

			// calculate the time- and radius dependent weighting factor
			//TODO no real gaussian here
			double weightingFactor = learnRate * exp(-(dist)/(2.0*rad));
			// check if change is still relevant
			if (weightingFactor < 0.01) {
				if (x == pos.x) // we are finished in y-direction
					finished = true;
				break;  // at least we are always finished in x-direction here
			}

			if (x <= maxX && y <= maxY)
				updateSingle(cv::Point(x, y), input, weightingFactor);
			// now with opposite indices
			int yy = pos.y+pos.y - y;
			int xx = pos.x+pos.x - x;
			if (x != xx && xx >= minX && y <= maxY)
				updateSingle(cv::Point(xx, y), input, weightingFactor);
			if (y != yy && x <= maxX && yy >= minY)
				updateSingle(cv::Point(x, yy), input, weightingFactor);
			if (x != xx && y != yy && xx >= minX && yy >= minY)
				updateSingle(cv::Point(xx, yy), input, weightingFactor);

			x++;
    }
		if (finished)
			break;
		y++;
	}
}

void SOM::updateNeighborhood3(const cv::Point &pos2d, const multi_img::Pixel &input,
                                     double radius, double learnRate) {
	int minI = 0;
	int maxI = width - 1;
	cv::Point3i pos(pos2d.x, pos2d.y / (maxI+1), pos2d.y % (maxI+1));
//	std::cerr << "Point " << pos2d.x << "." << pos2d.y << "\tis "
//			<< pos.x << "." << pos.y << "." << pos.z << std::endl;

	static int it = 0;
	if (!(++it % 100))
		std::cerr << "update " << it << " at " << pos.x << "." << pos.y << "." << pos.z
				  << " with radius " << radius << std::endl;
	bool finishedX, finishedY = false;
	int z = pos.z, y, x;
	while (1) { // z loop
		y = pos.y;
		while (1) { // y loop
			finishedX = false;
			x = pos.x;
			while (1) { // x loop
				// squared distance(topological) between winning and current neuron
				double dist = (pos.x - x) * (pos.x - x) + (pos.y - y) * (pos.y - y)
							  + (pos.z - z) * (pos.z - z);

				// calculate the time- and radius dependent weighting factor
				//TODO no real gaussian here
				double weightingFactor = learnRate * exp(-(dist)/(2.0*radius*radius*radius));
				// check if change is still relevant
				if (weightingFactor < 0.01) {
					if (x == pos.x)	// we are finished in y-direction
						finishedX = true;
					break;	// at least we are always finished in x-direction here
				}

				int yy = pos.y+pos.y - y;
				int xx = pos.x+pos.x - x;
				int zz = pos.z+pos.z - z;
				if (z <= maxI) {
					if (x <= maxI && y <= maxI)
						updateSingle(cv::Point3i(x, y, z), input, weightingFactor);
					if (x != xx && xx >= minI && y <= maxI)
						updateSingle(cv::Point3i(xx, y, z), input, weightingFactor);
					if (y != yy && x <= maxI && yy >= minI)
						updateSingle(cv::Point3i(x, yy, z), input, weightingFactor);
					if (x != xx && y != yy && xx >= minI && yy >= minI)
						updateSingle(cv::Point3i(xx, yy, z), input, weightingFactor);
				}
				if (zz >= minI) {
					if (x <= maxI && y <= maxI)
						updateSingle(cv::Point3i(x, y, zz), input, weightingFactor);
					if (x != xx && xx >= minI && y <= maxI)
						updateSingle(cv::Point3i(xx, y, zz), input, weightingFactor);
					if (y != yy && x <= maxI && yy >= minI)
						updateSingle(cv::Point3i(x, yy, zz), input, weightingFactor);
					if (x != xx && y != yy && xx >= minI && yy >= minI)
						updateSingle(cv::Point3i(xx, yy, zz), input, weightingFactor);
				}

				x++;
			}
			if (finishedX) {
				if (y == pos.y)
					finishedY = true;
				break;
			}
			y++;
		}
		if (finishedY)
			break;
		z++;
	}
}

double SOM::getDistance(const cv::Point2d &p1, const cv::Point2d &p2) const
{
	double dx = (p1.x - p2.x);
	double dy = (p1.y - p2.y);
	return std::sqrt(dx * dx + dy * dy);
}
