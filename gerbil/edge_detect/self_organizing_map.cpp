/*	
	Copyright(c) 2012 Ralph Muessig	and Johannes Jordan
	<johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "self_organizing_map.h"
#include <sm_factory.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <limits>

SOM::SOM(const vole::EdgeDetectionConfig &conf, int dimension)
	: dim(dimension), width(conf.width), height(conf.height),
	  neurons(Field(conf.height, Row(conf.width, Neuron(dimension)))),
	  config(conf)
{
	/// Create similarity measure
	distfun = vole::SMFactory<multi_img::Value>::spawn(config.similarity);
	assert(distfun);

	/// Uniformly randomizes each neuron
	// TODO: given interval [0..1] sure? purpose? it will not fit anyway
	cv::RNG rng(config.seed);
	
	for (int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			neurons[y][x].randomize(rng, 0., 1.);
		}
	}
}

SOM::SOM(const vole::EdgeDetectionConfig &conf, const multi_img &data)
	: dim(data.size()), width(conf.width), height(conf.height),
	  neurons(Field(conf.height, Row(conf.width, Neuron(data.size())))),
	  config(conf)
{
	/// Create similarity measure
	distfun = vole::SMFactory<multi_img::Value>::spawn(config.similarity);
	assert(distfun);

	/// Read SOM from multi_img

	for (int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			neurons[y][x] = data(y, x);
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
	double closestDistance = std::numeric_limits<double>::max();
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

static bool sortpair(std::pair<double, cv::Point> i,
					 std::pair<double, cv::Point> j) {
	return (i.first < j.first);
}

std::vector<std::pair<double, cv::Point> >
SOM::closestN(const multi_img::Pixel &inputVec, unsigned int N) const
{
	// initialize with maximum values
	std::vector<std::pair<double, cv::Point> > heap;
	for (int i = 0; i < N; ++i) {
		heap.push_back(std::make_pair(std::numeric_limits<double>::max(),
									  cv::Point(-1, -1)));
	}

	// find closest Neurons to inputVec in the SOM
	// iterate over all neurons in grid
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			double dist = distfun->getSimilarity(neurons[y][x], inputVec);
			// compare current distance with minimal found distance
			if (dist < heap[0].first) {
				// remove max. value in heap
				std::pop_heap(heap.begin(), heap.end(), sortpair);
				heap.pop_back();
				// add new value
				heap.push_back(std::make_pair(dist, cv::Point(x, y)));
				push_heap(heap.begin(), heap.end(), sortpair);
			}
		}
	}

	assert(heap[0].second.x >= 0);
	return heap;
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

void SOM::updateNeighborhood(const cv::Point &pos, const multi_img::Pixel &input, double radius, double learnRate)
{
	if (config.hack3d) {
		updateNeighborhood3(pos, input, radius, learnRate);
		return;
	}

	double rad = (height == 1 ? radius : radius*radius);

	int minX = 0; int minY = 0;
	int maxX = width - 1;
	int maxY = height - 1;

	bool finished = false;
	int y = pos.y, x;
	while (1) { // y loop
		x = pos.x;
		while (1) { // x loop
			// squared distance(topological) between winning and current neuron
			double dist = (pos.x - x) * (pos.x - x) + (pos.y - y) * (pos.y - y);

			// calculate the time- and radius dependent weighting factor
			//TODO no real gaussian here
			double weight = learnRate * exp(-(dist)/(2.0*rad));

			// check if change is still relevant
			if (weight < 0.01) {
				if (x == pos.x) // we are finished in y-direction
					finished = true;
				break;  // at least we are always finished in x-direction here
			}

			if (x <= maxX && y <= maxY)
				neurons[y][x].update(input, weight);
			// now with opposite indices
			int yy = pos.y+pos.y - y;
			int xx = pos.x+pos.x - x;
			if (x != xx && xx >= minX && y <= maxY)
				neurons[y][xx].update(input, weight);
			if (y != yy && x <= maxX && yy >= minY)
				neurons[yy][x].update(input, weight);
			if (x != xx && y != yy && xx >= minX && yy >= minY)
				neurons[yy][xx].update(input, weight);

			x++;
    }
		if (finished)
			break;
		y++;
	}
}

void SOM::updateSingle3(const cv::Point3i &pos, const multi_img::Pixel &input, double weight)
{
	neurons[pos.y * width + pos.z][pos.x].update(input, weight);
}

void SOM::updateNeighborhood3(const cv::Point &pos2d, const multi_img::Pixel &input,
                                     double radius, double learnRate) {
	int minI = 0;
	int maxI = width - 1;
	cv::Point3i pos(pos2d.x, pos2d.y / width, pos2d.y % width);
//	std::cerr << "Point " << pos2d.x << "." << pos2d.y << "\tis "
//			<< pos.x << "." << pos.y << "." << pos.z << std::endl;

/*	static int it = 0;
	if (!(++it % 100))
		std::cerr << "update " << it << " at " << pos.x << "." << pos.y << "." << pos.z
				  << " with radius " << radius << std::endl;*/
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
						updateSingle3(cv::Point3i(x, y, z), input, weightingFactor);
					if (x != xx && xx >= minI && y <= maxI)
						updateSingle3(cv::Point3i(xx, y, z), input, weightingFactor);
					if (y != yy && x <= maxI && yy >= minI)
						updateSingle3(cv::Point3i(x, yy, z), input, weightingFactor);
					if (x != xx && y != yy && xx >= minI && yy >= minI)
						updateSingle3(cv::Point3i(xx, yy, z), input, weightingFactor);
				}
				if (zz >= minI) {
					if (x <= maxI && y <= maxI)
						updateSingle3(cv::Point3i(x, y, zz), input, weightingFactor);
					if (x != xx && xx >= minI && y <= maxI)
						updateSingle3(cv::Point3i(xx, y, zz), input, weightingFactor);
					if (y != yy && x <= maxI && yy >= minI)
						updateSingle3(cv::Point3i(x, yy, zz), input, weightingFactor);
					if (x != xx && y != yy && xx >= minI && yy >= minI)
						updateSingle3(cv::Point3i(xx, yy, zz), input, weightingFactor);
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

double SOM::getDistance3(const cv::Point3d &p1, const cv::Point3d &p2) const
{
	double dx = (p1.x - p2.x);
	double dy = (p1.y - p2.y);
	double dz = (p1.z - p2.z);
	return std::sqrt(dx * dx + dy * dy + dz * dz);
}
