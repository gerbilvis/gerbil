#ifndef __SUPERPIXELS_H__
#define __SUPERPIXELS_H__

#include "vlib/include/image.h"
#include <vector>

namespace veksler {

typedef int Value;

void check_error(int boolE, const char *error_message);
void PlaceSeeds(vlib::image<unsigned char> *I, int width, int height, int num_pixels,
	std::vector<int> &Seeds, int *numSeeds, int PATCH_SIZE);
void MoveSeedsFromEdges(vlib::image<unsigned char> *I, int width, int height, int num_pixels,
	std::vector<int> &Seeds, int numSeeds, int PATCH_SIZE);
Value computeEnergy(std::vector<int> &labeling, int width, int height, int num_pixels,
	std::vector<Value> &horizWeights, std::vector<Value> &vertWeights,
	std::vector<Value> &diag1Weights, std::vector<Value> &diag2Weights,
	std::vector<int> &Seeds, vlib::image<unsigned char> *I, int TYPE);
void getBounds(int width, int height, std::vector<int> &Seeds, int *seedX, int *seedY,
	int *startX, int *startY, int *endX, int *endY, int label, int PATCH_SIZE);
void expandOnLabel(int label, int width, int height, int num_pixels,
	std::vector<int> &Seeds, int numSeeds, std::vector<int> &labeling,
	std::vector<Value> &horizWeights, std::vector<Value> &vertWeights, Value lambda,
	std::vector<Value> &diag1Weights, std::vector<Value> &diag2Weights, int PATCH_SIZE,
	std::vector<int> &changeMask, std::vector<int> &changeMaskNew, vlib::image<unsigned char> *I,
	int TYPE, float variance);
void initializeLabeling(std::vector<int> &labeling, int width, int height,
	std::vector<int> &Seeds, int numSeeds, int PATCH_SIZE);
float computeImageVariance(vlib::image<unsigned char> *I, int width, int height);
void loadEdges(std::vector<Value> &weights, int num_pixels, int width, int height,
	Value lambda, char *name);
void computeWeights(std::vector<Value> &weights, int num_pixels, int width,
	int height, Value lambda, float variance, int incrX, int incrY,
	vlib::image<unsigned char> *I, int TYPE);
int saveSegmentationColor(std::vector<int> &labeling, int width, int height, int num_pixels,
	vlib::image<unsigned char> *I, int numSeeds, char *name);
void purturbSeeds(std::vector<int> &order, int numSeeds);
void check_input_arguments(int argc);

} // namespace veksler

#endif // __SUPERPIXELS_H__
