#include "commands/veksler_config.h"
#include "veksler_segmentation.h"
#include "implementation/superpixels.h"
#include <iostream>

namespace superpixels {

VekslerSegmentation::VekslerSegmentation(vole::VekslerConfig cfg)
	: SuperpixelSegmentation(std::string("veksler"))
{
	localCfg = new vole::VekslerConfig();

	localCfg->input_file  = cfg.input_file;
	localCfg->input_file_16_bit  = cfg.input_file_16_bit;
	localCfg->min_intensity = cfg.min_intensity;
	localCfg->max_intensity = cfg.max_intensity;
	localCfg->output_file = cfg.output_file;
	localCfg->isGraphical = cfg.isGraphical;
	localCfg->patchSize   = cfg.patchSize;
	localCfg->type        = cfg.type;
	localCfg->iterations  = cfg.iterations;
	localCfg->lambda      = cfg.lambda;
	localCfg->verbosity   = cfg.verbosity;

	config = localCfg;
}

VekslerSegmentation::VekslerSegmentation(int patchSize, int type, int iterations, int lambda) :
	SuperpixelSegmentation(std::string("veksler"))
{
	localCfg = new vole::VekslerConfig();

	localCfg->input_file   = ""; // Problem: when hashing over the configuration,
						   // the hash values differ from runs where input
						   // and output files are known by using the other
						   // constructor
	localCfg->input_file_16_bit = "";
	localCfg->min_intensity = 0;
	localCfg->max_intensity = 1;
	localCfg->output_file = "";
	localCfg->isGraphical = false;
	localCfg->patchSize   = patchSize;
	localCfg->type        = type;
	localCfg->iterations  = iterations;
	localCfg->lambda      = lambda;

	// FIXME: this will leave ALL other config options uninitialized, e.g. verbosity!
	//        possible solution: also pass a meaningful config as a basic initialisation!

	config = localCfg;
}

VekslerSegmentation::~VekslerSegmentation()
{
	//superclass is deleting config in its destructor, hence also localCfg. We don't need to delete here. Weird/ugly but true.
}

std::vector<Superpixel> VekslerSegmentation::superpixels(const cv::Mat_<cv::Vec3b>& image)
{
	if (openCache(cache_dir) && isCaching()) {
		unsigned long int hashKey = localCfg->configHash();
		// TODO: verbosity test necessary! (config and localCfg have non meaningful verbosity values as they are not correctly initialized!
		std::cout << "hashKey: " << hashKey << std::endl;
		readFromCache(hashKey);
		if (_superpixels.size() > 0) return _superpixels;
	}

	const int width  = image.cols;
	const int height = image.rows;
	const int numPixels = width * height;

	vlib::image<unsigned char>* inputImage = new vlib::image<unsigned char>(width, height);

	int maxR = 0; int maxG = 0; int maxB = 0;

	for (int y = 0; y < height; y++) {
		for (int  x = 0; x < width; x++) {
			if (image[y][x][2] > maxR) maxR = image[y][x][2];
			if (image[y][x][1] > maxG) maxR = image[y][x][1];
			if (image[y][x][0] > maxB) maxR = image[y][x][0];
			double value = (0.3 * image(y, x)[2] + 0.59 * image(y, x)[1] + 0.11 * image(y, x)[0]);
			if (value > 255) {
				value = 255;
			}
			imRef(inputImage, x, y) = value;
		}
	}

	// TODO: can't that be dangerous in combination with random-dependent methods like IEBV?
	//       using a per commandline option passed random_seed won't reproduce same results anymore.
	srand(clock());

	const float variance = veksler::computeImageVariance(inputImage, width, height);

	std::vector<int> seeds(numPixels);
	int numSeeds = 0;
	veksler::PlaceSeeds(inputImage, width, height, numPixels, seeds, &numSeeds, localCfg->patchSize);
	veksler::MoveSeedsFromEdges(inputImage, width, height, numPixels, seeds, numSeeds, localCfg->patchSize);


	std::vector<int> horizWeights(numPixels, localCfg->lambda);
	std::vector<int> vertWeights(numPixels, localCfg->lambda);
	std::vector<int> diag1Weights(numPixels, localCfg->lambda);
	std::vector<int> diag2Weights(numPixels, localCfg->lambda);

	veksler::computeWeights(horizWeights, numPixels, width, height, localCfg->lambda, variance, -1, 0, inputImage, localCfg->type);
	veksler::computeWeights(vertWeights, numPixels, width, height, localCfg->lambda, variance, 0, -1, inputImage, localCfg->type);
	veksler::computeWeights(diag1Weights, numPixels, width, height, localCfg->lambda, variance, -1, -1, inputImage, localCfg->type);
	veksler::computeWeights(diag2Weights, numPixels, width, height, localCfg->lambda, variance, 1, -1, inputImage, localCfg->type);

	std::vector<int> labeling(numPixels);

	veksler::initializeLabeling(labeling, width, height, seeds, numSeeds, localCfg->patchSize);

	veksler::Value oldEnergy = 0;
	veksler::Value newEnergy = 0;

	std::vector<int> changeMask(numPixels, 1);
	std::vector<int> changeMaskNew(numPixels, 0);

	std::vector<int> order(numSeeds);
	for (int i = 0; i < numSeeds; i++) {
		order[i] = i;
	}

	int j = 0;

	while (true) {
		newEnergy = veksler::computeEnergy(labeling, width, height, numPixels, horizWeights, vertWeights, diag1Weights, diag2Weights, seeds, inputImage, localCfg->type);

		if (j == 0) {
			oldEnergy = newEnergy + 1;
		}

		if ((newEnergy == oldEnergy) || (j >= localCfg->iterations)) {
			break;
		}

		oldEnergy = newEnergy;

		for (int i = 0; i < numSeeds; i++) {
			veksler::expandOnLabel(order[i], width, height, numPixels, seeds, numSeeds, labeling, horizWeights, vertWeights,
				localCfg->lambda, diag1Weights, diag2Weights, localCfg->patchSize, changeMask, changeMaskNew, inputImage, localCfg->type, variance);
		}

		for (int i = 0; i < numPixels; i++) {
			changeMask[i] = changeMaskNew[i];
			changeMaskNew[i] = 0;
		}

		j++;
	}

	std::vector<Superpixel> segments(numSeeds);
	for (int y = 0; y < height; y++) {
		for (int  x = 0; x < width; x++) {
			cv::Point coord(x, y);
			segments[labeling[x + y * width]].coordinates.push_back(coord);
		}
	}

	_superpixels.clear();

	for (unsigned int i = 0; i < segments.size(); i++) {
		if (segments[i].coordinates.size() > 0) {
			_superpixels.push_back(segments[i]);
		}
	}

	delete inputImage;

	if (isCaching()) {
		unsigned long int hashKey = localCfg->configHash();
		writeToCache(hashKey); // , _superpixels, image.rows, image.cols);
	}

	return _superpixels;
}

std::vector<Superpixel> VekslerSegmentation::superpixels(const cv::Mat_<cv::Vec3d>& image)
{
	cv::Mat_<cv::Vec3b> inputImage;
	image.convertTo(inputImage, 8, 255);
	return superpixels(inputImage);
}

/*
std::string VekslerSegmentation::getCacheIdentifier()
{
	return std::string("veksler");
}

std::string VekslerSegmentation::getStorageFilename(std::string dir, unsigned long int hashKey)
{
	std::stringstream s;
	s << dir << "/" << "veksler_" << hashKey << ".png";
	return s.str();
//	return localCfg->output_file;
}
*/


void VekslerSegmentation::addCachingInfo(std::string input_file, std::string input_file_16_bit, std::string output_file, bool isGraphical)
{
	localCfg->input_file = input_file;
	localCfg->input_file_16_bit = input_file_16_bit;
	localCfg->output_file = output_file;
	localCfg->isGraphical = isGraphical;
}

bool VekslerSegmentation::writeToCache(unsigned long int hashKey)
{
	if (!cache_enabled) return false;
	// open file streams, serialize
	std::ofstream out;
	out.open(getStorageFilename(cache_dir, hashKey).c_str());
	if (!out.good()) return false;

	{
		boost::archive::text_oarchive oa(out);
		oa << *this;
	} // stream is closed when archive is destructed

	return true;
}

bool VekslerSegmentation::readFromCache(unsigned long int hashKey)
{
	if (!cache_enabled) return false;
	// open file streams, serialize
	std::ifstream in;
//	std::cout << "storage file: " 
	in.open(getStorageFilename(cache_dir, hashKey).c_str());
	if (!in.good()) return false;

	{
		boost::archive::text_iarchive ia(in);
		ia >> *this;
	} // stream is closed when archive is destructed
	
	return true;
}

} // namespace superpixels
