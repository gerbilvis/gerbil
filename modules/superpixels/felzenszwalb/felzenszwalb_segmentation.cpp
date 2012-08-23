#include "felzenszwalb_segmentation.h"
#include "implementation/segment-image.h"
#include "implementation/image.h"
#include "implementation/misc.h"
#include "commands/felzenszwalb_config.h"
#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <boost/serialization/tracking.hpp>


namespace superpixels
{

FelzenszwalbSegmentation::FelzenszwalbSegmentation(vole::FelzenszwalbConfig config)
	: SuperpixelSegmentation(std::string("felzenszwalb"))
{
	localCfg = new vole::FelzenszwalbConfig();

	localCfg->input_file  = config.input_file;
	localCfg->input_file_16_bit  = config.input_file_16_bit;
	localCfg->min_intensity = config.min_intensity;
	localCfg->max_intensity = config.max_intensity;
	localCfg->output_file = config.output_file;
	localCfg->isGraphical = config.isGraphical;
	localCfg->kThreshold  = config.kThreshold;
	localCfg->sigma       = config.sigma;
	localCfg->min_size    = config.min_size;

	this->config = localCfg;
}

FelzenszwalbSegmentation::FelzenszwalbSegmentation(int min_size, int kThreshold, double sigma) :
	SuperpixelSegmentation(std::string("felzenszwalb"))
{
	localCfg = new vole::FelzenszwalbConfig();

	localCfg->input_file   = ""; // Problem: when hashing over the configuration,
							  // the hash values differ from runs where input
							  // and output files are known by using the other
							  // constructor
	localCfg->input_file_16_bit = "";
	localCfg->min_intensity = 0;
	localCfg->max_intensity = 1;
	localCfg->output_file = "";
	localCfg->isGraphical = false;
	localCfg->kThreshold  = kThreshold;
	localCfg->sigma       = sigma;
	localCfg->min_size    = min_size;
	
	this->config = localCfg;
}

FelzenszwalbSegmentation::~FelzenszwalbSegmentation()
{
	// superclass is deleting config in its destructor, hence also localCfg. We don't need to delete here. Weird/ugly but true.
}

std::vector<Superpixel> FelzenszwalbSegmentation::superpixels(const cv::Mat_<cv::Vec3b>& image)
{
	unsigned long int hashKey = 0;
	if (openCache(cache_dir) && isCaching()) {
		hashKey = localCfg->configHash();
		readFromCache(hashKey);
		//std::vector<superpixels::SuperpixelSegmentation::Superpixel> segmentation = readFromCache(hashKey);
		if (_superpixels.size() > 0) return _superpixels;
	}

	height = image.rows;
	width  = image.cols;

	felzenszwalb::image<felzenszwalb::rgb>* inputImage = new felzenszwalb::image<felzenszwalb::rgb>(width, height);

	for (int y = 0; y < height; y++) {
		for (int  x = 0; x < width; x++) {
			imRef(inputImage, x, y).r = image(y, x)[2];
			imRef(inputImage, x, y).g = image(y, x)[1];
			imRef(inputImage, x, y).b = image(y, x)[0];
		}
	}

	int numSegments;
	felzenszwalb::image<felzenszwalb::rgb>* segmentation
		= felzenszwalb::segment_image(inputImage,
			localCfg->sigma,
			localCfg->kThreshold,
			localCfg->min_size,
			&numSegments);

	_superpixels = std::vector<Superpixel>(numSegments);
	std::vector<Superpixel>::iterator sup;
	for (sup = _superpixels.begin(); sup != _superpixels.end(); sup++) {
		sup->bbox = cv::Rect(INT_MAX, INT_MAX, INT_MIN, INT_MIN);
	}

	std::map<int, int> labels;
	int label = 0;

	for (int y = 0; y < height; y++) {
		for (int  x = 0; x < width; x++) {
			cv::Point coord(x, y);
			int value = (imRef(segmentation, x, y).r << 16) + (imRef(segmentation, x, y).g << 8) + imRef(segmentation, x, y).b;

			if (labels.find(value) == labels.end()) {
				if (label >= numSegments) {
					std::cerr << "Error in felzenszwalb superpixel segmentation!";
					_superpixels.clear();
					labels.clear();
					break;
				}

				labels[value] = label;
				label++;
			}

			Superpixel &sup = _superpixels[labels[value]];
			sup.coordinates.push_back(coord);
			sup.bbox.x      = std::min(sup.bbox.x,      coord.x);
			sup.bbox.width  = std::max(sup.bbox.width,  coord.x);
			sup.bbox.y      = std::min(sup.bbox.y,      coord.y);
			sup.bbox.height = std::max(sup.bbox.height, coord.y);
		}
	}

	for (sup = _superpixels.begin(); sup != _superpixels.end(); sup++) {
		sup->bbox.width  = sup->bbox.width  - sup->bbox.x + 1;
		sup->bbox.height = sup->bbox.height - sup->bbox.y + 1;
	}

	delete inputImage;
	delete segmentation;

	if (isCaching()) {
		writeToCache(hashKey);
	}

	return _superpixels;
}

std::vector<Superpixel> FelzenszwalbSegmentation::superpixels(const cv::Mat_<cv::Vec3d>& image)
{
	cv::Mat_<cv::Vec3b> inputImage;
	image.convertTo(inputImage, 8, 255);
	return superpixels(inputImage);
}

/*
std::string FelzenszwalbSegmentation::getCacheIdentifier()
{
	return std::string("felzenszwalb");
}
*/

/*
std::string FelzenszwalbSegmentation::getStorageFilename(std::string dir, unsigned long int hashKey)
{
	std::stringstream s;
	s << dir << "/" << "felzenszwalb_" << hashKey << ".png";
	return s.str();
//	return localCfg->output_file;
}
*/

void FelzenszwalbSegmentation::addCachingInfo(std::string input_file, std::string input_file_16_bit, std::string output_file, bool isGraphical)
{
	localCfg->input_file = input_file;
	localCfg->input_file_16_bit = input_file_16_bit;
	localCfg->output_file = output_file;
	localCfg->isGraphical = isGraphical;
}

bool FelzenszwalbSegmentation::writeToCache(unsigned long int hashKey)
{
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

bool FelzenszwalbSegmentation::readFromCache(unsigned long int hashKey)
{
	// open file streams, serialize
	std::ifstream in;
	in.open(getStorageFilename(cache_dir, hashKey).c_str());
	if (!in.good()) return false;

	{
		boost::archive::text_iarchive ia(in);
		ia >> *this;
	} // stream is closed when archive is destructed
	
	return true;
}

} // namespace superpixels
