#include "mapper.h"
#include <multi_img/multi_img.h>
#include <sm_factory.h>
#include <imginput.h>
#include <opencv2/highgui/highgui.hpp>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <iostream>
#include <exception>

namespace sm = similarity_measures;
namespace mapper {

int Mapper::execute()
{
	auto image = imginput::ImgInput(config.input).execute();
	cv::Mat1b mask = cv::imread(config.mask, CV_LOAD_IMAGE_GRAYSCALE);
	if (image->empty() || mask.empty())
		throw std::runtime_error("Image file(s) could not be read!");
	if (image->width != mask.cols || image->height != mask.rows)
		throw std::runtime_error("Image and mask geometries do not match!");

	image->rebuildPixels();
	auto spectra = image->getSegment(mask);
	multi_img::Pixel mean(image->size());
	for (auto p : spectra) {
		cv::add(mean, *p, mean);
	}
	cv::Mat1f meanMat(mean);
	meanMat /= spectra.size();

	if (config.verbosity > 1) {
		std::cout << "Reference spectrum: " << std::endl;
		for (auto v : mean) {
			std::cout << v << "  ";
		}
		std::cout << std::endl;
	}

	auto simfun = sm::SMFactory<multi_img::Value>::spawn(config.similarity);
	cv::Mat1f map(image->height, image->width);
	tbb::parallel_for(tbb::blocked_range2d<int>(0, map.rows, 0, map.cols),
		                  [&](tbb::blocked_range2d<int> r) {
			for (int y = r.rows().begin(); y != r.rows().end(); ++y) {
				for (int x = r.cols().begin(); x != r.cols().end(); ++x) {
					map(y,x) = -simfun->getSimilarity((*image)(y,x), mean);
				}
			}
		});

	cv::normalize(map, map, 0, 255, cv::NORM_MINMAX);
	cv::imwrite(config.output, map);

	return 0; // success
}

void Mapper::printShortHelp() const {
	std::cout << "Pixel-wise similarity mapper based on reference mask." << std::endl;
}

void Mapper::printHelp() const {
	std::cout << "Calculates the similarity of each pixel with the average\n"
	             "of the reference region in the same image.";
	std::cout << std::endl;
}

}
