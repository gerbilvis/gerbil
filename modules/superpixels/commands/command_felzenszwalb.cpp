#include "command_felzenszwalb.h"
#include "felzenszwalb/felzenszwalb_segmentation.h"
#include "normal_image.h"
#include <opencv2/highgui/highgui.hpp>

namespace vole {

CommandFelzenszwalb::CommandFelzenszwalb():
	Command("felzenszwalb",
		config,
		"Michael Bleier",
		"michael.bleier@mb.stud.uni-erlangen.de"
	)
{
}

int CommandFelzenszwalb::execute()
{

	if ((config.input_file.length() < 1) && (config.input_file_16_bit.length() < 1)) {
		std::cerr << "no input file given (-I or -J), aborted." << std::endl;
		return 1;
	}
	cv::Mat_<cv::Vec3d> normalized_image;
	if (config.input_file.length() > 0) {
		cv::Mat_<cv::Vec3b> img = cv::imread(config.input_file, 1 /* 8 bit RGB */);
		if (img.data == NULL) {
			std::cerr << "unable to read 8 bit image " << img << " (note: image must be RGB, 8 bits)" << std::endl;
			return 1;
		}
		int max = 255;
		normalized_image = vole::NormalImage::normalize((cv::Mat_<cv::Vec3b>)img, 0, max);
	} else {
		cv::Mat_<cv::Vec3s> img = cv::imread(config.input_file_16_bit, -1 /* read as-is (16 bit, breaks if image has alpha channel) */);
		if (img.data == NULL) {
			std::cerr << "unable to read 16 bit image " << img << " (note: image must be RGB, 16 bits, no alpha channel)" << std::endl;
			return 1;
		}
		int max = -1;
		normalized_image = vole::NormalImage::normalize((cv::Mat_<cv::Vec3s>)img, 0, max);
		config.min_intensity = 0;
		config.max_intensity = max;
	}

	superpixels::FelzenszwalbSegmentation felzenszwalb(config.min_size, config.kThreshold, config.sigma);

#if 0
	// Segment image into superpixels
	std::vector<superpixels::SuperpixelSegmentation::Superpixel> superpixels = felzenszwalb.superpixels(inputImage);

	// Iterate over all superpixels
	for (std::vector<superpixels::SuperpixelSegmentation::Superpixel>::iterator it = superpixels.begin(); it != superpixels.end(); it++) {
		std::vector<cv::Point> coords = it->coordinates;

		// Do something with superpixel coordinates
	}
#endif

	// Create an image with superpixels marked by random colors
	std::vector<superpixels::Superpixel> supixels = felzenszwalb.superpixels(normalized_image);
	cv::Mat_<cv::Vec3b> superpixelsImage = felzenszwalb.superpixelsImage(supixels,
		normalized_image.rows, normalized_image.cols,
		config.deterministic_coloring);

	if (config.isGraphical) {
		cv::Mat_<cv::Vec3b> randomlyColoredImage;
		if (config.deterministic_coloring) {
			randomlyColoredImage = felzenszwalb.superpixelsImage(supixels, normalized_image.rows, normalized_image.cols, false);
		} else {
			randomlyColoredImage = superpixelsImage;
		}

		cv::imshow("Felzenszwalb/Huttenlocher superpixel segmentation", randomlyColoredImage);
		cv::waitKey();
	}

	if (config.output_file.length() > 0) {
		cv::imwrite(config.output_file, superpixelsImage);
	}

	return 0;
}

void CommandFelzenszwalb::printShortHelp() const
{
	std::cout << "Superpixel segmentation by Felzenszwalb and Huttenlocher." << std::endl;
}

void CommandFelzenszwalb::printHelp() const
{
	std::cout << "Superpixel segmentation by Felzenszwalb and Huttenlocher." << std::endl;
}

} // vole
