#include "command_veksler.h"
#include "veksler/veksler_segmentation.h"
#include "normal_image.h"
#include <opencv2/highgui/highgui.hpp>

namespace vole {

CommandVeksler::CommandVeksler():
	Command("veksler",
		config,
		"Michael Bleier",
		"michael.bleier@mb.stud.uni-erlangen.de"
	)
{
}

int CommandVeksler::execute()
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
		int mymax = -1;
		for (int y = 0; y < normalized_image.rows; ++y) {
			for (int x = 0; x < normalized_image.cols; ++x) {
				for (int c = 0; c < 3; ++c) {
					if (normalized_image[y][x][c] > mymax) mymax = normalized_image[y][x][c];
				}
			}
		}
		config.min_intensity = 0;
		config.max_intensity = max;
		std::cout << "norm max: " << mymax << std::endl;
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

	superpixels::VekslerSegmentation veksler(config.patchSize, config.type, config.iterations, config.lambda);
	veksler.addCachingInfo(config.input_file, config.input_file_16_bit, config.output_file, config.isGraphical);

	// Create an image with superpixels marked by random colors
	std::vector<superpixels::Superpixel> supixels = veksler.superpixels(normalized_image);
	cv::Mat_<cv::Vec3b> superpixelsImage = veksler.superpixelsImage(supixels,
		normalized_image.rows, normalized_image.cols,
		config.deterministic_coloring);

	if (config.isGraphical) {
		cv::Mat_<cv::Vec3b> randomlyColoredImage;
		if (config.deterministic_coloring) {
			randomlyColoredImage = veksler.superpixelsImage(supixels, normalized_image.rows, normalized_image.cols, false);
		} else {
			randomlyColoredImage = superpixelsImage;
		}
		cv::imshow("Veksler superpixel segmentation", randomlyColoredImage);
		cv::waitKey();
	}

	if (config.output_file.length() > 0) {
		cv::imwrite(config.output_file, superpixelsImage);
	}

	return 0;
}

void CommandVeksler::printShortHelp() const
{
	std::cout << "Superpixel segmentation by Veksler et al.." << std::endl;
}

void CommandVeksler::printHelp() const
{
	std::cout << "Superpixel segmentation by Veksler et al.." << std::endl;
}

} // vole
