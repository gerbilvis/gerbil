#include "commands/command_srgb2rgb.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "img_read.h"

namespace rbase {

CommandSrgb2Rgb::CommandSrgb2Rgb()
 : vole::Command(
		"srgb2rgb",
		config,
		"Christian Riess",
		"christian.riess@informatik.uni-erlangen.de")
{
}

int CommandSrgb2Rgb::execute() {
	if (config.img.imageFile.length() < 1) {
		std::cerr << "input file name (--img.image) required!" << std::endl;
		return 1;
	}
	if (config.output_file.length() < 1) {
		std::cerr << "output file name (-O) required!" << std::endl;
		return 1;
	}
	if (config.maxTargetIntensity < 1) {
		std::cerr << "maxTargetIntensity (-M) must be greater or equal than 1" << std::endl;
		return 1;
	}
	cv::Mat_<cv::Vec3d> inputImage = iread::IRead::img_read_dl(
		config.img.imageFile,
		"srgb",
		config.img.dark_level,
		config.img.saturation_level
	);

	double maxInt = -1;
	for (int y = 0;  y < inputImage.rows; ++y)
		for (int x = 0; x < inputImage.cols; ++x)
			for (int c = 0; c < 3; ++c)
				if (inputImage[y][x][c] > maxInt)
					maxInt = inputImage[y][x][c];

	double multiplier = config.maxTargetIntensity / maxInt;
	if (maxInt <= 255) { // convert in cv::Vec3b
		cv::Mat_<cv::Vec3b> mat(inputImage.rows, inputImage.cols);
		for (int y = 0; y < inputImage.rows; ++y)
			for (int x = 0; x < inputImage.cols; ++x)
				for (int c = 0; c < 3; ++c)
					mat[y][x][c] = (uchar)
						std::max<int>(0, std::min<int>(255,
								inputImage[y][x][c]*multiplier));
		cv::imwrite(config.output_file, mat);
	} else { // else in cv::Vec3w
		cv::Mat_<cv::Vec3w> mat(inputImage.rows, inputImage.cols);
		for (int y = 0; y < inputImage.rows; ++y)
			for (int x = 0; x < inputImage.cols; ++x)
				for (int c = 0; c < 3; ++c)
					mat[y][x][c] = (unsigned short)
						std::max<int>(0, std::min<int>(65535,
								inputImage[y][x][c]*multiplier));
		cv::imwrite(config.output_file, mat);
	}
	if (config.verbosity > 0) {
		std::cout << "wrote linear image to " << config.output_file << std::endl;
	}

	return 0;
}


void CommandSrgb2Rgb::printShortHelp() const {
	std::cout << "Convert an sRGB image to RGB" << std::endl;
}

void CommandSrgb2Rgb::printHelp() const {
	std::cout << "Convert an sRGB image to RGB" << std::endl;
}


}
