#include "command_overlay_segmentation.h"
#include "superpixel_segmentation.h"
#include "normal_image.h"
#include <opencv2/highgui/highgui.hpp>

namespace vole {

CommandOverlaySegmentation::CommandOverlaySegmentation():
	Command("seg_overlay",
		config,
		"Christian Riess",
		"riess@i5.cs.fau.de"
	)
{
}

int CommandOverlaySegmentation::execute()
{
	if ((config.input_file.length() < 1) && (config.input_file_16_bit.length() < 1)) {
		std::cerr << "no input file given (-I or -J), aborted." << std::endl;
		return 1;
	}
	cv::Mat_<cv::Vec3d> norm_img;
	if (config.input_file.length() > 0) {
		cv::Mat_<cv::Vec3b> img = cv::imread(config.input_file, 1 /* 8 bit RGB */);
		if (img.data == NULL) {
			std::cerr << "unable to read 8 bit image " << img << " (note: image must be RGB, 8 bits)" << std::endl;
			return 1;
		}
		int max = 255;
		norm_img = vole::NormalImage::normalize((cv::Mat_<cv::Vec3b>)img, 0, max);
		int mymax = -1;
		for (int y = 0; y < norm_img.rows; ++y) {
			for (int x = 0; x < norm_img.cols; ++x) {
				for (int c = 0; c < 3; ++c) {
					if (norm_img[y][x][c] > mymax) mymax = norm_img[y][x][c];
				}
			}
		}
		config.min_intensity = 0;
		config.max_intensity = max;
	} else {
		cv::Mat_<cv::Vec3s> img = cv::imread(config.input_file_16_bit, -1 /* read as-is (16 bit, breaks if image has alpha channel) */);
		if (img.data == NULL) {
			std::cerr << "unable to read 16 bit image " << img << " (note: image must be RGB, 16 bits, no alpha channel)" << std::endl;
			return 1;
		}
		int max = -1;
		norm_img = vole::NormalImage::normalize((cv::Mat_<cv::Vec3s>)img, 0, max);
		config.min_intensity = 0;
		config.max_intensity = max;
	}

	// obtain annotation color
	size_t comma1 = config.annotation_color.find_first_of(',');
	size_t comma2 = config.annotation_color.find_first_of(',', comma1+1);
	if ((comma1 == std::string::npos) || (comma2 == std::string::npos) || (config.annotation_color.find_first_of(',', comma2+1) != std::string::npos))
	{
		std::cerr << "Invalid string for annotation color (option -A) given, format <R>,<G>,<B> expected; aborted." << std::endl;
		return 1;
	}
	cv::Vec3d annot;
    // if everything else is ok, start copying stuff
	{
		std::stringstream s;
		s << config.annotation_color.substr(0, comma1);
		s >> annot[2];
		s << config.annotation_color.substr(comma1+1, comma2-comma1-1);
		s >> annot[1];
		s << config.annotation_color.substr(comma2+1);
		s >> annot[0];
	}
	// normalize color if necessary
	if ((annot[0] > 1) || (annot[1] > 1) || (annot[2] > 1)) {
		annot = annot / 255;
	}
	

	// load superpixels image
	if (config.segmentation_file.length() < 1) {
		std::cerr << "Segmentation file (option -S) required, aborted." << std::endl;
		return 1;
	}
	cv::Mat_<cv::Vec3b> seg = cv::imread(config.segmentation_file);
	
	if ((seg.cols != norm_img.cols) || (seg.rows != norm_img.rows)) {
		std::cerr << "dimensions of input image (" << norm_img.cols << "x" << norm_img.rows << ") different from dimensions" << std::endl;
		std::cerr << "of segmentation image (" << seg.cols << "x" << seg.rows << "), aborted." << std::endl;
		return 1;
	}

	// first row
	for (int x = 1; x < seg.cols; ++x) {
		int n1 = seg[0][  x][2] + 256*seg[0][  x][1] + 256*256*seg[0][  x][0];
		int n3 = seg[0][x-1][2] + 256*seg[0][x-1][1] + 256*256*seg[0][x-1][0];
		if (n1 != n3) {
			norm_img[0][x] = annot;
			norm_img[0][x-1] = annot;
		}
	}
	// first column
	for (int y = 1; y < seg.rows; ++y) {
		int n1 = seg[  y][0][2] + 256*seg[y  ][0][1] + 256*256*seg[  y][0][0];
		int n2 = seg[y-1][0][2] + 256*seg[y-1][0][1] + 256*256*seg[y-1][0][0];

		if (n1 != n2) {
			norm_img[y][0] = annot;
			norm_img[y-1][0] = annot;
		}
	}

	// remainder of the image
	for (int y = 1; y < seg.rows; ++y) {
		for (int x = 1; x < seg.cols; ++x) {
			int n1 = seg[  y][x][2] + 256*seg[y  ][x][1] + 256*256*seg[  y][x][0];
			int n2 = seg[y-1][x][2] + 256*seg[y-1][x][1] + 256*256*seg[y-1][x][0];
			int n3 = seg[y][x-1][2] + 256*seg[y][x-1][1] + 256*256*seg[y][x-1][0];

			if (n1 != n2) {
				norm_img[y][x] = annot;
				norm_img[y-1][x] = annot;
			}
			if (n1 != n3) {
				norm_img[y][x] = annot;
				norm_img[y][x-1] = annot;
			}
		}
	}

	cv::imwrite(config.output_file, norm_img*255);
	return 0;
}

void CommandOverlaySegmentation::printShortHelp() const
{
	std::cout << "Create a composite image where the segmentation boundaries are painted over the image." << std::endl;
}

void CommandOverlaySegmentation::printHelp() const
{
	std::cout << "Create a composite image where the segmentation boundaries are painted over the image." << std::endl;
}

} // vole
