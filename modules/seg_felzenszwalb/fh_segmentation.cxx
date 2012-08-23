#include "fh_segmentation.h"

// original code by Felzenszwalb and Huttenlocher
#include "segment-image.h"

#include <pixel_operations.h>
#include <iostream>
#include <highgui.h>

namespace vole {

fhSegmentation::fhSegmentation(SegFelzenszwalbConfig &cfg) : cfg(cfg)
{

}

fhSegmentation::~fhSegmentation() 
{
}


// FIXME lots of code duplication with the next segment method!!
// FIXME The Chromaticity Image Option is NOT respected in the other functions
void fhSegmentation::segment(
	const cv::Mat_<cv::Vec3b> &src_img,
	cv::Mat_<unsigned int> &labels,
	std::vector<std::vector<cv::Point> > &linked_list,
	cv::Mat_<cv::Vec3b> &segmented_image)
{
	cv::Mat_<cv::Vec3b> img;
	if (cfg.chroma_img) {
		img = PixelOperations::getChromaticityImage255(src_img);
	} else {
		img = src_img;
	}

//	if (cfg.isGraphical) {
//		cv::imshow("Initial image", img);
//		cv::waitKey();
//	}

	image<rgb> *rgbimg = cvMat2imageRgb(img);
	int num_ccs;
	universe *u;
	segment_image(
		rgbimg, &u,
		cfg.sigma, cfg.k_threshold, cfg.min_size,
		&num_ccs);

	if (cfg.verbosity > 0) {
		std::cout << "Number of segments: " << num_ccs << std::endl;
	}
	
	// wrap result in label img and linked_list;
	int n_numbers = 0;
	std::map<int, int> components;
	int width = rgbimg->width();
	int height = rgbimg->height();
	labels = cv::Mat_<unsigned int>(height, width);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int comp = u->find(y * width + x);
			if (components.find(comp) == components.end()) {
				components[comp] = n_numbers;
				std::vector<cv::Point> tmp;
				linked_list.push_back(tmp);
				++n_numbers;
			}
			unsigned int normalized_comp = components[comp];
			labels[y][x] = normalized_comp;
			cv::Point i(x, y);
			linked_list[normalized_comp].push_back(i);
		}
	}

	segmented_image = cv::Mat_<cv::Vec3b>(height, width);
	// pick random colors for each component
	rgb *colors = new rgb[width * height];
	for (int i = 0; i < width * height; i++)
		colors[i] = random_rgb();

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int comp = u->find(y * width + x);
			segmented_image(y, x)[0] = colors[comp].b;
			segmented_image(y, x)[1] = colors[comp].g;
			segmented_image(y, x)[2] = colors[comp].r;
		}
	}
	delete u;
	delete rgbimg;
	delete[] colors;
}


// FIXME The Chromaticity Image Option is NOT respected here
void fhSegmentation::segment(
	cv::Mat_<cv::Vec3b> &img,
	cv::Mat_<unsigned int> &labels,
	std::vector<std::vector<cv::Point> > &linked_list)
{
	image<rgb> *rgbimg = cvMat2imageRgb(img);
	int num_ccs;
	universe *u;
	segment_image(
		rgbimg, &u,
		cfg.sigma, cfg.k_threshold, cfg.min_size,
		&num_ccs);

	if (cfg.verbosity > 0) {
		std::cout << "Number of segments: " << num_ccs << std::endl;
	}
	
	// wrap result in label img and linked_list;
	int n_numbers = 0;
	std::map<int, int> components;
	int width = rgbimg->width();
	int height = rgbimg->height();
	labels = cv::Mat_<unsigned int>(height, width);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int comp = u->find(y * width + x);
			if (components.find(comp) == components.end()) {
				components[comp] = n_numbers;
				std::vector<cv::Point> tmp;
				linked_list.push_back(tmp);
				++n_numbers;
			}
			unsigned int normalized_comp = components[comp];
			labels[y][x] = normalized_comp;
			cv::Point i(x, y);
			linked_list[normalized_comp].push_back(i);
		}
	}
	delete rgbimg;
	delete u;
}

// convert cv::Mat_<cv::Vec3b> to image<rgb>
// deletes any data that is contained in img and stores the matrix in it
image<rgb> *fhSegmentation::cvMat2imageRgb(cv::Mat_<cv::Vec3b> &mat) {
	image<rgb> *img = new image<rgb>(mat.cols, mat.rows);
	for (int y = 0; y < mat.rows; ++y) {
		cv::Vec3b *cur_row = mat[y];
		for (int x = 0; x < mat.cols; ++x) {
			img->access[y][x].r = cur_row[x][2];
			img->access[y][x].g = cur_row[x][1];
			img->access[y][x].b = cur_row[x][0];
		}
	}
	return img;
}

// convert image<rgb> to cv::Mat_<cv::Vec3b>
void fhSegmentation::imageRgb2cvMat(image<rgb> *img, cv::Mat_<cv::Vec3b> &mat) {
	mat = cv::Mat_<cv::Vec3b>(img->height(), img->width());
	for (int y = 0; y < img->height(); ++y) {
		cv::Vec3b *cur_row = mat[y];
		for (int x = 0; x < img->width(); ++x) {
			cur_row[x][2] = img->access[y][x].r;
			cur_row[x][1] = img->access[y][x].g;
			cur_row[x][0] = img->access[y][x].b;
		}
	}
}


cv::Mat_<cv::Vec3b> fhSegmentation::segment(cv::Mat_<cv::Vec3b> &src_img)
{
	cv::Mat_<cv::Vec3b> img;
	if (cfg.chroma_img) {
		img = PixelOperations::getChromaticityImage255(src_img);
	} else {
		img = src_img;
	}
	if (cfg.isGraphical) {
		cv::imshow("Initial image", img);
		cv::waitKey();
	}

	image<rgb> *rgbimg = cvMat2imageRgb(img);
	// segment it
	int num_ccs;
	image<rgb> *seg = segment_image(
		rgbimg,
		cfg.sigma,
		cfg.k_threshold,
		cfg.min_size,
		&num_ccs
	); 

	if (cfg.verbosity > 0) {
		std::cout << "Number of segments: " << num_ccs << std::endl;
	}

	cv::Mat_<cv::Vec3b> result_img;
	imageRgb2cvMat(seg, result_img);	

	delete rgbimg;
	return result_img;
}

}
