#ifndef IIC_NORMAL_IMAGE_H
#define IIC_NORMAL_IMAGE_H

#include <opencv2/imgproc/imgproc.hpp>

#ifdef WITH_QT
#include <QImage>
#endif // WITH_QT

namespace vole { namespace NormalImage {

static int defaultUpperLimit = -1;

// if upperLimit == -1: scaling is performed relatively
cv::Mat_<cv::Vec3d> normalize(cv::Mat_<cv::Vec3b> img, int lowerLimit=0, int &upperLimit=defaultUpperLimit);
cv::Mat_<cv::Vec3d> normalize(cv::Mat_<cv::Vec3s> img, int lowerLimit=0, int &upperLimit=defaultUpperLimit); // give explicit upper limit to fix boundary
cv::Mat_<cv::Vec3d> normalizeClipped(cv::Mat_<cv::Vec3s> img, int lowerLimit=0, int &upperLimit=defaultUpperLimit);

#ifdef WITH_QT
	cv::Mat_<cv::Vec3d> normalize(QImage *img, int lowerLimit=-1, int &upperLimit=defaultUpperLimit);
	double max_entry(QImage *img);
#endif // WITH_QT


template <typename T>
double max_entry_one_channel(cv::Mat_<T> img)
{
	double max = -1;
	for (int y = 0; y < img.rows; ++y)
		for (int x = 0; x < img.cols; ++x)
			if (img[y][x] > max)
				max = img[y][x];
	return max;
}

template <typename T>
double max_entry(cv::Mat_<T> img)
{
	int channels = img.channels();
	double max = -1;
	for (int y = 0; y < img.rows; ++y) 
		for (int x = 0; x < img.cols; ++x) 
			for (int c = 0; c < channels; ++c) 
				if (img[y][x][c] > max)
					max = img[y][x][c];
	return max;
}


} }


#endif // IIC_NORMAL_IMAGE_H
