/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>,
	Felix Lugauer, Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VOLE_GRADIENT_DIFFERENCE_H
#define VOLE_GRADIENT_DIFFERENCE_H

#include "similarity_measure.h"

namespace vole {

template<typename T>
class GradientDifference : public SimilarityMeasure<T> {

public:
	GradientDifference(int bins = 150)
		: SimilarityMeasure<T>(), hbins(bins) {}

	double getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2);
	static double getEntropy(const cv::Mat_<T> &img, int bins);

	int hbins;			//*< number of bins used for the entropy histogram

};

template<typename T>
inline double GradientDifference<T>::getSimilarity(const cv::Mat_<T> &img1, const cv::Mat_<T> &img2)
{
	this->check(img1, img2);

	cv::Mat_<float> G1x, G2x;
	cv::Mat_<float> G1y, G2y;

	cv::Mat_<float> gdiffX(img1.cols, img1.rows, 0.f);
	cv::Mat_<float>	gdiffY(img1.cols, img1.rows, 0.f);
	cv::Mat_<float>	gdiff(img1.cols, img1.rows, 0.f);
		
	// Calculate gradient of img1 and img2 in x and y direction
	cv::Sobel(img1, G1x, CV_32FC1, 1, 0);
	cv::Sobel(img1, G1y, CV_32FC1, 0, 1);

	cv::Sobel(img2, G2x, CV_32FC1, 1, 0);
	cv::Sobel(img2, G2y, CV_32FC1, 0, 1);

	double scale = -1.0;
	double optEnt = img1.cols * img1.rows;
	// minimize entropy of difference images
	for (double step = 0.0; step < 2.0; step += 0.2) {

		// difference of gradients
		gdiffX = abs(G1x - step*G2x);
		gdiffY = abs(G1y - step*G2y);

		cv::MatConstIterator_<float> itx = gdiffX.begin(), ity = gdiffY.begin();
		cv::MatIterator_<float> itdest = gdiff.begin();
		for (; itdest != gdiff.end(); ++itx, ++ity, ++itdest) {
			float X = *itx, Y = *ity;
			*itdest = sqrt(X*X + Y*Y);
		}

		double ent = getEntropy(gdiff, hbins);
		if (ent < optEnt) {
			optEnt = ent;
			scale = step;
		}
	}

	// calculate gdiffs with optimized parameters
	gdiffX = abs(G1x - scale*G2x);
	gdiffY = abs(G1y - scale*G2y);

	// calculate standard deviation of gdiff
	cv::Scalar meanX, meanY;
	cv::Scalar stddevX, stddevY;
	cv::meanStdDev(gdiffX, meanX, stddevX);
	cv::meanStdDev(gdiffY, meanY, stddevY);

//		std::cout << "MeanX: " << meanX << " VarX: " << varX << " scale: " << scale << std::endl;
/* varX == 0 -> gets ugly
		if( varX == varY ) {
			varX = 0.00001;
			varY = 0.00001;
		}
*/
//	if(varX < 0.001 && varY < 0.001) {
//		return sampleNumber;
//	}
	double ret = 0.;

	cv::MatConstIterator_<float> itx = gdiffX.begin(), ity = gdiffY.begin();
	for (; itx != gdiffX.end(); ++itx, ++ity) {
		ret += stddevX[0] / (stddevX[0] + (*itx * *itx));
		ret += stddevY[0] / (stddevY[0] + (*ity * *ity));
	}

	return ret;
}

template <typename T>
inline double GradientDifference<T>::getEntropy(const cv::Mat_<T> &img, int bins) {
	cv::Mat_<float> hist;

	// we are lazy and don't fiddle out which values to expect
	double minVal, maxVal;
	cv::minMaxLoc(img, &minVal, &maxVal);

	// each has only one element (one-dimensional histogram)
	int histSize[] = { bins };
	float range[] = { (float)minVal, (float)maxVal };
	const float *ranges[] = { range };
	int channels[] = { 0 }; // we compute the histogram for the 0-th channel

	// calc histogram
	cv::calcHist(&img, 1, channels, cv::Mat(), hist, 1, histSize, ranges);

	double entropy = 0.0;
	double pixels = img.rows * img.cols;

	for (int i = 0; i < bins; i++) {
		double p = hist(i) / pixels;
		if (p != 0)
			entropy += (-p) * (log(p) / log(2.0));
	}

	return entropy;
}

}

#endif // VOLE_GRADIENT_DIFFERENCE_H
