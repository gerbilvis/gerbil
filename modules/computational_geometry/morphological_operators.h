#ifndef VOLE_MORPHOLOGICAL_OPERATORS_H_
#define VOLE_MORPHOLOGICAL_OPERATORS_H_


#include "cv.h"


namespace vise {

class MorphologicalOperators
{
public:
	MorphologicalOperators()
	{
	}

	/*! dilates the values that are larger than threshold with a square shaped kernel with side length kernelSize, which should be an odd integer
	*/
	template <typename T>
	static cv::Mat_<T> dilate(cv::Mat_<T> input, int kernelSize, T threshold)
	{
		cv::Mat_<T> output;
		dilate<T>(input, output, kernelSize, threshold);
		return output;
	}

	/*! dilates the values that are larger than threshold with a square shaped kernel with side length kernelSize, which should be an odd integer
	*/
	template <typename T>
	static cv::Mat_<T> erode(cv::Mat_<T> input, int kernelSize, T threshold)
	{
		cv::Mat_<T> output;
		erode<T>(input, output, kernelSize, threshold);
		return output;
	}

	/*! dilates the values that are larger than threshold with a square shaped kernel with side length kernelSize, which should be an odd integer
	*/
	template <typename T>
	static void dilate(cv::Mat_<T> &input, cv::Mat_<T> &output, int kernelSize, T threshold)
	{
		double minVal, maxVal;
		cv::minMaxLoc(input, &minVal, &maxVal);
		std::cout << "dilate: min: " << minVal << ", max: " << maxVal << ", threshold: " << threshold << std::endl;
		input.copyTo(output);
		int leftOffset = kernelSize / 2;

		for (int y = 0; y < input.rows; y++) {
			for (int x = 0; x < input.cols; x++) {
				for (int i = y-leftOffset; i <= y+leftOffset; i++) {
					for (int j = x-leftOffset; j <= x+leftOffset; j++) {
						if (i < 0 || j < 0 || i >= input.rows || j >= input.cols) continue;
						if (input[i][j] >= threshold) {
							output[y][x] = input[i][j];
							goto thisPixelIsDone;
						}
					}
				}
	thisPixelIsDone: ;
			}
		}
	}

	/*! dilates all values that are larger than 0 with kernel size 3, useful for binary images
	*/
	template <typename T>
	static void dilate(cv::Mat_<T> &input, cv::Mat_<T> &output) {
		dilate<T>(input, output, 3, 0);
	}

	/*! erodes the values that are larger than threshold with a square shaped kernel with side length kernelSize, which should be an odd integer
	*/
	template <typename T>
	static void erode(cv::Mat_<T> &input, cv::Mat_<T> &output, int kernelSize, T threshold)
	{
		double minVal, maxVal;
		cv::minMaxLoc(input, &minVal, &maxVal);
		std::cout << "erode: min: " << minVal << ", max: " << maxVal << ", threshold: " << threshold << std::endl;

		input.copyTo(output);
		int leftOffset = kernelSize / 2;

		for (int y = 0; y < input.rows; y++) {
			for (int x = 0; x < input.cols; x++) {
				for (int i = y-leftOffset; i <= y+leftOffset; i++) {
					for (int j = x-leftOffset; j <= x+leftOffset; j++) {
						if (i < 0 || j < 0 || i >= input.rows || j >= input.cols) continue;
						int intensity = input[i][j];
						if (intensity < threshold) {
							output[y][x] = intensity;
							goto thisPixelIsDone;
						}
					}
				}
	thisPixelIsDone: ;
			}
		}
	}


	/*! erodes the values that are larger than 0 with kernel size 3, useful for binary images
	*/
	template <typename T>
	static void erode(cv::Mat_<T> &input, cv::Mat_<T> &output) {
		erode<T>(input, output, 3, 0);
	}

};

}

#endif // VOLE_MORPHOLOGICAL_OPERATORS_H_

