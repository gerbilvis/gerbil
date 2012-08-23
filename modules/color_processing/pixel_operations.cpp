#include "pixel_operations.h"

/* Computes chromaticities in a range between 0 and 255
 */
namespace PixelOperations {
	cv::Mat_<cv::Vec3b> getChromaticityImage255(const cv::Mat_<cv::Vec3b> &intensity_img) {
		cv::Mat_<cv::Vec3b> chroma_img(intensity_img.rows, intensity_img.cols);
		for (int y = 0; y < intensity_img.rows; ++y) {
			const cv::Vec3b *cur_row = intensity_img[y];
			cv::Vec3b *target_row = chroma_img[y];
			for (int x = 0; x < intensity_img.cols; ++x) {
				double sum = cur_row[x][0] + cur_row[x][1] + cur_row[x][2];
				target_row[x][0] = 255.0 * (cur_row[x][0] / sum);
				target_row[x][1] = 255.0 * (cur_row[x][1] / sum);
				target_row[x][2] = 255.0 * (cur_row[x][2] / sum);
			}
		}
		return chroma_img;
	}

	void excludeDarkBrightAndScalePixels(
		std::vector<cv::Vec3d> &input,
		std::vector<cv::Vec3d> &filtered,
		double min_intensity, double max_intensity)
	{
		filtered.assign(input.size(), cv::Vec3d());
		int count = 0;
		for (unsigned int i = 0; i < input.size(); ++i) {
			bool broken = false;
			for (int c = 0; c < 3; ++c) {
				// dividing by a fix value of 255.0, we are assuming pixel values between 0 and 255.0.
				// (make it image type dependent? Or as an option?)
				double scaled_color_value = input[i][c] / 255.0;

				if ((scaled_color_value < min_intensity) || (scaled_color_value > max_intensity)) {
					broken = true;
					break;
				} else {
					filtered[count][c] = static_cast<double>(scaled_color_value);
				}
			}
			if (!broken) count++;
		}
		filtered.resize(count);
	}

	void excludeDarkBrightAndScalePixels(
		cv::Mat_<cv::Vec3b> &region,
		std::vector<cv::Vec3d> &filtered,
		double min_intensity, double max_intensity)
	{
		filtered.assign(region.rows*region.cols, cv::Vec3d());
		int count = 0;
		for (int y = 0; y < region.rows; ++y) {
			for (int x = 0; x < region.cols; ++x) {
				bool broken = false;
				for (int c = 0; c < 3; ++c) {
					// dividing by a fix value of 255.0, we are assuming pixel values between 0 and 255.0.
					// (make it image type dependent? Or as an option?)
					double scaled_color_value = static_cast<double>(region[y][x][c]) / 255.0;

					if ((scaled_color_value < min_intensity) || (scaled_color_value > max_intensity)) {
						broken = true;
						break;
					} else {
						filtered[count][c] = scaled_color_value;
					}
				}
				if (!broken) count++;
			}
		}
		filtered.resize(count);
	}
}
