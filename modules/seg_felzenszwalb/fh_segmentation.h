#ifndef FH_SEGMENTATION_H
#define FH_SEGMENTATION_H

#include "seg_felzenszwalb_config.h"
#include "image.h"
#include "misc.h"

#include "cv.h"

namespace vole {

/** \addtogroup VoleModules
 * */

/** \ingroup VoleModules
 * 	Core interface to the Felzenszwalb/Huttenlocher segmentation
 */
class fhSegmentation {
public:
	fhSegmentation(SegFelzenszwalbConfig &cfg);
	~fhSegmentation();

	// use this one
	void segment(
		const cv::Mat_<cv::Vec3b> &img,
		cv::Mat_<unsigned int> &labels,
		std::vector<std::vector<cv::Point> > &linked_list,
		cv::Mat_<cv::Vec3b> &segmented_image);

	cv::Mat_<cv::Vec3b> segment(cv::Mat_<cv::Vec3b> &img);

	/* DO NOT use the following before fixing them! */
	void segment(cv::Mat_<cv::Vec3b> &in,
	             cv::Mat_<unsigned int> &labels,
	             std::vector<std::vector<cv::Point> > &out);

	SegFelzenszwalbConfig &cfg;

protected:
	image<rgb> *cvMat2imageRgb(cv::Mat_<cv::Vec3b> &mat);
	void imageRgb2cvMat(image<rgb> *img, cv::Mat_<cv::Vec3b> &mat);
};

}

#endif // FH_SEGMENTATION_H
