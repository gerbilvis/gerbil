/*
	Copyright(c) 2013 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "rgb.h"
#include <imginput.h>

#ifdef WITH_SOM
#include <sm_config.h>
#include <sm_factory.h>
#include <gensom.h>
#include <som_cache.h>
#endif // WITH_SOM

#include <stopwatch.h>
#include <multi_img.h>
#include <opencv2/highgui/highgui.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <iostream>
#include <vector>
#include <algorithm>

#ifdef WITH_BOOST
#include <shared_data.h>
#endif

namespace rgb {

RGB::RGB()
 : Command(
		"rgb",
		config,
		"Johannes Jordan",
		"johannes.jordan@informatik.uni-erlangen.de")
{}

int RGB::execute()
{
	multi_img::ptr src = imginput::ImgInput(config.input).execute();
	if (src->empty())
		return 1;

	cv::Mat3f bgr = execute(*src);
	if (bgr.empty())
		return 1;

	if (config.verbosity & 2) {
		cv::imshow("Result", bgr);
		cv::waitKey();
	}
	
	cv::imwrite(config.output_file, bgr*255.);
	return 0;
}

#ifdef WITH_BOOST
std::map<std::string, boost::any> RGB::execute(
		std::map<std::string, boost::any> &input, ProgressObserver *po)
{
	// BUG BUG BUG
	// we need to copy the source image, because the pointer stored in the
	// shared data object may be deleted.
	// FIXME TODO gerbil's SharedData concept is broken.
	multi_img *srcimg;
	{
		SharedMultiImgPtr src =
				boost::any_cast<SharedMultiImgPtr>(input["multi_img"]);
		SharedDataLock lock(src->mutex);
		if ((**src).empty())
			assert(false);

		// copy, see above
		srcimg = new multi_img(**src);
	}

	cv::Mat3f bgr = execute(*srcimg, po);
	delete srcimg;

	if (bgr.empty()) {
		std::cerr << "RGB::execute(): empty result";
	}
	std::map<std::string, boost::any> output;
	output["multi_img"] = (cv::Mat3f)(bgr*255.0f);
	return output;
}
#endif

cv::Mat3f RGB::execute(const multi_img& src, ProgressObserver *po)
{
	cv::Mat3f bgr;

	switch (config.algo) {
	case COLOR_XYZ:
		bgr = src.bgr(); // progress observer is not used in XYZ calculation
		break;
	case COLOR_PCA:
		bgr = executePCA(src, po);
		break;
	case COLOR_SOM:
#ifdef WITH_SOM
		bgr = executeSOM(src, po);
#else
		throw std::runtime_error("RGB::execute(): SOM module not available.");
#endif
		break;
	default:
		throw std::runtime_error("RGB::execute(): bad config.algo");
	}

	return bgr;
}

cv::Mat3f RGB::executePCA(const multi_img& src, ProgressObserver *po)
{
	// cover cases of lt 3 channels
	unsigned int components = std::min(3u, src.size());
	multi_img pca3 = src.project(src.pca(components));

	bool cont = (!po) || po->update(.7f); // TODO: values
	if (!cont) return cv::Mat3f();

	if (config.pca_stretch)
		pca3.data_stretch_single(0., 1.);
	else
		pca3.data_rescale(0., 1.);

	cont = (!po) || po->update(.8f); // TODO: values
	if (!cont) return cv::Mat3f();

	std::vector<cv::Mat> vec(3);
	// initialize all of them in the case the source had less than 3 channels
	vec[0] = vec[1] = vec[2] = pca3[0]; // green: component 1
	if (pca3.size() > 1)
		vec[2] = pca3[1]; // red: component 2
	if (pca3.size() > 2)
		vec[0] = pca3[2]; // blue: component 3

	cv::Mat3f bgr;
	cv::merge(vec, bgr);
	return bgr;
}

#ifdef WITH_SOM
using namespace som;

// Compute weighted coordinates of multi_img pixels in SOM with dimensionality <= 3.
//
// For posToBGR == true, swap coordinates for false-color result.
template <bool posToBGR>
class SomRgbTbb {
public:
	typedef cv::Mat_<cv::Vec<GenSOM::value_type, 3> > Mat3;

	SomRgbTbb(const SOMClosestN &lookup, const std::vector<float> &weights,
			  Mat3& output, ProgressObserver *po = 0)
		: lookup(lookup), weights(weights), output(output), po(po) {}

	void operator()(const tbb::blocked_range2d<int> &r) const
	{
		typedef cv::Point3_<GenSOM::value_type> Point3;

		// iterate over all pixels in range
		float done = 0;
		float total = (lookup.height * lookup.width);
		for (int y = r.rows().begin(); y < r.rows().end(); ++y) {
			for (int x = r.cols().begin(); x < r.cols().end(); ++x) {
				SOMClosestN::resultAccess closest =
						lookup.closestN(cv::Point2i(x, y));
				Point3 weighted(0, 0, 0);
				std::vector<DistIndexPair>::const_iterator it = closest.first;
				for (int k = 0; it != closest.last; ++k, ++it) {
					size_t somidx = it->index;
					Point3 pos = vec2Point3(lookup.som.getCoord(somidx));
					weighted += weights[k] * pos;
				}
				if (posToBGR) { // 3D coord -> BGR color
					// for 2D SOM: weighted.z == 0 -> use only G and R
					Point3 tmp = weighted;
					weighted.x = tmp.z;  // B
					weighted.y = tmp.y;  // G
					weighted.z = tmp.x;  // R
				}
				output(y, x) = weighted;
				done++;
				if (po && ((int)done % 1000 == 0)) {
					if (!po->update(done / total, true))
						return; // abort if told so. This is thread-save.
					done = 0;
				}
			}
		}
		if (po)
			po->update(done / total, true);
	}

private:
	const SOMClosestN &lookup;
	const std::vector<float> &weights;
	Mat3 &output;
	ProgressObserver *po;
};

cv::Mat3f RGB::executeSOM(const multi_img &img, ProgressObserver *po,
						  boost::shared_ptr<GenSOM> som)
{
	typedef cv::Mat_<cv::Vec<GenSOM::value_type, 3> > Mat3;

	Stopwatch total("Total runtime of false-color image generation");

	img.rebuildPixels(false);
	ProgressObserver *calcPo;

	if (!som) {
		calcPo = (po ? new ChainedProgressObserver(po, .6f) : 0);
		som = boost::shared_ptr<GenSOM>(GenSOM::create(config.som, img, calcPo));
		delete calcPo;
	}
	if (po && !po->update(.6f))
		return Mat3();

	Stopwatch watch("Pixel color mapping");
	// compute lookup cache
	calcPo = (po ? new ChainedProgressObserver(po, .35f) : 0);
	SOMClosestN lookup(*som, img, config.som_depth, calcPo);
	delete calcPo;
	if (po && !po->update(.95f))
		return Mat3();

	// compute weighted coordinates of multi_img pixels in 3D SOM
	Mat3 bgr(img.height, img.width);
	calcPo = (po ? new ChainedProgressObserver(po, .05f) : 0);
	std::vector<float> weights =
			neuronWeightsGeometric<float>(config.som_depth);
	SomRgbTbb<true> comp(lookup, weights, bgr, calcPo);
	tbb::parallel_for(tbb::blocked_range2d<int>(0, img.height, // row range
												0, img.width), // column range
					  comp);
	// DEBUG: run sequentially (BTW currently it is too fast to profit from TBB)
	// comp(tbb::blocked_range2d<int>(0, img.height, 0, img.width));
	delete calcPo;

	return bgr;
}
#endif // WITH_SOM

void RGB::printShortHelp() const {
	std::cout << "RGB image creation (true-color or false-color)" << std::endl;
}


void RGB::printHelp() const {
	std::cout << "RGB image creation (true-color or false-color)" << std::endl;
	std::cout << std::endl;
	std::cout << "XYZ does a true-color image creation using a standard white balancing.\n"
	             "PCA and SOM do false-coloring.\"";
	std::cout << std::endl;
}

} // module namespace
