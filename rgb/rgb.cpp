/*
	Copyright(c) 2013 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "rgb.h"

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

namespace gerbil {


/* Forward progress information from SOMTrainer to RGB command. */
class SOMProgressObserver : public vole::ProgressObserver {
public:
	SOMProgressObserver(RGB *rgbCmd) : rgbCmd(rgbCmd) {}
    virtual bool update(int percent)
    {
        rgbCmd->setSomProgress(percent);
        return true; // if false: cancel job
    }
private:
	RGB *const rgbCmd;
};


RGB::RGB()
 :  Command(
		"rgb",
		config,
		"Johannes Jordan",
		"johannes.jordan@informatik.uni-erlangen.de"),
	srcimg(NULL),
	calcPo(new SOMProgressObserver(this)),
	abortFlag(false)
{}

RGB::~RGB()
{
	delete calcPo;
	delete srcimg;
}

int RGB::execute()
{
	multi_img::ptr src = vole::ImgInput(config.input).execute();
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
std::map<std::string, boost::any> RGB::execute(std::map<std::string, boost::any> &input, vole::ProgressObserver *po)
{
	{
		SharedMultiImgPtr src = boost::any_cast<SharedMultiImgPtr>(input["multi_img"]);
		SharedDataLock lock(src->mutex);
		if ((**src).empty())
			assert(false);

		// BUG BUG BUG
		// we need to copy the source image, because the pointer stored in the
		// shared data object may be deleted.
		// FIXME TODO gerbil's SharedData concept is broken.

		// copy
		srcimg = new multi_img(**src);
	}

	cv::Mat3f bgr = execute(*srcimg, po);
	if (bgr.empty())
		assert(false);

	std::map<std::string, boost::any> output;
	output["multi_img"] = (cv::Mat3f)(bgr*255.0f);
	return output;
}
#endif

cv::Mat3f RGB::execute(const multi_img& src, vole::ProgressObserver *po)
{
	cv::Mat3f bgr;

	this->po = po;

	switch (config.algo) {
	case COLOR_XYZ:
		bgr = src.bgr(); // updateProgress is not called in XYZ calculation
		break;
	case COLOR_PCA:
		bgr = executePCA(src);
		break;
	case COLOR_SOM:
#ifdef WITH_SOM
		bgr = executeSOM(src);
#else
		throw std::runtime_error("RGB::execute(): SOM module not available.");
#endif
		break;
	default:
		throw std::runtime_error("RGB::execute(): bad config.algo");
	}

	this->po = NULL;
	return bgr;
}

cv::Mat3f RGB::executePCA(const multi_img& src)
{
	// cover cases of lt 3 channels
	unsigned int components = std::min(3u, src.size());
	multi_img pca3 = src.project(src.pca(components));

	bool cont = progressUpdate(70.0f, po); // TODO: values
	if (!cont) return cv::Mat3f();

	if (config.pca_stretch)
		pca3.data_stretch_single(0., 1.);
	else
		pca3.data_rescale(0., 1.);

	cont = progressUpdate(80.0f, po); // TODO: values
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

// Compute weighted coordinates of multi_img pixels in SOM with dimensionality <= 3.
//
// For posToBGR == true, swap coordinates for false color result.
template <bool posToBGR>
class SomRgbTbb {
public:
	typedef cv::Mat_<cv::Vec<GenSOM::value_type, 3> > Mat3;

	SomRgbTbb(RGB &o, multi_img const& img, GenSOM *som, Mat3& weightedPos)
		: o(o),
		  som(som),
		  weightedPos(weightedPos),
		  weigths(o.config.som_depth),
		  closestN(*som, img, o.config.som_depth)
	{
		neuronWeightsGeometric(weigths);
	}

	void operator()(const tbb::blocked_range2d<int> &r) const
	{
		typedef cv::Point3_<GenSOM::value_type> Point3;

		// iterate over all pixels in range
		for (int y = r.rows().begin(); y < r.rows().end(); ++y) {
			for (int x = r.cols().begin(); x < r.cols().end(); ++x) {
				SOMClosestN::resultAccess closest =
						closestN.closestN(cv::Point2i(x, y));
				Point3 weighted(0,0,0);
				for (int k = 0; k < o.config.som_depth; ++k) {
					size_t somidx = (closest.first+k)->index;
					Point3 pos = vec2Point3(som->getCoord(somidx));
					weighted += weigths[k] * pos;
				}
				if (posToBGR) { // 3D coord -> BGR color
					// for 2D SOM: weighted.z == 0 -> use only G and R
					Point3 tmp = weighted;
					weighted.x = tmp.z;  // B
					weighted.y = tmp.y;  // G
					weighted.z = tmp.x;  // R
				}
				weightedPos(y, x) = weighted;
			}
		}
	}

private:
	RGB &o;
	GenSOM *som;
	Mat3& weightedPos;
	std::vector<float> weigths;
	SOMClosestN closestN;
};

// TODO: call progressUpdate
cv::Mat3f RGB::executeSOM(const multi_img &img)
{
	typedef cv::Mat_<cv::Vec<GenSOM::value_type, 3> > Mat3;

	vole::Stopwatch total("Total runtime of SOM generation");

	img.rebuildPixels(false);

	boost::shared_ptr<GenSOM> som(GenSOM::create(config.som, img));

	//  false color image
	Mat3 bgr(img.height, img.width);

	{
		vole::Stopwatch watch("False Color Image Generation");
		// compute weighted coordinates of multi_img pixels in 3D SOM
		tbb::parallel_for(tbb::blocked_range2d<int>(0, img.height, // row range
													0, img.width), // column range
						  SomRgbTbb<true>(*this, img, som.get(), bgr));

		// DEBUG: run sequentially
//		SomRgbTbb<true> somRgbTbb(*this, img, som.get(), bgr);
//		somRgbTbb(tbb::blocked_range2d<int>(0, img.height, 0, img.width));
	}


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

void RGB::setSomProgress(int percent)
{
	// assuming som training done == our progress 100%,
	// which appears to be a good estimate.
	po && po->update(percent);
}

bool RGB::progressUpdate(float percent, vole::ProgressObserver *po)
{
	if (po == NULL)
		return true;

	return po->update(percent);
}
}
