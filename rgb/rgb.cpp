/*
	Copyright(c) 2013 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "rgb.h"

#ifdef WITH_EDGE_DETECT
#include <sm_config.h>
#include <sm_factory.h>
#include <som_trainer.h>
#endif

#include <stopwatch.h>
#include <multi_img.h>
#include <opencv2/highgui/highgui.hpp>
#include <tbb/parallel_for.h>
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
 : Command(
		"rgb",
		config,
		"Johannes Jordan",
		"johannes.jordan@informatik.uni-erlangen.de"),
   calcPo(new SOMProgressObserver(this)),
   srcimg(NULL),
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
#ifdef WITH_EDGE_DETECT
		bgr = executeSOM(src);
		break;
#else
		std::cerr << "FATAL: SOM functionality missing!" << std::endl;
#endif
	default:
		true;
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

#ifdef WITH_EDGE_DETECT
struct SOMTBB {

	SOMTBB (const multi_img& src, SOM *som, const std::vector<double>& w,
			cv::Mat3f &dst)
		: src(src), som(som), weight(w), dst(dst) {}

	void operator()(const tbb::blocked_range<int>& r) const
	{
		// (we just need the correct length,
		// values will be overwritten in closestN)
		std::vector<std::pair<double, SOM::iterator> > coords(
			weight.size(), std::make_pair(0.0, som->end()));
		for (int i = r.begin(); i != r.end(); ++i) {
			som->closestN(src.atIndex(i), coords);

			// set color values
			cv::Point3d avg(0., 0., 0.);
			for (int j = 0; j < coords.size(); ++j) {
				cv::Point3d pos = coords[j].second.get3dPos();
				avg += weight[j] * pos;
			}
			cv::Vec3f &pixel = dst(i / dst.cols, i % dst.cols);
			pixel = som->getColor(avg);
		}
	}

	const multi_img& src;
	SOM *const som;
	const std::vector<double> &weight;
	cv::Mat3f &dst;
};

// TODO: call progressUpdate
cv::Mat3f RGB::executeSOM(const multi_img &img)
{
	vole::Stopwatch total("Total runtime of SOM generation");

	img.rebuildPixels(false);

	SOM *som = SOMTrainer::train(config.som, img, abortFlag, calcPo);
	if (som == NULL)
		return cv::Mat3f();

	if (config.som.output_som || config.som.verbosity >= 3) {
		multi_img somimg = som->export_2d();
		somimg.write_out(config.output_file + "_som");
	}

	cv::Mat3f bgr(img.height, img.width);
	cv::Mat3f::iterator it = bgr.begin();

	int N = config.som_depth;

	// calculate weights
	std::vector<double> weights;
	// (for N == 1, the lower weight calculation would divide by zero)
	if (config.som_linear || N == 1) {
		for (int i = 0; i < N; ++i)
			weights.push_back(1./(double)N);
	} else {
		/* each weight is half of the preceeding weight in the ranking
		   examples; N=2: 0.667, 0.333; N=4: 0.533, 0.267, 0.133, 0.067 */
		double normalization = (double)((1 << N) - 1); // 2^N - 1
		for (int i = 0; i < N; ++i) {
			// (2^[N-1..0]) / normalization
			weights.push_back((double)(1 << (N - i - 1)) / normalization);
		}
	}

	// set RGB pixels
	{
		vole::Stopwatch watch("False Color Image Generation");
		tbb::parallel_for(tbb::blocked_range<int>(0, img.height*img.width),
		                  SOMTBB(img, som, weights, bgr));
	}

	delete som;
	return bgr;
}
#endif

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
