/*
	Copyright(c) 2013 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "rgb.h"

#ifdef WITH_EDGE_DETECT
#include <som_trainer.h>
#endif

#include <stopwatch.h>
#include <multi_img.h>
#include <progress_observer.h>
#include <opencv2/highgui/highgui.hpp>
#include <tbb/parallel_for.h>
#include <iostream>
#include <vector>
#include <algorithm>

#ifdef WITH_BOOST
#include <shared_data.h>
#endif

namespace gerbil {

RGB::RGB()
 : Command(
		"rgb",
		config,
		"Johannes Jordan",
		"johannes.jordan@informatik.uni-erlangen.de")
{}

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
	SharedMultiImgPtr src = boost::any_cast<SharedMultiImgPtr>(input["multi_img"]);
	if ((**src).empty())
		assert(false);

	cv::Mat3f bgr = execute(**src, po);
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

	switch (config.algo) {
	case COLOR_XYZ:
		bgr = src.bgr(); // updateProgress is not called in XYZ calculation
		break;
	case COLOR_PCA:
		bgr = executePCA(src, po);
		break;
	case COLOR_SOM:
#ifdef WITH_EDGE_DETECT
		bgr = executeSOM(src, po);
		break;
#else
		std::cerr << "FATAL: SOM functionality missing!" << std::endl;
#endif
	default:
		true;
	}

	return bgr;
}

cv::Mat3f RGB::executePCA(const multi_img& src, vole::ProgressObserver *po)
{
	multi_img pca3 = src.project(src.pca(3));

	bool cont = progressUpdate(70.0f, po); // TODO: values
	if (!cont) return cv::Mat3f();

	if (config.pca_stretch)
		pca3.data_stretch_single(0., 1.);
	else
		pca3.data_rescale(0., 1.);

	cont = progressUpdate(80.0f, po); // TODO: values
	if (!cont) return cv::Mat3f();

//	bgr = pca3.Mat();
	// green: component 1, red: component 2, blue: component 3
	std::vector<cv::Mat> vec(3);
	vec[0] = pca3[2]; vec[1] = pca3[0]; vec[2] = pca3[1];
	cv::Mat3f bgr;
	cv::merge(vec, bgr);
	return bgr;
}

static bool sortpair(std::pair<double, cv::Point> i,
					 std::pair<double, cv::Point> j) {
	return (i.first < j.first);
}

struct SOMTBB {

	SOMTBB (const multi_img& src, const SOM *som, const std::vector<double>& w,
			cv::Mat3f &dst)
	: src(src), som(som), w(w), dst(dst) {}

	void operator()(const tbb::blocked_range<int>& r) const
	{
		for (int i = r.begin(); i != r.end(); ++i) {
			std::vector<std::pair<double, cv::Point> > coords =
					som->closestN(src.atIndex(i), w.size());
			std::sort(coords.begin(), coords.end(), sortpair);
			cv::Point3d avg(0., 0., 0.);
			for (int j = 0; j < coords.size(); ++j) {
				cv::Point3d c(coords[j].second.x,
							  coords[j].second.y / som->getWidth(),
							  coords[j].second.y % som->getWidth());
				avg += w[j] * c;
			}
			cv::Vec3f &pixel = dst(i / dst.cols, i % dst.cols);
			pixel[0] = (float)(avg.x);
			pixel[1] = (float)(avg.y);
			pixel[2] = (float)(avg.z);
		}
	}
	
	const multi_img& src;
	const SOM *som;
	const std::vector<double> &w;
	cv::Mat3f &dst;
};

#ifdef WITH_EDGE_DETECT
// TODO: call progressUpdate
cv::Mat3f RGB::executeSOM(const multi_img& img, vole::ProgressObserver *po)
{
	img.rebuildPixels(false);
	config.som.hack3d = true;
	config.som.height = config.som.width * config.som.width;

	SOM *som = SOMTrainer::train(config.som, img);
	if (som == NULL)
		return cv::Mat3f();

	if (config.som.output_som) {
		multi_img somimg = som->export_2d();
		somimg.write_out(config.output_file + "_som");
	}

	vole::Stopwatch watch("False Color Image Generation");

	cv::Mat3f bgr(img.height, img.width);
	cv::Mat3f::iterator it = bgr.begin();
	if (config.som_depth < 2) {
		for (unsigned int i = 0; it != bgr.end(); ++i, ++it) {
			cv::Point n = som->identifyWinnerNeuron(img.atIndex(i));
			(*it)[0] = n.x;
			(*it)[1] = n.y / som->getWidth();
			(*it)[2] = n.y % som->getWidth();
		}
		bgr /= config.som.width;
	} else {
		int N = config.som_depth;

		// calculate weights including normalization (RGB space width)
		std::vector<double> weights;
		if (config.som_linear) {
			for (int i = 0; i < N; ++i)
				weights.push_back(1./(double)(N * config.som.width));
		} else {
			/* each weight is half of the preceeding weight in the ranking
			   examples; N=2: 0.667, 0.333; N=4: 0.533, 0.267, 0.133, 0.067 */
			double normalization = (double)((1 << (N)) - 1)
				                   * (double)config.som.width;
			for (int i = 0; i < N; ++i)
				weights.push_back((double)(1 << (N - i - 1)) / normalization);
		}

		// set RGB pixels
		tbb::parallel_for(tbb::blocked_range<int>(0, img.height*img.width),
						  SOMTBB(img, som, weights, bgr));
	}
	if (config.verbosity & 4) {
		multi_img somimg = som->export_2d();
		
		// SOM code assumes and sets [0..1], need to correct
		somimg.minval = img.minval;
		somimg.maxval = img.maxval;
		somimg.write_out(config.output_file + "-som");
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

bool RGB::progressUpdate(float percent, vole::ProgressObserver *po)
{
	if (po == NULL)
		return true;

	return po->update(percent);
}
}
