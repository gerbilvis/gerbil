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
#include <fstream>
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

#ifdef WITH_EDGE_DETECT
// TODO: schlechte lastverteilung
struct SOMTBB {

	SOMTBB (const multi_img& src, SOM *som, const std::vector<double>& w,
			cv::Mat3f &dst, vole::somtype somtype, cv::Mat1f &stddevs)
		: src(src), som(som), weight(w), dst(dst), somtype(somtype), stddevs(stddevs) { }

	void operator()(const tbb::blocked_range<int>& r) const
	{
		for (int i = r.begin(); i != r.end(); ++i) {
			std::vector<std::pair<double, SOM::iterator> > coords =
					som->closestN(src.atIndex(i), weight.size());

			// Calculate stdandard deviation between positions of the BMUs
			if (!stddevs.empty())
			{
				// calculate mean
				cv::Point3d mean(0.0, 0.0, 0.0);
				for (int j = 0; j < coords.size(); ++j) {
					mean += coords[j].second.get3dPos();
				}
				mean *= 1.0 / (double)coords.size();

				// calculate stddev
				double stddev = 0;
				for (int j = 0; j < coords.size(); ++j) {
					cv::Point3d sampleDev = coords[j].second.get3dPos() - mean;
					stddev += sampleDev.dot(sampleDev);
				}
				stddev = std::sqrt(stddev / coords.size());

				stddevs[i / stddevs.cols][i % stddevs.cols] = (float)stddev;
			}

			// set color values
			cv::Point3d avg(0., 0., 0.);
			for (int j = 0; j < coords.size(); ++j) {
				cv::Point3d pos = coords[j].second.get3dPos();
				avg += weight[j] * pos;
			}
			cv::Vec3f &pixel = dst(i / dst.cols, i % dst.cols);
			if (somtype == vole::SOM_CUBE)
			{
				// rgb
				pixel[0] = (float)(avg.x);
				pixel[1] = (float)(avg.y);
				pixel[2] = (float)(avg.z);
			}
			else
			{
				// hsv
				const double PI = 3.141592653589793;

				// calculate hsv values
				double h, s, v;
				h = std::atan2(avg.y, avg.x) + PI; // only x and y affect hue, h has range 0 - 2*PI
				s = std::sqrt(avg.x * avg.x + avg.y * avg.y); // only x and y affect saturation
				if (avg.z > 0) s /= 0.5 * avg.z; // normalize by radius at height avg.z of the cone
				else s = 0;
				v = avg.z; // the cone has a height of 1

				// convert hsv2rgb - see german wikipedia article
				double f, p, q, t;
				int h_i = (int)(h / (PI / 3));
				f = (h / (PI / 3)) - h_i;
				p = v * (1 - s);
				q = v * (1 - s * f);
				t = v * (1 - s * (1-f));

				switch (h_i)
				{
				case 0:
				case 6:
					pixel = cv::Vec3f(v, t, p);
					break;
				case 1:
					pixel = cv::Vec3f(q, v, p);
					break;
				case 2:
					pixel = cv::Vec3f(p, v, t);
					break;
				case 3:
					pixel = cv::Vec3f(p, q, v);
					break;
				case 4:
					pixel = cv::Vec3f(t, p, v);
					break;
				case 5:
					pixel = cv::Vec3f(v, p, q);
					break;
				}

				if (pixel[0] < 0) pixel[0] = 0;
				if (pixel[1] < 0) pixel[1] = 0;
				if (pixel[2] < 0) pixel[2] = 0;
				if (pixel[0] > 1) pixel[0] = 1;
				if (pixel[1] > 1) pixel[1] = 1;
				if (pixel[2] > 1) pixel[2] = 1;
			}
		}
	}

	const multi_img& src;
	SOM *const som;
	const std::vector<double> &weight;
	cv::Mat3f &dst;
	cv::Mat1f &stddevs;
	vole::somtype somtype;
};

// TODO: call progressUpdate
cv::Mat3f RGB::executeSOM(const multi_img& img, vole::ProgressObserver *po)
{
	img.rebuildPixels(false);

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

	int N = config.som_depth;

	// calculate weights including normalization (RGB space width)
	std::vector<double> weights;
	if (config.som_linear) {
		for (int i = 0; i < N; ++i)
			weights.push_back(1./(double)N);
	} else {
		/* each weight is half of the preceeding weight in the ranking
		   examples; N=2: 0.667, 0.333; N=4: 0.533, 0.267, 0.133, 0.067 */
		double normalization = (double)((1 << N) - 1); // 2^N - 1
		for (int i = 0; i < N; ++i)
			weights.push_back((double)(1 << (N - i - 1)) / normalization); // (2^[N-1..0]) / normalization
	}

	// keep the rgb values in the range of 0..1
	if (config.som.type == vole::SOM_CUBE)
		for (int i = 0; i < N; ++i)
			weights[i] /= config.som.sidelength;

	// set RGB pixels
	cv::Mat1f stddevs;
	if (config.som.verbosity >= 3)
		stddevs = cv::Mat1f(bgr.rows, bgr.cols);

	tbb::parallel_for(tbb::blocked_range<int>(0, img.height*img.width),
					  SOMTBB(img, som, weights, bgr, config.som.type, stddevs));

	if (!stddevs.empty())
	{
		std::ofstream file("bmu_stddev.data");
		if (!file.is_open())
			std::cerr << "Could not open stddev data file" << std::endl;
		else
		{
			float min_stddev = std::numeric_limits<float>::max(), max_stddev = 0;
			double mean_stddev = 0;
			for (int y = 0; y < stddevs.rows; ++y)
			{
				for (int x = 0; x < stddevs.cols; ++x)
				{
					float val = stddevs[y][x];
					file << val << std::endl;

					min_stddev = std::min(min_stddev, val);
					mean_stddev += val;
					max_stddev = std::max(max_stddev, val);
				}
			}
			mean_stddev /= stddevs.rows * stddevs.cols;
			file.close();
			std::cout << "Stats about stddev of distance between BMU positions in the SOM:" << std::endl;
			std::cout << "Min:  " << min_stddev  << std::endl;
			std::cout << "Mean: " << mean_stddev << std::endl;
			std::cout << "Max:  " << max_stddev  << std::endl;
		}

		double min, max;
		cv::minMaxLoc(stddevs, &min, &max);
		cv::imwrite("stddev.png", stddevs * (255.0 / max));
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
