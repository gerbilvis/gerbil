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

// #define PREPROCESSING
#endif

#include <stopwatch.h>
#include <multi_img.h>
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


/* Forward progress information from SOMTrainer to RGB command. */
class SOMProgressObserver : public vole::ProgressObserver {
public:
	SOMProgressObserver(RGB *rgbCmd) : rgbCmd(rgbCmd) {}
	virtual bool update(int percent) { rgbCmd->setSomProgress(percent); }
private:
	RGB *const rgbCmd;
};


RGB::RGB()
 : Command(
		"rgb",
		config,
		"Johannes Jordan",
		"johannes.jordan@informatik.uni-erlangen.de"),
   calcPo(new SOMProgressObserver(this))
{}

RGB::~RGB()
{
	delete calcPo;
}

int RGB::execute()
{
	multi_img::ptr src = vole::ImgInput(config.input).execute();
	if (src->empty())
		return 1;

#ifdef PREPROCESSING
	// (preprocessing always changes the input)
	if (config.som.verbosity >= 3)
#else
	if (config.som.verbosity && config.input.gradient)
#endif
	{
		vole::ImgInputConfig inputConf = config.input;
		inputConf.gradient = false;		
		orig_img = vole::ImgInput(inputConf).execute();
		orig_img->rebuildPixels(false);
	}

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
template<typename T>
void writeDensityMap(std::string filename, cv::Point res,
	const std::vector<T> &x, const std::vector<T> &y)
{
	assert(x.size() == y.size());

	typename std::vector<T>::const_iterator itX, itY;
	cv::Point_<T> min(std::numeric_limits<T>::max(),
	                  std::numeric_limits<T>::max());
	cv::Point_<T> max(std::numeric_limits<T>::min(),
	                  std::numeric_limits<T>::min());

	for (itX = x.begin(); itX != x.end(); ++itX)
	{
		min.x = std::min(min.x, *itX);
		max.x = std::max(max.x, *itX);
	}
	for (itY = y.begin(); itY != y.end(); ++itY)
	{
		min.y = std::min(min.y, *itY);
		max.y = std::max(max.y, *itY);
	}

	int maxHistVal = 0;
	cv::Mat_<int> hist = cv::Mat_<int>::zeros(res.y, res.x);
	for (itX = x.begin(), itY = y.begin(); itX != x.end(); ++itX, ++itY)
	{
		cv::Point2d pos(*itX, *itY); // min - max
		pos -= min;                  //   0 - (max-min)
		pos.x /= max.x - min.x;      //   0 - 1
		pos.y /= max.y - min.y;
		pos.x *= res.x;              //   0 - res
		pos.y *= res.y;
		cv::Point idx = pos;

		// clamp to [0 - res[
		if (idx.x < 0) idx.x = 0;
		else if (idx.x >= res.x) idx.x = res.x - 1;
		if (idx.y < 0) idx.y = 0;
		else if (idx.y >= res.y) idx.y = res.y - 1;

		int cnt = ++hist(idx);
		if (cnt > maxHistVal) maxHistVal = cnt;
	}

	std::ofstream metafile((filename + ".meta").c_str());
	if (!metafile.is_open())
	{
		std::cerr << "Could not open meta file " << filename << "." << std::endl;
		return;
	}
	metafile << "set xrange [" << min.x << ":" << max.x << "]" << std::endl;
	metafile << "set yrange [" << min.y << ":" << max.y << "]" << std::endl;
	metafile << "set size ratio 1" << std::endl;
	metafile << "set cbrange [0:" << maxHistVal << "]" << std::endl;
	metafile.close();
	
	std::ofstream file((filename + ".data").c_str());
	if (!file.is_open())
	{
		std::cerr << "Could not open data file " << filename << "." << std::endl;
		return;
	}

	// outer loop has to be x because of gnuplot format!
	for (int idxX = 0; idxX < res.x; ++idxX)
	{
		for (int idxY = 0; idxY < res.y; ++idxY)
		{
			double xLoc = min.x + (double)idxX / (double)res.x * (double)(max.x - min.x);
			double yLoc = min.y + (double)idxY / (double)res.y * (double)(max.y - min.y);
			file << xLoc << "\t" << yLoc << "\t" << hist(idxY, idxX) << std::endl;
		}
		file << std::endl;
	}
	file.close();
}

// TODO: schlechte lastverteilung
struct SOMTBB {

	SOMTBB (const multi_img& src, SOM *som, const std::vector<double>& w,
			cv::Mat3f &dst, cv::Mat1f &stddevs, cv::Mat3d &avg_coords)
		: src(src), som(som), weight(w), dst(dst), stddevs(stddevs), avg_coords(avg_coords) {}

	void operator()(const tbb::blocked_range<int>& r) const
	{
		// (we just need the correct length,
		// values will be overwritten in closestN)
		std::vector<std::pair<double, SOM::iterator> > coords(
			weight.size(), std::make_pair(0.0, som->end()));
		for (int i = r.begin(); i != r.end(); ++i) {
			som->closestN(src.atIndex(i), coords);

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
			pixel = som->getColor(avg);

			// save avg for statistic calculation
			if (!avg_coords.empty())
			{
				cv::Vec3d &pixel = avg_coords(i / dst.cols, i % dst.cols);
				pixel[0] = avg.x;
				pixel[1] = avg.y;
				pixel[2] = avg.z;
			}
		}
	}

	const multi_img& src;
	SOM *const som;
	const std::vector<double> &weight;
	cv::Mat3f &dst;
	cv::Mat1f &stddevs;
	cv::Mat3d &avg_coords;
};

// TODO: call progressUpdate
cv::Mat3f RGB::executeSOM(const multi_img &input_img)
{
	vole::Stopwatch total("Total runtime of SOM generation");

#ifdef PREPROCESSING
	// This is an experiment to ignore / reduce the impact of the length of pixel vectors
	// to use it, change the img parameter to input_img and outcomment the following
	// IMPORTANT for statistics: also outcommend && config.input.gradient) in the constructor!
	// the parameter config.similarity.measure = MOD_SPEC_ANGLE should also be added
	// we need a non-const instance
	multi_img imgCopy = input_img;

	//const multi_img::Value max_len = std::sqrt(img.size() * (5 * 5));
	for (int y = 0; y < imgCopy.height; ++y)
	for (int x = 0; x < imgCopy.width;  ++x)
	{
		multi_img::Pixel p = imgCopy(y, x);
		multi_img::Value sumOfSquares = 0;
		for (multi_img::Pixel::const_iterator pit = p.begin(); pit != p.end(); ++pit)
		{
			multi_img::Value val = *pit;
			sumOfSquares += val * val;
		}
		multi_img::Value len = std::sqrt(sumOfSquares);

		//multi_img::Value factor = len > 0 ? max_len / len : std::numeric_limits<double>::infinity();
		//if (factor < 1) // do not scale up short vectors, because they are very noisy anyways
		multi_img::Value factor = len > 0 ? log(len + 1) / len : std::numeric_limits<double>::infinity();
		if (factor < std::numeric_limits<double>::infinity())
		{
			for (multi_img::Pixel::iterator pit = p.begin(); pit != p.end(); ++pit)
			{
				*pit *= factor;
			}
			imgCopy.setPixel(y, x, p);
		}
	}
	const multi_img &img = imgCopy;
#else
	const multi_img &img = input_img;
#endif

	img.rebuildPixels(false);

	SOM *som = SOMTrainer::train(config.som, img, calcPo);
	if (som == NULL)
		return cv::Mat3f();

	if (config.som.output_som) {
		multi_img somimg = som->export_2d();
		somimg.write_out(config.output_file + "_som");
	}
	else if (config.som.verbosity >= 3) {
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

	cv::Mat1f stddevs;
	cv::Mat3d avg_coords;
	if (config.som.verbosity >= 3 && N > 1) // (there is no variance for N = 1!)
	{
		stddevs = cv::Mat1f(bgr.rows, bgr.cols);
		avg_coords = cv::Mat3d(bgr.rows, bgr.cols);
	}

	// set RGB pixels
	{
		vole::Stopwatch watch("False Color Image Generation");
		tbb::parallel_for(tbb::blocked_range<int>(0, img.height*img.width),
		                  SOMTBB(img, som, weights, bgr, stddevs, avg_coords));
	}

	if (config.som.verbosity >= 3 && N > 1)
	{
		vole::Stopwatch watch("Stddev Data Generation");
		std::ofstream file("bmu_stddev.data");
		if (!file.is_open())
			std::cerr << "Could not open stddev data file." << std::endl;
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

	if (config.som.verbosity >= 3)
	{
		std::cout << "Creating inter-neuron distance plots." << std::endl;

		cv::Point plotResolution(500, 500);
		std::vector<double> xValsEucl, yValsEucl, xValsSpec, yValsSpec;

		vole::SMConfig smconf_euclidean;
		smconf_euclidean.measure = vole::EUCLIDEAN;
		vole::SimilarityMeasure<multi_img::Value> *euclidean;
		euclidean = vole::SMFactory<multi_img::Value>::spawn(smconf_euclidean);

		vole::SMConfig smconf_specAngle;
		smconf_specAngle.measure = vole::MOD_SPEC_ANGLE;
		vole::SimilarityMeasure<multi_img::Value> *specAngle;
		specAngle = vole::SMFactory<multi_img::Value>::spawn(smconf_specAngle);

		// L2 / euclidean between neurons
		som->getNeuronDistancePlot(euclidean, xValsEucl, yValsEucl);
		writeDensityMap("euclidean_euclidean", plotResolution,
		                xValsEucl, yValsEucl);

		// spectral angle between neurons
		som->getNeuronDistancePlot(specAngle, xValsSpec, yValsSpec);
		writeDensityMap("euclidean_specAngle", plotResolution,
		                xValsSpec, yValsSpec);

		// clear arrays
		xValsEucl.clear();
		yValsEucl.clear();
		xValsSpec.clear();
		yValsSpec.clear();
		std::cout << "Creating inter-pixel distance plots." << std::endl;

		// comparison between position distance in SOM with orig image pixels
		const int diffVals = 40;
		xValsEucl.reserve(img.width * img.height * diffVals);
		yValsEucl.reserve(img.width * img.height * diffVals);
		xValsSpec.reserve(img.width * img.height * diffVals);
		yValsSpec.reserve(img.width * img.height * diffVals);
		cv::Mat_<int> shuffledY(1, img.width * img.height * diffVals);
		cv::Mat_<int> shuffledX(1, img.width * img.height * diffVals);
		cv::RNG rng(config.som.seed);

		// generate random sequence of the input x,y range
		rng.fill(shuffledY, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(img.height));
		rng.fill(shuffledX, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(img.width));

		cv::MatConstIterator_<int> itY = shuffledY.begin();
		cv::MatConstIterator_<int> itX = shuffledX.begin();

		// Old version calculated inter-neuron distance only on the winner neuron
		//std::cout << "Creating winner-neuron cache. This may take a while..." << std::endl;

		// to avoid the long preload calculation, one could save
		// the BMU-iterators in the SOMTBB task (only the best one)
		//SOM::Cache *cache = som->createCache(img.height, img.width);
		//{
		//	vole::Stopwatch watch("Runtime for Preloading the Cache");
		//	cache->preload(img);
		//}

		std::cout << "Calculating distances to " << diffVals << " random neighbours per pixel..." << std::endl;

		for (int y = 0; y < img.height; ++y)
		for (int x = 0; x < img.width;  ++x)
		{
			cv::Point p(x, y);
			//SOM::iterator iter = cache->getWinnerNeuron(p);
			//cv::Point3d pos = iter.get3dPos();
			const cv::Vec3d &vecPos = avg_coords(p);
			cv::Point3d pos(vecPos[0], vecPos[1], vecPos[2]);

			for (int i = 0; i < diffVals; ++i)
			{
				cv::Point p2(*itX++, *itY++);
				// make sure that the points are different
				while (p == p2) {
					p2 = cv::Point(rng(img.width), rng(img.height));
				}
				//SOM::iterator iter2 = cache->getWinnerNeuron(p2);
				//cv::Point3d pos2 = iter2.get3dPos();
				const cv::Vec3d &vecPos2 = avg_coords(p2);
				cv::Point3d pos2(vecPos2[0], vecPos2[1], vecPos2[2]);

				cv::Point3d d = pos - pos2;
				double neuronPositionDistance = std::sqrt(d.dot(d));
				xValsEucl.push_back(neuronPositionDistance);
				xValsSpec.push_back(neuronPositionDistance);

				const multi_img &no_grad_img = orig_img ? *orig_img : img;
				double origValueDistEucl = euclidean->getSimilarity(
					no_grad_img(p.y, p.x),
					no_grad_img(p2.y, p2.x));
				yValsEucl.push_back(origValueDistEucl);
				double origValueDistSpec = specAngle->getSimilarity(
					no_grad_img(p.y, p.x),
					no_grad_img(p2.y, p2.x));
				yValsSpec.push_back(origValueDistSpec);
			}
		}

		std::cout << "Writing data to files..." << std::endl;

		writeDensityMap("euclidean_origEuclidean", plotResolution,
		                xValsEucl, yValsEucl);
		writeDensityMap("euclidean_origSpecAngle", plotResolution,
		                xValsSpec, yValsSpec);

		delete euclidean;
		delete specAngle;
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
