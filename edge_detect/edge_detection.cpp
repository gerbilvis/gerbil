#include "edge_detection.h"
#include <som_cache.h>

#include <imginput.h>
#include <sm_factory.h>
#include <stopwatch.h>

#include <boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <boost/make_shared.hpp>

using namespace som;

namespace edge_detect {

EdgeDetection::EdgeDetection()
	: Command("edge_detect",
			  config,
			  "Johannes Jordan",
			  "johannes.jordan@informatik.uni-erlangen.de")
{
}

EdgeDetection::~EdgeDetection()
{
}

class EdgeTBB {
public:
	EdgeTBB(const GenSOM *som, const SOMClosestN *lookup,
			similarity_measures::SimilarityMeasure<float> *simfun,
			cv::Mat1f &dx, cv::Mat1f &dy, bool absolute)
		: som(som), lookup(lookup), simfun(simfun), dx(dx), dy(dy),
		  absolute(absolute)
	{}

	void operator()(const tbb::blocked_range2d<int> &r) const
	{
		for (int y = r.rows().begin(); y < r.rows().end(); ++y) {
			for (int x = r.cols().begin(); x < r.cols().end(); ++x) {
				// access closestn results; *n*orth, *w*est, *e*ast, *s*outh,
				SOMClosestN::resultAccess ranw, ran, rane, raw, rae,
						rase, ras, rasw;
				ranw = lookup->closestN(cv::Point2i(x-1,y-1));
				ran  = lookup->closestN(cv::Point2i(x,y-1));
				rane = lookup->closestN(cv::Point2i(x+1,y-1));
				raw  = lookup->closestN(cv::Point2i(x-1,y));
				rae  = lookup->closestN(cv::Point2i(x+1,y));
				rasw = lookup->closestN(cv::Point2i(x-1,y+1));
				ras  = lookup->closestN(cv::Point2i(x,y+1));
				rase = lookup->closestN(cv::Point2i(x+1,y+1));

				// SOM locations of neurons (n-D som)
				std::vector<float> pnw = som->getCoord(ranw.first->index);
				std::vector<float> pn  = som->getCoord(ran.first->index);
				std::vector<float> pne = som->getCoord(rane.first->index);
				std::vector<float> pw  = som->getCoord(raw.first->index);
				std::vector<float> pe  = som->getCoord(rae.first->index);
				std::vector<float> psw = som->getCoord(rasw.first->index);
				std::vector<float> ps  = som->getCoord(ras.first->index);
				std::vector<float> pse = som->getCoord(rase.first->index);

				std::vector<float> west(pnw.size()), east(pnw.size());
				std::vector<float> north(pnw.size()), south(pnw.size());

				for (size_t i=0; i < pnw.size(); ++i) {
					west[i] = .25 * (pnw[i] - 2*pw[i] - psw[i]);
					east[i] = .25 * (pne[i] - 2*pe[i] - pse[i]);

					north[i] = .25 * (pnw[i] - 2*pn[i] - pne[i]);
					south[i] = .25 * (psw[i] - 2*ps[i] - pse[i]);
				}
				dx(y, x) = simfun->getSimilarity(west, east);
				dy(y, x) = simfun->getSimilarity(north, south);

				if (!absolute) {
					std::vector<float> origin(west.size(), 0.f);
					if (simfun->getSimilarity(east, origin)
						> simfun->getSimilarity(west, origin))
						dx(y, x) = -dx(y, x);
					if (simfun->getSimilarity(south, origin)
						> simfun->getSimilarity(north, origin))
						dy(y, x) = -dy(y, x);
				}
			}
		}
	}

private:
	const GenSOM *som;
	const SOMClosestN *lookup;
	similarity_measures::SimilarityMeasure<float> *simfun;
	cv::Mat1f &dx, &dy;
	bool absolute;
};

int EdgeDetection::execute()
{
	// catch this early
	if(!boost::filesystem::is_directory(config.outputDir)) {
		std::stringstream ss;
		ss << "EdgeDetection::execute: " <<
			  "Output path does not exist: '" << config.outputDir <<"'";
		throw std::runtime_error(ss.str());
	}

	multi_img::ptr img;
	imginput::ImgInput ii(config.input);

	img = ii.execute();
	if (img->empty()) {
		throw std::runtime_error
				("EdgeDetection::execute: imginput module failed to read image.");
	}
	img->rebuildPixels(false);

	// load or train SOM
	boost::shared_ptr<GenSOM> som(GenSOM::create(config.som, *img));

	Stopwatch watch("Edge Map Generation");

	// build lookup table
	boost::shared_ptr<SOMClosestN> lookup =
			boost::make_shared<SOMClosestN>(*som, *img, 1);

	cv::Mat1f dx = cv::Mat1f::zeros(img->height, img->width);
	cv::Mat1f dy = cv::Mat1f::zeros(img->height, img->width);

	boost::shared_ptr<similarity_measures::SimilarityMeasure<float> >
			simfun(similarity_measures::SMFactory<float>
				   ::spawn(config.som.similarity));

	tbb::parallel_for(tbb::blocked_range2d<int>(1, img->height - 1, // row range
												1, img->width - 1), // column range
					  EdgeTBB(som.get(), lookup.get(), simfun.get(), dx, dy,
							  config.absolute));

	std::string dxfname = (boost::filesystem::path(config.outputDir)
						   / "dx.png").native();
	cv::imwrite(dxfname, (config.absolute ? dx : (dx + 0.5f)) * 255.f);

	std::string dyfname = (boost::filesystem::path(config.outputDir)
						   / "dy.png").native();
	cv::imwrite(dyfname, (config.absolute ? dy : (dy + 0.5f)) * 255.f);

	return 0;
}

void EdgeDetection::printShortHelp() const
{
	std::cout << "Edge detection in multispectral images using SOM." << std::endl;
}

void EdgeDetection::printHelp() const
{
	std::cout << "Edge detection in multispectral images using SOM." << std::endl;
	std::cout << std::endl;
	std::cout << "Please read \"Jordan, J., Angelopoulou E.: Edge Detection in Multispectral\n"
				 "Images Using the N-Dimensional Self-Organizing Map.\" (ICIP 2011)"
			  << std::endl;
}

} // module namespace
