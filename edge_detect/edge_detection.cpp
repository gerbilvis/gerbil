#include "edge_detection.h"
#include <som_cache.h>

#include <imginput.h>
#include <sm_factory.h>

#include <boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <tbb/blocked_range2d.h>
#include <boost/make_shared.hpp>

namespace vole {

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
	vole::ImgInput ii(config.imgInputCfg);

	img = ii.execute();
	if (img->empty()) {
		throw std::runtime_error
				("EdgeDetection::execute: imginput module failed to read image.");
	}
	img->rebuildPixels(false);

	// load or train SOM
	boost::shared_ptr<GenSOM> genSom(GenSOM::create(config.somCfg, *img));

	// build lookup table
	boost::shared_ptr<SOMClosestN> lookup =
			boost::make_shared<SOMClosestN>(*genSom, *img, 1);

	cv::Mat1f dx = cv::Mat1f::zeros(img->height, img->width);
	cv::Mat1f dy = cv::Mat1f::zeros(img->height, img->width);

	boost::shared_ptr<SimilarityMeasure<float> >
			simfun(SMFactory<float>::spawn(config.somCfg.similarity));

	/* TODO: parallelize with TBB */

	for (int y = 1; y < img->height-1; ++y) {
		for (int x = 1; x < img->width-1; ++x) {
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
			const std::vector<float> pnw = genSom->getCoord(ranw.first->index);
			const std::vector<float> pn  = genSom->getCoord(ran.first->index);
			const std::vector<float> pne = genSom->getCoord(rane.first->index);
			const std::vector<float> pw  = genSom->getCoord(raw.first->index);
			const std::vector<float> pe  = genSom->getCoord(rae.first->index);
			const std::vector<float> psw = genSom->getCoord(rasw.first->index);
			const std::vector<float> ps  = genSom->getCoord(ras.first->index);
			const std::vector<float> pse = genSom->getCoord(rase.first->index);

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

			if (!config.absolute) {
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

} // namespace vole
