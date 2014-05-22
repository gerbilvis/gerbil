#include "edge_detection.h"
#include <som_cache.h>

#include <imginput.h>
#include <sm_factory.h>

#include <boost/filesystem.hpp>

#include <opencv2/highgui/highgui.hpp>


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

	typedef boost::shared_ptr<multi_img> multi_img_ptr;

	multi_img_ptr img;
	vole::ImgInput ii(config.imgInputCfg);

	img = ii.execute();
	if (img->empty()) {
		throw std::runtime_error("EdgeDetection::execute: imginput module failed to read image.");
	}
	img->rebuildPixels(false);

	typedef boost::shared_ptr<GenSOM> GenSOMPtr;
	GenSOMPtr genSom;

	// load or train SOM
	genSom = GenSOMPtr(GenSOM::create(config.somCfg, *img));

	typedef boost::shared_ptr<SOMClosestN> SOMClosestNPtr;
	SOMClosestNPtr  closestn = SOMClosestNPtr(new SOMClosestN(*genSom,
							*img,
							1));

	cv::Mat1f dx = cv::Mat1f::zeros(img->height, img->width);
	cv::Mat1f dy = cv::Mat1f::zeros(img->height, img->width);

	typedef boost::shared_ptr<SimilarityMeasure<float> > SimilarityMeasurePtr;
	SimilarityMeasurePtr simfun = SimilarityMeasurePtr(
				SMFactory<float>::spawn(config.somCfg.similarity));

	/* TODO: parallelize with TBB */

	for (int y = 1; y < img->height-1; ++y) {
		for (int x = 1; x < img->width-1; ++x) {
			// access closestn results
			const SOMClosestN::resultAccess ranw = closestn->closestN(cv::Point2i(x-1,y-1));
			const SOMClosestN::resultAccess ran = closestn->closestN(cv::Point2i(x,y-1));
			const SOMClosestN::resultAccess rane = closestn->closestN(cv::Point2i(x+1,y-1));
			const SOMClosestN::resultAccess rae = closestn->closestN(cv::Point2i(x+1,y));
			const SOMClosestN::resultAccess rase = closestn->closestN(cv::Point2i(x+1,y+1));
			const SOMClosestN::resultAccess ras = closestn->closestN(cv::Point2i(x,y+1));
			const SOMClosestN::resultAccess rasw = closestn->closestN(cv::Point2i(x-1,y+1));
			const SOMClosestN::resultAccess raw = closestn->closestN(cv::Point2i(x-1,y));

			// euclidian SOM locations of neurons (n-D som)
			const std::vector<float> pnw = genSom->getCoord(ranw.first->index);
			const std::vector<float> pn = genSom->getCoord(ran.first->index);
			const std::vector<float> pne = genSom->getCoord(rane.first->index);
			const std::vector<float> pe = genSom->getCoord(rae.first->index);
			const std::vector<float> pse = genSom->getCoord(rase.first->index);
			const std::vector<float> ps = genSom->getCoord(ras.first->index);
			const std::vector<float> psw = genSom->getCoord(rasw.first->index);
			const std::vector<float> pw = genSom->getCoord(raw.first->index);

			std::vector<float> hv = std::vector<float>(pnw.size());
			std::vector<float> vv = std::vector<float>(pnw.size());
			std::vector<float> nv = std::vector<float>(pnw.size()); // null vector

			for (size_t i=0; i < pnw.size(); ++i) {
				hv[i] = .25 * (pnw[i] - 2*pw[i] - psw[i]) -
						.25 * (pne[i] - 2*pe[i] - pse[i]);

				vv[i] = .25 * (pnw[i] - 2*pn[i] - pne[i]) -
						.25 * (psw[i] - 2*ps[i] - pse[i]);
				nv[i] = 0.f;
			}
			dx[y][x] = simfun->getSimilarity(nv, hv);
			dy[y][x] = simfun->getSimilarity(nv, vv);

			/* TODO: use absolute position hack for signed output. */
		}
	}


	cv::Mat dxout;
	dx.convertTo(dxout, CV_8UC1, 255.);
	std::string dxfname = (boost::filesystem::path(config.outputDir) / "dx.png").native();
	cv::imwrite(dxfname, dxout);

	cv::Mat dyout;
	dy.convertTo(dyout, CV_8UC1, 255.);
	std::string dyfname = (boost::filesystem::path(config.outputDir) / "dy.png").native();
	cv::imwrite(dyfname, dyout);

	return 0;
}

void EdgeDetection::printShortHelp() const
{
	std::cout << "Edge detection in multispectral images using SOM." << std::endl;
}

void EdgeDetection::printHelp() const
{
	std::cout << "Edge detection in multispectral images using SOM." << std::endl;

}

} // namespace vole
