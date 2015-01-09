#ifndef MEANSHIFT_H
#define MEANSHIFT_H

#include "meanshift_config.h"
#include "mfams.h"
#include "progress_observer.h"
#include <multi_img.h>
#ifdef WITH_SEG_FELZENSZWALB
#include <felzenszwalb.h>
#endif

#include <boost/make_shared.hpp>

namespace seg_meanshift {

class MeanShift {

public:
	/** Result of MeanShift calculation.
	 *
	 *  If the MeanShift calculation failed or was aborted, modes is empty.
	 */
	struct Result {
		Result()
			: labels(new cv::Mat1s()),
			modes(new std::vector<multi_img::Pixel>())
		{}
		Result(const std::vector<multi_img::Pixel>& m,
			   const cv::Mat1s& l)
		{ setModes(m); setLabels(l); }

		// default copy and assignment OK

		void setModes(const std::vector<multi_img::Pixel>& in) {
			modes = boost::make_shared<std::vector<multi_img::Pixel> >(in);
		}
		void setLabels(const cv::Mat1s& in) {
			labels = boost::make_shared<cv::Mat1s>(in);
		}
		void printModes() const {
			if (modes->size() < 2) {
				std::cout << "No modes found!" << std::endl;
				return;
			}
			std::cout << modes->size() << " distinct modes found:" << std::endl;
			for (size_t i = 0; i < modes->size(); ++i) {
				multi_img::Pixel mode = modes->at(i);
				for (size_t d = 0; d < mode.size(); ++d)
					std::cout << mode[d] << "\t";
				std::cout << std::endl;
			}
		}
		boost::shared_ptr<cv::Mat1s> labels;
		boost::shared_ptr<std::vector<multi_img::Pixel> > modes;
	};

	MeanShift(const MeanShiftConfig& config) : config(config) {}

	KLResult findKL(const multi_img& input, ProgressObserver *po = 0);
	Result execute(const multi_img& input, ProgressObserver *po = 0,
	               vector<double> *bandwidths = 0,
	               const multi_img& spinput = multi_img());

#ifdef WITH_SEG_FELZENSZWALB
	static std::vector<FAMS::Point> prepare_sp_points(const FAMS &fams,
									  const seg_felzenszwalb::segmap &map);
	static void cleanup_sp_points(std::vector<FAMS::Point> &points);
	static cv::Mat1s segmentImageSP(const FAMS &fams, const cv::Mat1i &lookup);
#endif

	// terrible hack superpixel sizes
	std::vector<int> spsizes;

private:
	const MeanShiftConfig &config;

};

}

#endif
