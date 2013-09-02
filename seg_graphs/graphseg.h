#ifndef GRAPHSEG_H
#define GRAPHSEG_H

#include "graphseg_config.h"
#include <multi_img.h>

namespace vole {

class GraphSeg {
public:
	GraphSeg(const GraphSegConfig& config) : config(config) {}

	cv::Mat1b execute(const multi_img& input,
	                       const cv::Mat1b& seeds,
	                       cv::Mat1b *proba_map = 0);

private:
	static cv::Rect bbox(const cv::Mat1b& seeds);

	const GraphSegConfig &config;
};

}
#endif
