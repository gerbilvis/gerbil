/*
	Copyright(c) 2011 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "graphseg.h"
#include "graph_alg.h"
#include "graph.h"

#include <sm_factory.h>
#ifdef WITH_SOM
#include <gensom.h>
#include <som_distance.h>
#endif

#include <stopwatch.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cstdio>
#include <algorithm>

namespace seg_graphs {

cv::Mat1b GraphSeg::execute(const multi_img& input,
                                  const cv::Mat1b& seeds,
                                  cv::Mat1b *proba_map) {
	Stopwatch running_time("Total Running Time");
	cv::Mat1b output;
	int i;

	if ((seeds.cols != input.width)||(seeds.rows != input.height)) {
		std::cerr << "ERROR: Seed file dimensions do not match image dimensions!"
		          << std::endl;
		return output; // which is empty so far
	}

	Graph graph(seeds.cols, seeds.rows);

	/* extract seeds */
	cv::Mat1b::const_iterator it;
	if (config.multi_seed == true) {		// multilabel seed image
		graph.max_label = 0;
		for (i = 0, it = seeds.begin(); it < seeds.end(); ++i, ++it) {
			if (*it > 0) {
				graph.seeds.push_back(std::make_pair(i, *it));
				if (*it > graph.max_label)
					graph.max_label = *it;
			}
		}
	} else {
		graph.max_label = 2;
		for (i = 0, it = seeds.begin(); it < seeds.end(); ++i, ++it) {
			if (*it > 192) {
				graph.seeds.push_back(std::make_pair(i, 1));
			} else if (*it < 64) {
				graph.seeds.push_back(std::make_pair(i, 2));
			}
		}
	}

	// edge weights
	similarity_measures::SimilarityMeasure<multi_img::Value> *distfun;
#ifdef WITH_SOM
	boost::shared_ptr<som::GenSOM> som; // create in this scope for survival
	if (!config.som_similarity) {
		distfun = similarity_measures::SMFactory<multi_img::Value>
				::spawn(config.similarity);
	} else {
		input.rebuildPixels();
		som = boost::shared_ptr<som::GenSOM>(
					som::GenSOM::create(config.som, input));
		distfun = new som::SOMDistance<multi_img::Value>(*som, input);
	}
#else
	distfun = SMFactory<multi_img::Value>::spawn(config.similarity);
#endif

	assert(distfun);

	Stopwatch watch;
	if (config.algo == WATERSHED2) {
		/* Kruskal & RW on plateaus multiseeds linear time */

		graph.color_standard_weights(input, distfun, true);
		watch.print_reset("Graph coloring");
		// for PW, color_standard_weights is always called with geodesic = true
		output = graph.PowerWatershed_q2(config.geodesic, proba_map);
		watch.print("Segmentation");
	} else {
		graph.color_standard_weights(input, distfun, config.geodesic);
		watch.print_reset("Graph coloring");
		if (config.algo == KRUSKAL) { // Kruskal
			output = graph.MSF_Kruskal();
		} else if (config.algo == PRIM) { // Prim RB tree
			output = graph.MSF_Prim();
		}
		watch.print("Segmentation");
	}

	delete distfun;

	return output;
}

cv::Rect GraphSeg::bbox(const cv::Mat1b &seeds)
{
	int left = seeds.cols, right = 0, top = seeds.rows, bot = 0;
	for (int y = 0; y < seeds.rows; ++y) {
		const uchar* row = seeds[y];
		for (int x = 0; x < seeds.cols; ++x) {
			if (row[x] > 63 && row[x] < 193)
				continue;
			left = std::min(left, x);
			right = std::max(right, x);
			top = std::min(top, y);
			bot = std::max(bot, y);
		}
	}
	return cv::Rect(left, top, right - left + 1, bot - top + 1);
}

}
