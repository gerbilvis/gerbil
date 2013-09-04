/*
Copyright (C) 2006 Pedro Felzenszwalb, 2012 Johannes Jordan

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#include "felzenszwalb.h"
#include <sm_factory.h>
#include <cstdlib>
#include <boost/unordered_map.hpp>

namespace gerbil {
  namespace felzenszwalb {

void equalizeHist(cv::Mat_<float> &target, int bins);

std::pair<cv::Mat1i, segmap> segment_image(const multi_img &im,
							 const FelzenszwalbConfig &config)
{
	vole::SimilarityMeasure<multi_img::Value> *distfun;
	distfun = vole::SMFactory<multi_img::Value>::spawn(config.similarity);
	assert(distfun);

	int width = im.width;
	int height = im.height;

	// build graph
	edge *edges = new edge[width*height*4];
	std::vector<float> weights;
	
	int num = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			cv::Point coord1(x, y);
			const multi_img::Pixel &p1 = im(coord1);

			if (x < width-1) {
				edges[num].a = y * width + x;
				edges[num].b = y * width + (x+1);
				cv::Point coord2(x+1, y);
				const multi_img::Pixel &p2 = im(coord2);
				weights.push_back((float)distfun->getSimilarity(p1, p2, coord1, coord2));
				num++;
			}

			if (y < height-1) {
				edges[num].a = y * width + x;
				edges[num].b = (y+1) * width + x;
				cv::Point coord2(x, y+1);
				const multi_img::Pixel &p2 = im(coord2);
				weights.push_back((float)distfun->getSimilarity(p1, p2, coord1, coord2));
				num++;
			}

			if ((x < width-1) && (y < height-1)) {
				edges[num].a = y * width + x;
				edges[num].b = (y+1) * width + (x+1);
				cv::Point coord2(x+1, y+1);
				const multi_img::Pixel &p2 = im(coord2);
				weights.push_back((float)distfun->getSimilarity(p1, p2, coord1, coord2));
				num++;
			}

			if ((x < width-1) && (y > 0)) {
				edges[num].a = y * width + x;
				edges[num].b = (y-1) * width + (x+1);
				cv::Point coord2(x+1, y-1);
				const multi_img::Pixel &p2 = im(coord2);
				weights.push_back((float)distfun->getSimilarity(p1, p2, coord1, coord2));
				num++;
			}
		}
	}
	
	if (config.eqhist) {
		cv::Mat_<float> tmp(weights);
		equalizeHist(tmp, 20000);
	}
	
	int i;
	for (i = 0; i < weights.size(); ++i)
		edges[i].w = weights[i];
	

	// segment
	universe *u = segment_graph(width*height, num, edges, config.c);

	// post process small components
	for (i = 0; i < num; i++) {
	int a = u->find(edges[i].a);
	int b = u->find(edges[i].b);
	if ((a != b) &&
	    ((u->size(a) < config.min_size) || (u->size(b) < config.min_size)))
		u->join(a, b);
	}
	delete[] edges;

	// create index map and sets of segments
	cv::Mat1i indices(height, width);
	segmap segments;
	// provide mapping between old universe index and new sequential index
	boost::unordered_map<int, int> mapping;

	cv::Mat1i::iterator it = indices.begin();
	for (int coord = 0; it != indices.end(); ++it, ++coord) {
		int index = u->find(coord);
		if (mapping.find(index) == mapping.end()) {
			mapping[index] = segments.size();
			segments.push_back(std::vector<int>(1, coord));
		} else {
			segments[mapping[index]].push_back(coord);
		}
		*it = mapping[index];
	}

	delete u;
	return std::make_pair(indices, segments);
}

void equalizeHist(cv::Mat_<float> &target, int bins)
{
	float binsf = bins - 1.f;

	/* find range, assume positive values */
	double mi, ma;
	cv::minMaxLoc(target, &mi, &ma);
	mi = 0;

	/* calculate histogram */
	cv::Mat_<float> hist;
	int histSize[] = { bins };
	float range1[] = { (float)mi, (float)ma };
	const float *ranges[] = { range1 };
	int channels[] = { 0 };

	cv::calcHist(&target, 1, channels, cv::Mat(), hist, 1, histSize, ranges,
	             true, false);
	
	// normalize
	hist *= binsf / (float)(2 * target.rows * target.cols);
	
	// compute CDF
	cv::Mat_<float> cdf;
	cv::integral(hist, cdf, CV_32F);
	
	// apply histogram equalization
	for (int y = 0; y < target.rows; ++y) {
		float *row = target[y];
		for (int x = 0; x < target.cols; ++x) {
			int coord = (int)std::ceil(row[x] * (binsf / ma) + 0.5f);
			if (coord < 0 || coord > bins) {
				std::cerr << "(" << x << ", " << y << "): " << coord << std::endl;
				row[x] = 0.f;
			} else {
				row[x] = (float)cdf(coord, 1);
			}
		}
	}
}

} } // namespace
