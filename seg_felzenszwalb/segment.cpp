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
#include <cstdlib>

namespace gerbil {
  namespace felzenszwalb {

std::pair<cv::Mat1i, segmap> segment_image(const multi_img &im,
							 vole::SimilarityMeasure<multi_img::Value> *distfun,
							 float c, int min_size)
{
	int width = im.width;
	int height = im.height;

	// build graph
	edge *edges = new edge[width*height*4];
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
				edges[num].w = (float)distfun->getSimilarity(p1, p2, coord1, coord2);
				num++;
			}

			if (y < height-1) {
				edges[num].a = y * width + x;
				edges[num].b = (y+1) * width + x;
				cv::Point coord2(x, y+1);
				const multi_img::Pixel &p2 = im(coord2);
				edges[num].w = (float)distfun->getSimilarity(p1, p2, coord1, coord2);
				num++;
			}

			if ((x < width-1) && (y < height-1)) {
				edges[num].a = y * width + x;
				edges[num].b = (y+1) * width + (x+1);
				cv::Point coord2(x+1, y+1);
				const multi_img::Pixel &p2 = im(coord2);
				edges[num].w = (float)distfun->getSimilarity(p1, p2, coord1, coord2);
				num++;
			}

			if ((x < width-1) && (y > 0)) {
				edges[num].a = y * width + x;
				edges[num].b = (y-1) * width + (x+1);
				cv::Point coord2(x+1, y-1);
				const multi_img::Pixel &p2 = im(coord2);
				edges[num].w = (float)distfun->getSimilarity(p1, p2, coord1, coord2);
				num++;
			}
		}
	}

	// segment
	universe *u = segment_graph(width*height, num, edges, c);

	// post process small components
	for (int i = 0; i < num; i++) {
	int a = u->find(edges[i].a);
	int b = u->find(edges[i].b);
	if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
		u->join(a, b);
	}
	delete[] edges;

	// create index map and sets of segments
	cv::Mat1i indices(height, width);
	segmap segments;

	for (int y = 0; y < height; y++) {
		int *row = indices[y];
		for (int x = 0; x < width; x++) {
			int coord = y * width + x;
			int index = u->find(coord);
			if (segments.find(index) == segments.end())
				segments[index] = std::vector<int>(1, coord);
			else
				segments[index].push_back(coord);
			row[x] = index;
		}
	}

	delete u;
	return std::make_pair(indices, segments);
}

} } // namespace
