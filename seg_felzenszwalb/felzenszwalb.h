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

#ifndef FELZENSZWALB_H
#define FELZENSZWALB_H

#include "felzenszwalb2_config.h"
#include <multi_img.h>

namespace gerbil {
  namespace felzenszwalb {

	typedef std::vector<std::vector<int> > segmap;

	// disjoint-set forests using union-by-rank and path compression (sort of).
	struct uni_elt {
		int rank;
		int p;
		int size;
	};

	class universe {
	public:
		universe(int elements);
		~universe();
		int find(int x);
		void join(int x, int y);
		int size(int x) const { return elts[x].size; }
		int num_sets() const { return num; }

	private:
		uni_elt *elts;
		int num;
	};

	// graph
	struct edge {
		float w;
		int a, b;
	};

	inline bool operator<(const edge &a, const edge &b) {
		return a.w < b.w;
	}

	universe* segment_graph(int n_vertices, int n_edges, edge *edges, float c);
	std::pair<cv::Mat1i, segmap> segment_image(const multi_img &im,
	                                          const FelzenszwalbConfig &config);
  }
}

#endif
