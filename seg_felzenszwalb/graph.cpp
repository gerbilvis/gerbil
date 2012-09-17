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
#include <algorithm>
#include <cmath>

namespace gerbil {
  namespace felzenszwalb {

// threshold function
#define THRESHOLD(size, c) (c/size)

//disjoint-set forest functions

universe::universe(int elements) {
  elts = new uni_elt[elements];
  num = elements;
  for (int i = 0; i < elements; i++) {
	elts[i].rank = 0;
	elts[i].size = 1;
	elts[i].p = i;
  }
}

universe::~universe() {
  delete [] elts;
}

int universe::find(int x) {
  int y = x;
  while (y != elts[y].p)
	y = elts[y].p;
  elts[x].p = y;
  return y;
}

void universe::join(int x, int y) {
  if (elts[x].rank > elts[y].rank) {
	elts[y].p = x;
	elts[x].size += elts[y].size;
  } else {
	elts[x].p = y;
	elts[y].size += elts[x].size;
	if (elts[x].rank == elts[y].rank)
	  elts[y].rank++;
  }
  num--;
}

/*
 * Segment a graph
 *
 * Returns a disjoint-set forest representing the segmentation.
 *
 * num_vertices: number of vertices in graph.
 * num_edges: number of edges in graph
 * edges: array of edges.
 * c: constant for treshold function.
 */
universe* segment_graph(int num_vertices, int num_edges, edge *edges, float c)
{
  // sort edges by weight
  std::sort(edges, edges + num_edges);

  // make a disjoint-set forest
  universe *u = new universe(num_vertices);

  // init thresholds
  float *threshold = new float[num_vertices];
  for (int i = 0; i < num_vertices; i++)
    threshold[i] = THRESHOLD(1,c);

  // for each edge, in non-decreasing weight order...
  for (int i = 0; i < num_edges; i++) {
    edge *pedge = &edges[i];
    
    // components conected by this edge
    int a = u->find(pedge->a);
    int b = u->find(pedge->b);
    if (a != b) {
      if ((pedge->w <= threshold[a]) &&
	  (pedge->w <= threshold[b])) {
	u->join(a, b);
	a = u->find(a);
	threshold[a] = pedge->w + THRESHOLD(u->size(a), c);
      }
    }
  }

  // free up
  delete[] threshold;
  return u;
}

} } // namespace
