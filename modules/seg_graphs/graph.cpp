/*
	This file was edited by Johannes Jordan <johannes.jordan@cs.fau.de>.
	The modified version is distributed to you under the terms of the
	GNU General Public License version 3, as published by the Free Software
	Foundation. You can find it here:	
	http://www.gnu.org/licenses/gpl.html

	The original work was licensed under the CeCILL license, see the original
	copyright notice below. Find the original source code here:
	http://sourceforge.net/projects/powerwatershed/
	
	The licenses are compatible; you may redistribute this file under the
	terms of one of them or both.
	
	In parts copyright(c) 2010 Johannes Jordan.
	
*/

/*Copyright ESIEE (2009)

   Author :
   Camille Couprie (c.couprie@esiee.fr)

   Contributors :
   Hugues Talbot (h.talbot@esiee.fr)
   Leo Grady (leo.grady@siemens.com)
   Laurent Najman (l.najman@esiee.fr)

   This software contains some image processing algorithms whose purpose is to be
   used primarily for research.

   This software is governed by the CeCILL license under French law and
   abiding by the rules of distribution of free software.  You can  use,
   modify and/ or redistribute the software under the terms of the CeCILL
   license as circulated by CEA, CNRS and INRIA at the following URL
   "http://www.cecill.info".

   As a counterpart to the access to the source code and  rights to copy,
   modify and redistribute granted by the license, users are provided only
   with a limited warranty  and the software's author,  the holder of the
   economic rights,  and the successive licensors  have only  limited
   liability.

   In this respect, the user's attention is drawn to the risks associated
   with loading,  using,  modifying and/or developing or reproducing the
   software by the user in light of its specific status of free software,
   that may mean  that it is complicated to manipulate,  and  that  also
   therefore means  that it is reserved for developers  and  experienced
   professionals having in-depth computer knowledge. Users are therefore
   encouraged to load and test the software's suitability as regards their
   requirements in conditions enabling the security of their systems and/or
   data to be ensured and,  more generally, to use and operate it in the
   same conditions as regards security.

   The fact that you are presently reading this means that you have had
   knowledge of the CeCILL license and that you accept its terms.
 */

#include "graph.h"
#include "graph_alg.h" // for geodesic

namespace powerwaterseg {

Graph::Graph(int width, int height) : width(width), height(height)
{
	// hardcoded for 4-connected lattice
	degree = 4;	// number of neighbors per node (pixel)
	degree2 = 2*(degree - 1); // number of neighbors of both nodes of an edge
	int edgec = (width*(height - 1)) + ((width - 1)*height);

	edges.resize(edgec);
	generateEdges();
}

void Graph::generateEdges()
{
	int x, y;
	register int M = 0;

	for (y = 0; y < (height - 1); y++) {
		for (x = 0; x < width; x++) {
			edges[M].nodes[0] = y*width + x;
			edges[M].nodes[1] = (y + 1)*width + x;
			++M;
		}
	}
	// horizontal
	for (y = 0; y < height; y++) {
		for (x = 0; x < (width - 1); x++) {
			edges[M].nodes[0] = y*width + x;
			edges[M].nodes[1] = y*width + x + 1;
			++M;
		}
	}
}

int Graph::bucket(float weight)
{
	return (int)std::floor(weight / bucketsize);
}

/* =================================================================== */
int Graph::neighbor(int i,   /* node index */
			 int k) { /* number of the desired neighbor node of node i */
/* =================================================================== */
/*
   return the index of the k_th neighbor NODE
	   5         From the top :
	 3 0 1       4: slice below,  2 : slice above
	   6
   return -1 if the neighbor is outside the image */
	int y     = i / (width);
	int x     = i % (width);

	switch (k) {
	case 1:
		return (x < width - 1 ? i + 1 : -1);
	case 3:
		return (x > 0 ? i - 1 : -1);
	case 2:
		return (y < height - 1 ? i + width : -1);
	case 4:
		return (y > 0 ? i - width : -1);
	}
	return -1; //never happens
}

/* ============================================================================= */
int Graph::neighbor_node_edge(int i,   /* node index */
					   int k) { /* number of the desired edge neighbor of node i */
/* ============================================================================== */
/* return the index of the k_th edge neighbor of the node "i"
	   4
	 3 0 1
	   2
   return -1 if the neighbor is outside the image */
	int size = width * height;
	int zp    = i % (size);
	int z     = i / (size);
	int V     = (height - 1) * width;
	int H     = (width - 1) * height;
	switch (k) {
	case 1:
		if (zp % width >= width - 1)
			return -1;
		else
			return (zp + V) - (zp / width) + z * (V + H + size);
	case 3:
		if (zp % width == 0)
			return -1;
		else
			return (zp + V) - (zp / width) - 1 + z * (V + H + size);
	case 2:
		if (zp / width >= height - 1)
			return -1;
		else
			return zp + z * (V + H + size);
	case 4:
		if (zp / width == 0)
			return -1;
		else
			return zp - width + z * (V + H + size);
	case -1:
		return i + size;
	case 0:
		return i + size * 2;
	}
	return -1; //never happens
}

/* ================================================================== */
int Graph::neighbor_edge(int i,  /* edge index */
				  int k) {  /* number of the desired neighbor of edge i */
/* =================================================================== */
/* return the index of the k_th neighbor EDGE

   %      1       _|_
   %    2 0 6     _|_          2 1      _|_|_
   %    3 0 5      |         3 0 0 6     | |
   %      4                    4 5
   %
   % return -1 if the neighbor is outside the image


   % indexing edges 2D
   %
   %    _4_ _5_
   %  0|  1|
   %   |_6_|_7_
   %  2|  3|
   %   |   |

 */
	int V = (height - 1) * width; // nb vertical edges (see compute_edges)
	if (i >= V) {
		//horizontal
		switch (k) {
		case 2:
			if ((i - V) < width - 1)
				return -1;
			else
				return ((i - V) / (width - 1) - 1) * width + ((i - V) % (width - 1));
		case 3:
			if ((i - V) % (width - 1) == 0)
				return -1;
			else
				return i - 1;
		case 4:
			if (i > (width - 1) * height + V - width)
				return -1;
			else
				return ((i - V) / (width - 1) - 1) * width + ((i - V) % (width - 1)) + width;
		case 5:
			if (i > (width - 1) * height + V - width)
				return -1;
			else
				return ((i - V) / (width - 1) - 1) * width + ((i - V) % (width - 1)) + width + 1;
		case 6:
			if ((i - V) % (width - 1) == width - 2)
				return -1;
			else
				return i + 1;
		case 1:
			if (i - V < width - 1)
				return -1;
			else
				return ((i - V) / (width - 1) - 1) * width + ((i - V) % (width - 1)) + 1;
		}
	} else { //vertical
		switch (k) {
		case 6:
			if (i % width == width - 1)
				return -1;
			else
				return (i + V) - (i / width);
		case 1:
			if (i < width)
				return -1;
			else
				return i - width;
		case 2:
			if (i % width == 0)
				return -1;
			else
				return (i + V) - (i / width) - 1;
		case 3:
			if (i % width == 0)
				return -1;
			else
				return (i + V) - (i / width) - 1 + width - 1;
		case 4:
			if (i >= V - width)
				return -1;
			else
				return i + width;
		case 5:
			if (i % width == width - 1)
				return -1;
			else
				return (i + V) - (i / width) + width - 1;
		}
	}
	return -1; //never happens
}

/* ================================================================================================= */
void Graph::color_standard_weights(const multi_img & image,
						vole::SimilarityMeasure<multi_img::Value> *distfun,
						bool geodesic) {
/* ================================================================================================== */
/* Computes weights inversely proportional to the image gradient for 2D images */

	bool gray = (image.size() == 1);

	// make sure we don't run into cache misses
	if (!gray)
		image.rebuildPixels();
	const multi_img::Band& band0 = image[0];

	if (gray) {
		max_weight = 255.f; // we will never adjust it
	} else {
		max_weight = 0.f;
	}

	// import edge coloring from image
	for (unsigned int i = 0; i < edges.size(); i++) {
		// hackish! rewrite edges code! width == number of columns
		cv::Point coord1(edges[i].nodes[0] % width, edges[i].nodes[0] / width),
		          coord2(edges[i].nodes[1] % width, edges[i].nodes[1] / width);

		if (gray) {
			edges[i].weight = std::abs(band0(coord1) - band0(coord2));
		} else {
			const multi_img::Pixel &p1 = image(coord1), &p2 = image(coord2);
			edges[i].weight = (float)distfun->getSimilarity(p1, p2, coord1, coord2);
			max_weight = std::max<float>(edges[i].weight, max_weight);
		}
	}

	bucketsize = max_weight / 250.f; // TODO: make this user-selectable

	if (!geodesic) {
		for (unsigned int i = 0; i < edges.size(); i++)
			edges[i].weight = max_weight - edges[i].weight;

	/* RESULT:
		edges[].weight: regular weights (maxw - X)
		edges[].norm_weight: unset
	*/
		return;
	}

	for (unsigned int i = 0; i < edges.size(); i++)
		edges[i].norm_weight = max_weight - edges[i].weight;

	/* fill in initial weights for edges originating from seeds */
	float *seeds_function = (float*)calloc(edges.size(), sizeof(float));
	for (unsigned int j = 0; j < seeds.size(); j++) {
		for (int k = 1; k <= degree; k++) {
			int n = neighbor_node_edge(seeds[j].first, k);
			if (n != -1)
				seeds_function[n] = edges[n].norm_weight;
		}
	}

	gageodilate_union_find(seeds_function);
	free(seeds_function);

	/* RESULT:
		edges[].weight: reconstructed weights
		edges[].norm_weight: regular weights (maxw - X)
	*/
}

}
