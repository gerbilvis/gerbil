/*
	Copyright(c) 2011 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef GRAPH_H
#define GRAPH_H

#include <similarity_measure.h>
#include <multi_img.h>
#include <vector>

namespace powerwaterseg {

struct Edge {
	int nodes[2];
	float weight, norm_weight;
};

struct Graph {
	Graph(int width, int height);

	/* mesh indexing */
	int neighbor_node_edge(int indice, int num);
	int neighbor(int indice, int num);
	int neighbor_edge(int indice, int num);

	/* buckets (for PowerWatershed_q2) */
	int bucket(float weight);

	/* graph coloring */
	void color_standard_weights(const multi_img &image, vole::SimilarityMeasure<multi_img::Value> *distfun, bool geodesic);

	/* graph algorithms: geodesic reconstruction */
	void element_link_geod_dilate(int n, int p, int *Fth);
	void gageodilate_union_find(float *F);

	/* graph algorithms: spanning forest & power watersheds */
	cv::Mat1b MSF_Prim();
	cv::Mat1b MSF_Kruskal();
	cv::Mat1b PowerWatershed_q2(bool geodesic, cv::Mat1b *out_proba);


	std::vector<Edge> edges;
	float max_weight;
	float bucketsize;
	int degree, degree2;
	int width, height;

	std::vector<std::pair<int, unsigned char> > seeds;
	int max_label;

private:
	// build mesh. called by constructor
	void generateEdges();
};

}

#endif // GRAPH_H
