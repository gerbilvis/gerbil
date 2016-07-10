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
 
#include "graph_alg.h"
#include "sorting.h"
#include "graph.h"
#include <limits>
#include <set>
#include <stack>
#include <cstdio>

namespace seg_graphs {

#define epsilon std::numeric_limits<float>::epsilon()
#define SIZE_MAX_PLATEAU 1000000

/*=====================================================================================*/
cv::Mat1b Graph::MSF_Prim() {
/*=====================================================================================*/
/* returns a segmentation performed by Prim's algorithm for Maximum Spanning Forest computation */
	int u, v, x, y, x_1, y_1;
	int N = width*height;                       /* nb of nodes */
	int M = edges.size(); /* number of edges */

	// G is a labeling map
	unsigned char * G = (unsigned char*)calloc(N, sizeof(unsigned char));
	for (size_t i = 0; i < seeds.size(); i++)
		G[seeds[i].first] = seeds[i].second;

	std::vector<bool> indics(M, false);
	std::multiset<Score> L;

	// initialize at seed points
	size_t i = 0;
	for (u = 0; u < M && i < seeds.size(); u++) {
		if (u == seeds[i].first) {
			for (x = 1; x <= degree; x++) {
				y = neighbor_node_edge(seeds[i].first, x);
				if ((y != -1) && (!indics[y])) {
					Score entry = { max_weight - edges[y].weight, y };
					L.insert(entry);
					indics[y] = true;
				}
			}
			i++;
		}
	}

	while (!L.empty()) {
		u = L.begin()->index;
		L.erase(L.begin());
		x = edges[u].nodes[0];
		y = edges[u].nodes[1];
		if (G[x] > G[y]) {
			std::swap(x, y);
		}
		if ((std::min<unsigned char>(G[x], G[y]) == 0)
		 && (std::max<unsigned char>(G[x], G[y]) > 0)) {
			G[x] = G[y];
			for (int i = 1; i <= degree; i++) {
				v = neighbor_node_edge(x, i);
				if ((v != -1) && (!indics[v])) {
					x_1 = edges[v].nodes[0];
					y_1 = edges[v].nodes[1];
					if   ((std::min<unsigned char>(G[x_1], G[y_1]) == 0)
					    &&(std::max<unsigned char>(G[x_1], G[y_1]) >  0)) {
						Score entry = { max_weight - edges[v].weight, v };
						L.insert(entry);
						indics[v] = true;
					}
				}
			}
		}
		indics[u] = false;
	}

	cv::Mat1b ret(height, width);
	cv::MatIterator_<uchar> it = ret.begin();

	for (int i = 0; i < N; i++, it++)
		*it = G[i] - 1;

	free(G);
	return ret;
}



/*=====================================================================*/
cv::Mat1b Graph::MSF_Kruskal() {
/*=====================================================================*/
/*returns a segmentation performed by Kruskal's algorithm for Maximum Spanning Forest computation*/
	int k, x, y, e1, e2;
	int N, M;

	N = width * height; /* number of vertices */
	M = edges.size(); /*number of edges*/

	int * Mrk = (int*)calloc(N, sizeof(int));
	if (Mrk == NULL) {
		fprintf(stderr, "lMSF() : malloc failed\n"); exit(0);
	}
	for (size_t i = 0; i < seeds.size(); i++) {
		Mrk[seeds[i].first] = seeds[i].second;
	}
	int * Rnk = (int*)calloc(N, sizeof(int));
	if (Rnk == NULL) {
		fprintf(stderr, "lMSF() : malloc failed\n"); exit(0);
	}
	int * Fth = (int*)malloc(N * sizeof(int));
	if (Fth == NULL) {
		fprintf(stderr, "lMSF() : malloc failed\n"); exit(0);
	}
	for (k = 0; k < N; k++) {
		Fth[k] = k;
	}

	// create indices sorted by edge weight, descending
	int *Es = new int[M];
	{
		std::vector<Score> sorter(M);
		for (k = 0; k < M; k++) {
			Score entry = { edges[k].weight, k };
			sorter[k] = entry;
		}
		std::sort(sorter.begin(), sorter.end());
		for (k = 0; k < M; k++) {
			Es[k] = sorter[M - k - 1].index;
		}
	}

	int nb_arete = 0;
	int e_max, root;

	/* beginning of main loop */

	int cpt_aretes = 0;

	while (nb_arete < N - (int)seeds.size()) {
		e_max = Es[cpt_aretes];
		// printf("%d \n", e_max);
		cpt_aretes = cpt_aretes + 1;
		e1         = edges[e_max].nodes[0];
		e2         = edges[e_max].nodes[1];
		x          = element_find(e1, Fth);
		y          = element_find(e2, Fth);

		if ((x != y) && (!(Mrk[x] >= 1 && Mrk[y] >= 1))) {
			root     = element_link(x, y, Rnk, Fth);
			nb_arete = nb_arete + 1;
			if (Mrk[x] >= 1)
				Mrk[root] = Mrk[x];
			else if (Mrk[y] >= 1)
				Mrk[root] = Mrk[y];
		}
	}

	//building the map (find the root vertex of each tree)
	int * Map2 = (int *)malloc(N * sizeof(int));
	int * Map  = (int *)malloc(N * sizeof(int));
	for (int i = 0; i < N; i++)
		Map2[i] = element_find(i, Fth);

	bool* Fullseeds = (bool *)calloc(N, sizeof(bool));
	for (size_t i = 0; i < seeds.size(); i++) {
		Fullseeds[seeds[i].first] = 1;
		Map[seeds[i].first]       = (int)seeds[i].second;
	}

	for (int i = 0; i < N; i++)
		Mrk[i] = false;

	for (size_t i = 0; i < seeds.size(); i++) {
		std::stack<int> lifo;
		lifo.push(seeds[i].first);
		while (!lifo.empty()) {
			x = lifo.top();
			lifo.pop();
			Mrk[x] = true;
			for (k = 1; k <= degree; k++) {
				y = neighbor(x, k);
				if (y != -1) {
					if (Map2[y] == Map2[seeds[i].first] && Fullseeds[y] != 1 &&
						Mrk[y] == false) {
						lifo.push(y);
						Map[y] = seeds[i].second;
						Mrk[y] = true;
					}
				}
			}
		}
	}

	cv::Mat1b ret(height, width);
	cv::MatIterator_<uchar> it = ret.begin();

	for (int i = 0; i < N; i++, it++)
		*it = Map[i] - 1;


	free(Mrk);
	free(Rnk);
	free(Fullseeds);
	delete[] Es;
	free(Map);
	free(Map2);
	free(Fth);
	return ret;
}

/*========================================================================================================*/
void memory_allocation_PW(bool ** indic_E,     /* indicator for edges */
						  bool ** indic_P,     /* indicator for edges */
						  int  ** indic_VP,    /* indicator for nodes */
						  int  ** Rnk,         /* array needed for union-find efficiency */
						  int  ** Fth,         /* array for storing roots of merged nodes trees */
						  int  ** local_seeds, /* array for storing the index of seeded nodes of a plateau */
						  int  ** LCVP,        /* list of vertices of a plateau */
						  int  ** Es,          /* array of sorted edges according their reconstructed weight */
						  int  ** NEs,         /* array of sorted edges according their weight */
						  int  N,              /* number of vertices */
						  int  M) {            /* number of edges*/
/*==========================================================================================================*/
#undef F_NAME
#define F_NAME    "memory_allocation_PW"

	*indic_E = (bool*)calloc(M, sizeof(bool));
	if (*indic_E == NULL) {
		fprintf(stderr, "%s: calloc indic_E failed\n", F_NAME); exit(0);
	}
	*indic_P = (bool*)calloc(M, sizeof(bool));
	if (*indic_P == NULL) {
		fprintf(stderr, "%s: calloc indic_P failed\n", F_NAME); exit(0);
	}
	*indic_VP = (int*)calloc(N, sizeof(int));
	if (*indic_VP == NULL) {
		fprintf(stderr, "%s: calloc indic_VP failed\n", F_NAME); exit(0);
	}
	*Rnk = (int*)calloc(N, sizeof(int));
	if (*Rnk == NULL) {
		fprintf(stderr, "%s: malloc Rnk failed\n", F_NAME); exit(0);
	}
	*Fth = (int*)malloc(N * sizeof(int));
	if (*Fth == NULL) {
		fprintf(stderr, "%s: malloc Fth failed\n", F_NAME); exit(0);
	}

	*local_seeds = (int*)malloc(N * sizeof(int));
	if (*local_seeds == NULL) {
		fprintf(stderr, "%s: malloc local_seeds failed\n", F_NAME); exit(0);
	}

	*LCVP = (int*)malloc(N * sizeof(int)); // vertices of a plateau.
	if (*LCVP == NULL) {
		fprintf(stderr, "%s: malloc LCVP failed\n", F_NAME); exit(0);
	}

	*Es = (int*)malloc(M * sizeof(int));
	if (*Es == NULL) {
		fprintf(stderr, "%s: malloc Es failed\n", F_NAME); exit(0);
	}

	*NEs = (int*)malloc(M * sizeof(int));
	if (*NEs == NULL) {
		fprintf(stderr, "%s: malloc NEs failed\n", F_NAME); exit(0);
	}
}


/*===================================================================================*/
void merge_node(int      e1,          /* index of node 1 */
				int      e2,          /* index of node 2 */
				int      *Rnk,       /* array needed for union-find efficiency */
				int      *Fth,        /* array for storing roots of merged nodes trees */
				double** proba,    /* array for storing the result x */
				int      nb_labels) { /* nb of labels */
/*===================================================================================*/
/* update the result, Rnk and Fth arrays when 2 nodes are merged */
	int k, re1, re2;
	re1 = element_find(e1, Fth);
	re2 = element_find(e2, Fth);

	if ((re1 != re2) && (!(proba[0][re1] >= 0 && proba[0][re2] >= 0))) {
		element_link(re1, re2, Rnk, Fth);
		if (proba[0][re2] >= 0 && proba[0][re1] < 0)
			for (k = 0; k < nb_labels - 1; k++)
				proba[k][re1] = proba[k][re2];
		else if (proba[0][re1] >= 0 && proba[0][re2] < 0)
			for (k = 0; k < nb_labels - 1; k++)
				proba[k][re2] = proba[k][re1];
	}
}

/*==================================================================================================================*/
cv::Mat1b Graph::PowerWatershed_q2(bool geodesic, cv::Mat1b *out_proba) {
/*==================================================================================================================*/
/*returns the result x of the energy minimization : min_x lim_p_inf sum_e_of_E w_{ij}^p |x_i-x_j|^2 */
#undef F_NAME
#define F_NAME    "PowerWatershed_q2"

	int    i, j, k, x, y, e1, e2, re1, re2, p, xr;
	int    N = width*height; // number of vertices
	int    M = edges.size(); // number of edges
	double val;
	int    argmax;
	int    nb_vertices, e_max, Ne_max, nb_edges, Nnb_edges;
	bool     success, different_seeds;
	int curbucket;
	std::stack<int> lifo;         /* stack for plateau labeling */
	std::vector<int> lcp;         /* list of the edges belonging to a plateau */
	bool     *indic_E;
	bool     *indic_P;
	int      *indic_VP;
	int      *Rnk;
	int      *Fth;
	int      *local_seeds;
	int      *LCVP;
	int      *Es;
	int      *NEs;

	memory_allocation_PW(&indic_E, &indic_P, &indic_VP, &Rnk, &Fth,
						 &local_seeds, &LCVP, &Es, &NEs, N,
						 M);

	double **proba = (double **)malloc((max_label - 1) * sizeof(double*));
	for (i = 0; i < max_label - 1; i++) {
		proba[i] = (double *)malloc(N * sizeof(double));
		for (j = 0; j < N; j++)
			proba[i][j] = -1;
	}
	int **edgesLCP = (int**)malloc(2 * sizeof(int*));
	if (edgesLCP == NULL) {
		fprintf(stderr, "%s: malloc edgesLCP failed\n", F_NAME); exit(0);
	}
	for (k = 0; k < 2; k++) {
		edgesLCP[k] = (int*)malloc(M * sizeof(int));
		if (edgesLCP[k] == NULL) {
			fprintf(stderr, "%s: malloc edgesLCP failed\n", F_NAME); exit(0);
		}
	}

	// initialize probabilities for seed points
	for (size_t i = 0; i < seeds.size(); i++)
		for (j = 0; j < max_label - 1; j++) {
			if (seeds[i].second == j + 1)
				proba[j][seeds[i].first] = 1;
			else
				proba[j][seeds[i].first] = 0;
		}

	for (k = 0; k < N; k++)
		Fth[k] = k;

	double** local_labels =
		(double**)malloc((max_label - 1) * sizeof(double*));
	if (local_labels == NULL) {
		fprintf(stderr, "%s: malloc local_labels failed\n", F_NAME); exit(0);
	}
	for (i = 0; i < max_label - 1; i++) {
		local_labels[i] = (double*)malloc(N * sizeof(double));
		if (local_labels[i] == NULL) {
			fprintf(stderr, "%s: malloc local_labels failed\n", F_NAME); exit(0);
		}
	}

	float* sorted_weights = (float*)malloc(M * sizeof(float));
	if (sorted_weights == NULL) {
		fprintf(stderr, "%s: malloc sorted_weights failed\n", F_NAME); exit(0);
	}

	for (k = 0; k < M; k++) {
		sorted_weights[k] = edges[k].weight;
		Es[k] = k;
	}

	// sort in descending order
	sortRange(sorted_weights, Es, M, true);

	int cpt_aretes  = 0;
	int Ncpt_aretes = 0;

	// beginning of main loop
	while (cpt_aretes < M) {
		do {
			e_max      = Es[cpt_aretes];
			cpt_aretes = cpt_aretes + 1;
			if (cpt_aretes == M)
				break;
		} while (indic_E[e_max] == true);

		if (cpt_aretes == M)
			break;

		//1. Computing the edges of the plateau LCP linked to the edge e_max
		lifo.push(e_max);
		indic_P[e_max] = true;
		indic_E[e_max] = true;
		lcp.push_back(e_max);
		nb_vertices = 0;
		nb_edges    = 0;
		curbucket   = bucket(edges[e_max].weight);

		// 2. putting the edges and vertices of the plateau into arrays
		while (!lifo.empty()) {
			x = lifo.top();
			lifo.pop();
			e1  = edges[x].nodes[0]; e2 = edges[x].nodes[1];
			re1 = element_find(e1, Fth);
			re2 = element_find(e2, Fth);
			if (proba[0][re1] < 0 || proba[0][re2] < 0) {
				if (indic_VP[e1] == 0) {
					LCVP[nb_vertices] = e1;
					nb_vertices++;
					indic_VP[e1] = 1;
				}
				if (indic_VP[e2] == 0) {
					LCVP[nb_vertices] = e2;
					nb_vertices++;
					indic_VP[e2] = 1;
				}
				edgesLCP[0][nb_edges] = e1;
				edgesLCP[1][nb_edges] = e2;
				NEs[nb_edges]         = x;

				nb_edges++;
			}

			for (k = 1; k <= degree2; k++) {
				y = neighbor_edge(x, k);
				if (y != -1) {
//					std::cerr << wmax << "\t" << edges[y].weight;
//					if ((indic_P[y] == false) && (edges[y].weight == wmax)) {
					if ((indic_P[y] == false) &&
						(bucket(edges[y].weight) == curbucket)) {
//						std::cerr << "\t***";
						indic_P[y] = true;
						lifo.push(y);
						lcp.push_back(y);
						indic_E[y] = true;
					}
//					std::cerr << std::endl;
				}
			}
		}
		for (j = 0; j < nb_vertices; j++)
			indic_VP[LCVP[j]] = 0;
		for (size_t j = 0; j < lcp.size(); j++)
			indic_P[lcp[j]] = false;

		// 3. If e_max belongs to a plateau
		if (nb_edges > 0) {
			// 4. Evaluate if there are differents seeds on the plateau
			p = 0;
			different_seeds = false;

			for (i = 0; i < max_label - 1; i++) {
				val = -0.5;
				for (j = 0; j < nb_vertices; j++) {
					x  = LCVP[j];
					xr = element_find(x, Fth);
					if ((fabs(proba[i][xr] - val) > epsilon)&&(proba[i][xr] >= 0)) {
						p++; val = proba[i][xr];
					}
				}
				if (p >= 2) {
					different_seeds = true;
					break;
				} else
					p = 0;
			}

			if (different_seeds == true) {
				// 5. Sort the edges on the plateau according to their (normal) weight
				if (geodesic) {
					for (k = 0; k < nb_edges; k++)
						sorted_weights[k] = edges[NEs[k]].weight;
				} else {
					for (k = 0; k < nb_edges; k++)
						sorted_weights[k] = edges[NEs[k]].norm_weight;
				}

				// sort in descending order
				sortRange(sorted_weights, NEs, nb_edges, true);

				// Merge nodes for edges of real max weight
				nb_vertices = 0;
				Nnb_edges   = 0;
				for (Ncpt_aretes = 0; Ncpt_aretes < nb_edges; Ncpt_aretes++) {
					Ne_max = NEs[Ncpt_aretes];
					e1     = edges[Ne_max].nodes[0];
					e2     = edges[Ne_max].nodes[1];
/*					if (( geodesic && edges[Ne_max].weight != wmax) ||
						(!geodesic && edges[Ne_max].norm_weight != wmax)) {*/
					if (( geodesic && bucket(edges[Ne_max].weight) != curbucket) ||
						(!geodesic && bucket(edges[Ne_max].norm_weight) != curbucket)) {
						merge_node(e1, e2, Rnk, Fth, proba, max_label);
					} else {
						re1 = element_find(e1, Fth);
						re2 = element_find(e2, Fth);
						if ((re1 != re2) &&
							((proba[0][re1] < 0 || proba[0][re2] < 0))) {
							if (indic_VP[re1] == 0) {
								LCVP[nb_vertices] = re1;
								nb_vertices++;
								indic_VP[re1] = 1;
							}
							if (indic_VP[re2] == 0) {
								LCVP[nb_vertices] = re2;
								nb_vertices++;
								indic_VP[re2] = 1;
							}
							edgesLCP[0][ Nnb_edges] = re1;
							edgesLCP[1][ Nnb_edges] = re2;
							Nnb_edges++;
						}
					}
				}
				for (i = 0; i < max_label - 1; i++) {
					k = 0;
					for (j = 0; j < nb_vertices; j++) {
						xr = LCVP[j];
						if (proba[i][xr] >= 0) {
							local_labels[i][k] = proba[i][xr];
							local_seeds[k]     = xr;
							k++;
						}
					}
				}

				// 6. Execute Random Walker on plateaus

				if (nb_vertices < SIZE_MAX_PLATEAU)
					success = RandomWalker(edgesLCP, Nnb_edges, LCVP, indic_VP,
						 nb_vertices, local_seeds, local_labels, k, max_label, proba);
				if ((nb_vertices >= SIZE_MAX_PLATEAU) || (success == false)) {
					printf(
						"Oversized plateau (%d vertices,%d edges), ignored by RW.\n",
						nb_vertices, Nnb_edges);
					for (j = 0; j < Nnb_edges; j++) {
						e1 = edgesLCP[0][j];
						e2 = edgesLCP[1][j];
						merge_node(e1, e2, Rnk, Fth, proba, max_label);
					}
				}

				for (j = 0; j < nb_vertices; j++)
					indic_VP[LCVP[j]] = 0;
			} else    { // if different seeds = false
					// 7. Merge nodes for edges of max weight
				for (j = 0; j < nb_edges; j++) {
					e1 = edgesLCP[0][j];
					e2 = edgesLCP[1][j];
					merge_node(e1, e2, Rnk, Fth, proba, max_label);
				}
			}
		}
		lcp.clear();
	} // end main loop

	//building the final proba map (find the root vertex of each tree)
	for (i = 0; i < N; ++i) {
		j  = i;
		xr = i;
		while (Fth[i] != i) {
			i  = xr;
			xr = Fth[i];
		}
		for (k = 0; k < max_label - 1; ++k)
			proba[k][j] = proba[k][i];
		i = j;
	}

	//writing results

	double maxi;
	cv::Mat1b ret(height, width);
	cv::MatIterator_<uchar> it = ret.begin();
	for (j = 0; j < N; ++j, ++it) {
		maxi = 0; argmax = 0; val = 1;
		for (k = 0; k < max_label - 1; ++k) {
			if (proba[k][j] > maxi) {
				maxi   = proba[k][j];
				argmax = k;
			}
			val = val - proba[k][j];
		}
		if (val > maxi)
			argmax = k;
		*it = argmax;
	}

	if (out_proba) {
		out_proba->create(height, width);
		it = out_proba->begin();
		for (j = 0; j < N; ++j, ++it)
			*it = (unsigned char)(255 - 255 * proba[0][j]);
	}

	for (i = 0; i < 2; i++)
		free(edgesLCP[i]);
	free(edgesLCP);

	free(Rnk);
	free(local_seeds);
	for (i = 0; i < max_label - 1; i++)
		free(local_labels[i]);
	free(local_labels);

	free(LCVP);
	free(Es);
	free(NEs);
	free(indic_E);
	free(indic_VP);
	free(indic_P);
	free(Fth);
	for (i = 0; i < max_label - 1; i++)
		free(proba[i]);
	free(proba);
	free(sorted_weights);

	return ret;
}

}
