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
#include "graph_alg.h"
#include "sorting.h"
#include <algorithm>
#include <cstdlib>

namespace powerwaterseg {

/*================================================*/
void Graph::element_link_geod_dilate(int      n,
							  int      p,
							  int      *Fth) {
/*================================================*/
	int r = element_find(n, Fth);

	if (r != p) {
		if ((edges[r].norm_weight == edges[p].norm_weight)
		     || (edges[p].norm_weight >= edges[r].weight)) {
			Fth[r] = p;
			edges[p].weight = std::max<float>(edges[r].weight, edges[p].weight);
		} else
			edges[p].weight = max_weight;
	}
}


/* ==================================================================================================== */
void Graph::gageodilate_union_find(float *F) {
/* ===================================================================================================== */
/* reconstruction by dilation of g under f.  Union-find method described by Luc Vicent. */
	int  p, i, n;
	bool * Mrk = (bool*)calloc(edges.size(), sizeof(bool));
	int  * Fth = (int*)malloc(edges.size() * sizeof(int)); // indices for sorting

	// Es : E sorted by decreasing weights
	int * Es = (int*)malloc(edges.size() * sizeof(int));

	for (unsigned int k = 0; k < edges.size(); k++) {
		Fth[k] = k;
		edges[k].weight = F[k];
		F[k]   = edges[k].norm_weight;
		Es[k]  = k;
	}

	sortRange(F, Es, edges.size());

	/* first pass */
	for (int k = (int)edges.size() - 1; k >= 0; k--) {
		p = Es[k];
		for (i = 1; i <= degree2; i += 1) {
			n = neighbor_edge(p, i);
			if (n != -1)
				if (Mrk[n])
					element_link_geod_dilate(n, p, Fth);

			Mrk[p] = true;
		}
	}

	/* second pass */
	for (unsigned int k = 0; k < edges.size(); k++) {
		p = Es[k];
		if (Fth[p] == p) { // p is root
			if (edges[p].weight == max_weight)
				edges[p].weight = edges[p].norm_weight;
		} else
			edges[p].weight = edges[Fth[p]].weight;
	}

	free(Es);
	free(Mrk);
	free(Fth);
}

}
