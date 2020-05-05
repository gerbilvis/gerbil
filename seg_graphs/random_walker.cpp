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

   Author : Camille Couprie

   Contributors :
   Leo Grady (leo.grady@siemens.com)
   Hugues Talbot (h.talbot@esiee.fr)
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

extern "C" {
	#include "csparse/cs.h"
}
#include "graph_alg.h"
#include "sorting.h"

namespace seg_graphs {

/*===============================*/
bool fill_A(cs   * A,               /* matrix A to fill */
			int  N,                 /* nb of nodes */
			int  M,                 /* nb of edges */
			int  numb_boundary,     /* nb of seeds */
			int  ** index_edges,    /* array of node index composing edges */
			bool * seeded_vertex,   /* index of seeded nodes */
			int  * indic_sparse,    /* array of index separating seeded and unseeded nodes */
			int  * nb_same_edges) { /* indicator of same edges presence */
/*===============================*/
// building matrix A (laplacian for unseeded nodes)
	int k;
	int rnz = 0;

	// fill the diagonal
	for (k = 0; k < N; k++)
		if (seeded_vertex[k] == false) {
			A->x[rnz] = indic_sparse[k]; //value
			A->i[rnz] = rnz;             //position 1
			A->p[rnz] = rnz;             //position 2
			rnz++;
		}

	int rnzs = 0;
	int rnzu = 0;

	for (k = 0; k < N; k++)
		if (seeded_vertex[k] == true) {
			indic_sparse[k] = rnzs;
			rnzs++;
		} else    {
			indic_sparse[k] = rnzu;
			rnzu++;
		}

	for (k = 0; k < M; k++) {
		if ((seeded_vertex[index_edges[0][k]] == false) &&
			(seeded_vertex[index_edges[1][k]] == false)) {
			A->x[rnz] = -nb_same_edges[k] - 1;
			A->i[rnz] = indic_sparse[index_edges[0][k]];
			A->p[rnz] = indic_sparse[index_edges[1][k]];
			rnz++;
			A->x[rnz] = -nb_same_edges[k] - 1;
			A->p[rnz] = indic_sparse[index_edges[0][k]];
			A->i[rnz] = indic_sparse[index_edges[1][k]];
			rnz++;
			k = k + nb_same_edges[k];
		}
	}
	A->nz = rnz;
	A->m  = N - numb_boundary;
	A->n  = N - numb_boundary;
	return true;
}

/*=======================================================================================*/
void fill_B(cs   * B,             /* matrix B to fill */
			int  N,               /* nb of nodes */
			int  M,               /* nb of edges */
			int  numb_boundary,   /* nb of seeds */
			int  ** index_edges,  /* array of node index composing edges */
			bool * seeded_vertex, /* index of seeded nodes */
			int  * indic_sparse,  /* array of index separating seeded and unseeded nodes */
			int  * nb_same_edges) { /* indicator of same edges presence */
/*=======================================================================================*/
// building matrix B (laplacian for seeded nodes)
	int k;
	int rnz;

	rnz = 0;
	for (k = 0; k < M; k++) {
		if (seeded_vertex[index_edges[0][k]] == true) {
			B->x[rnz] = -nb_same_edges[k] - 1;
			B->p[rnz] = indic_sparse[index_edges[0][k]];
			B->i[rnz] = indic_sparse[index_edges[1][k]];
			rnz++;
			k = k + nb_same_edges[k];
		} else if (seeded_vertex[index_edges[1][k]] == true) {
			B->x[rnz] = -nb_same_edges[k] - 1;;
			B->p[rnz] = indic_sparse[index_edges[1][k]];
			B->i[rnz] = indic_sparse[index_edges[0][k]];
			rnz++;
			k = k + nb_same_edges[k];
		}
	}

	B->nz = rnz;
	B->m  = N - numb_boundary;
	B->n  = numb_boundary;
}

/*======================================================================*/
void TriEdges(int ** index_edges,  /* array of vertices composing edges */
			  int M,               /* nb of edges */
			  int * nb_same_edges) { /* indicator of same edges presence  */
/*======================================================================*/
// sort the array of vertices composing edges by ascending node index
// and fill an indicator of same edges presence
	int i, j;
	TriRapideStochastique(index_edges[0], index_edges[1], 0, M - 1);
	i = 0;
	while (i < M) {
		j = i;
		while ((i < M - 1) && (index_edges[0][i] == index_edges[0][i + 1]))
			i++;
		if (i != j)
			TriRapideStochastique(&index_edges[1][j], &index_edges[0][j], 0,
								  i - j);
		i++;
	}
	for (i = 0; i < M; i++) {
		j = 0;
		while ((i + j < M - 1) &&
			   (index_edges[0][i + j] == index_edges[0][i + j + 1] &&
		index_edges[1][i + j] == index_edges[1][i + j + 1]))
			j++;
		nb_same_edges[i] = j;
	}
}

/*===========================================================================================*/
bool RandomWalker(int      ** index_edges,     /* list of edges */
				  int      M,                  /* number of edges */
				  int      * index,            /* list of vertices */
				  int      * indic_vertex,     /* boolean array of vertices */
				  int      N,                  /* number of vertices */
				  int      * index_seeds,      /* list of nodes that are seeded*/
				  double ** boundary_values, /* associated values for seeds (labels)*/
				  int      numb_boundary,      /* number of seeded nodes */
				  int      nb_labels,          /* number of possible different labels values */
				  double ** proba) {         /* output : solution to the Dirichlet problem */
/*===========================================================================================*/
/*
   Function RandomWalker computes the solution to the Dirichlet problem (RW potential function)
   on a general graph represented by an edge list, given boundary conditions (seeds, etc.)
 */
#undef F_NAME
#define F_NAME    "RandomWalker"
	int  i, j, k, l, v1, v2;
	bool * seeded_vertex = (bool*)calloc(N, sizeof(bool));
	int  * indic_sparse  = (int*)calloc(N, sizeof(int));
	int  * nb_same_edges = (int*)calloc(M, sizeof(int));

	// Indexing the edges, and the seeds
	for (i = 0; i < N; i++)
		indic_vertex[index[i]] = i;

	for (j = 0; j < M; j++) {
		v1 = indic_vertex[index_edges[0][j]];
		v2 = indic_vertex[index_edges[1][j]];
		if (v1 < v2) {
			for (i = 0; i < 2; i++) {
				index_edges[i][j] = indic_vertex[index_edges[i][j]];
				indic_sparse[index_edges[i][j]]++;
			}
		} else    {
			index_edges[1][j] = v1;
			index_edges[0][j] = v2;
			indic_sparse[index_edges[0][j]]++;
			indic_sparse[index_edges[1][j]]++;
		}
	}
	TriEdges(index_edges, M, nb_same_edges);

	for (i = 0; i < numb_boundary; i++) {
		index_seeds[i] = indic_vertex[index_seeds[i]];
		seeded_vertex[index_seeds[i]] = true;
	}

	cs *A2, *A, *B2, *B;
	//The system to solve is A x = -B X2

	// building matrix A : laplacian for unseeded nodes
	A2 = cs_spalloc(N - numb_boundary, N - numb_boundary, M * 2 + N, 1, 1);
	if (fill_A(A2, N, M, numb_boundary, index_edges, seeded_vertex,
			   indic_sparse, nb_same_edges) == true) {
		// A = compressed-column form of A2
		A = cs_compress(A2);
		cs_spfree(A2);

		// building boundary matrix B
		B2 = cs_spalloc(N - numb_boundary, numb_boundary, 2 * M + N, 1, 1);
		fill_B(B2, N, M, numb_boundary, index_edges, seeded_vertex,
			   indic_sparse,
			   nb_same_edges);
		B = cs_compress(B2);
		cs_spfree(B2);

		// building the right hand side of the system
		cs     * X = cs_spalloc(numb_boundary, 1, numb_boundary, 1, 1);
		cs     * X2;
		int    rnz, cpt;
		cs     * b_tmp;
		double *b = (double *)malloc((N - numb_boundary) * sizeof(double));
		for (l = 0; l < nb_labels - 1; l++) {
			// building vector X
			rnz = 0;
			for (i = 0; i < numb_boundary; i++) {
				X->x[rnz] = boundary_values[l][i];
				X->p[rnz] = 0;
				X->i[rnz] = i;
				rnz++;
			}
			X->nz = rnz;
			X->m  = numb_boundary;
			X->n  = 1;

			X2    = cs_compress(X);
			b_tmp = cs_multiply(B, X2);

			for (i = 0; i < N - numb_boundary; i++)
				b[i] = 0;

			for (i = 0; i < b_tmp->nzmax; i++)
				b[b_tmp->i[i]] = -b_tmp->x[i];

			//solve Ax=b by LU decomposition, order = 1
			cs_lusol(1, A, b, 1e-7);

			cpt = 0;
			for (k = 0; k < N; k++)
				if (seeded_vertex[k] == false) {
					proba[l][index[k]] = (double)b[cpt];
					cpt++;
				}

			//Enforce boundaries exactly
			for (k = 0; k < numb_boundary; k++)
				proba[l][index[index_seeds[k]]] =
					(double)boundary_values[l][k];
			cs_spfree(X2);
			cs_spfree(b_tmp);
		}

		free(seeded_vertex);
		free(indic_sparse);
		free(nb_same_edges);
		cs_spfree(X);
		cs_spfree(A);
		cs_spfree(B);
		free(b);
		return true;
	}

	free(seeded_vertex);
	free(indic_sparse);
	free(nb_same_edges);
	return false;
}

}
