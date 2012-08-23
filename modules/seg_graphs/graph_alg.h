#ifndef GRAPH_ALG_H
#define GRAPH_ALG_H

namespace powerwaterseg {

// union find (maybe improve performance with re-assign operators)
/*================================================*/
inline int element_link(int x, int y, int *Rnk, int *Fth) {
/*================================================*/
	if (Rnk[x] > Rnk[y]) {
		int t = x; x = y; y = t; // swap x,y
	}
	if (Rnk[x] == Rnk[y]) {
		Rnk[y] = Rnk[y] + 1;
	}
	Fth[x] = y;
	return y;
}

/*===============================*/
inline int element_find(int x, int *Fth) {
/*===============================*/
	while (Fth[x] != x)
		x = Fth[x]; // climb up
	return x;
}

/*******************************************************
Function RandomWalker computes the solution to the Dirichlet problem (RW potential function) 
on a general graph represented by an edge list, given boundary conditions (seeds, etc.)
*******************************************************/
bool RandomWalker(int **index_edges,           /* list of edges */
			 int M,                       /* number of edges */
			 int *index,                 /* list of vertices */
			 int *indic_vertex,          /* boolean array of vertices*/
			 int N,                       /* number of vertices */  
			 int *index_seeds,            /* list of nodes that are seeded*/
			 double **boundary_values,  /* associated values for seeds (labels)*/
			 int numb_boundary,           /* number of seeded nodes */
			 int nb_labels,               /* nb of different labels*/
			 double **proba);         /* solution to the Dirichlet problem*/

}
#endif
