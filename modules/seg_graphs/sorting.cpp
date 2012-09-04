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

#include <stdlib.h>
#include <string.h>

#if !defined _MSC_VER && !defined __BORLANDC__
  #include <stdint.h>
#else
	// bullshit defines needed for M$ visual studio versions < 2010 (no <stdint.h>)
	#ifndef uint32_t
	#define uint32_t unsigned int
	#endif
	#ifndef int32_t
	#define int32_t int
	#endif
	#ifndef int16_t
	#define int16_t short
	#endif
#endif

#include "sorting.h"

#include <algorithm>
#include <vector>

namespace powerwaterseg {

/* =============================================================== */
int32_t Partitionner(int *A, int * I, int32_t p, int32_t r) {
/* =============================================================== */
/*
   partitionne les elements de A entre l'indice p (compris) et l'indice r (compris)
   en deux groupes : ceux <= A[p] et les autres.
 */
	int     t;
	int     t1;
	int     x = A[p];
	int32_t i = p - 1;
	int32_t j = r + 1;
	while (1) {
		do
			j--;while (A[j] > x);
		do
			i++;while (A[i] < x);
		if (i < j) {
			t    = A[i];
			A[i] = A[j];
			A[j] = t;
			t1   = I[i];
			I[i] = I[j];
			I[j] = t1;
		} else
			return j;
	} /* while (1) */
}     /* Partitionner() */

/* =============================================================== */
int32_t PartitionStochastique(int *A, int * I, int32_t p, int32_t r) {
/* =============================================================== */
/*
   partitionne les elements de A entre l'indice p (compris) et l'indice r (compris)
   en deux groupes : ceux <= A[q] et les autres, avec q tire au hasard dans [p,r].
 */
	int     t;
	int     t1;
	int32_t q;

	q    = p + (rand() % (r - p + 1));
	t    = A[p];    /* echange A[p] et A[q] */
	A[p] = A[q];
	A[q] = t;

	t1   = I[p];     /* echange I[p] et I[q] */
	I[p] = I[q];
	I[q] = t1;

	return Partitionner(A, I, p, r);
} /* PartitionStochastique() */



/* =============================================================== */
void TriRapideStochastique(int * A, int *I, int32_t p, int32_t r) {
/* =============================================================== */
/*
   trie les valeurs du tableau A de l'indice p (compris) a l'indice r (compris)
   par ordre croissant
 */
	int32_t q;
	if (p < r) {
		q = PartitionStochastique(A, I, p, r);
		TriRapideStochastique(A, I, p, q);
		TriRapideStochastique(A, I, q + 1, r);
	}
} /* TriRapideStochastique() */




/* =============================================================== */
int32_t Partitionner_dec(uint32_t *A, int * I, int32_t p, int32_t r) {
/* =============================================================== */
/*
   partitionne les elements de A entre l'indice p (compris) et l'indice r (compris)
   en deux groupes : ceux <= A[p] et les autres.
 */
	uint32_t t;
	int      t1;
	uint32_t x = A[p];
	int32_t  i = p - 1;
	int32_t  j = r + 1;
	while (1) {
		do
			j--;while (A[j] < x);
		do
			i++;while (A[i] > x);
		if (i < j) {
			t    = A[i];
			A[i] = A[j];
			A[j] = t;
			t1   = I[i];
			I[i] = I[j];
			I[j] = t1;
		} else
			return j;
	} /* while (1) */
}     /* Partitionner() */

/* =============================================================== */
int32_t PartitionStochastique_dec(uint32_t *A, int * I, int32_t p, int32_t r) {
/* =============================================================== */
/*
   partitionne les elements de A entre l'indice p (compris) et l'indice r (compris)
   en deux groupes : ceux <= A[q] et les autres, avec q tire au hasard dans [p,r].
 */
	int16_t t;
	int     t1;
	int32_t q;

	q    = p + (rand() % (r - p + 1));
	t    = A[p];    /* echange A[p] et A[q] */
	A[p] = A[q];
	A[q] = t;

	t1   = I[p];     /* echange I[p] et I[q] */
	I[p] = I[q];
	I[q] = t1;

	return Partitionner_dec(A, I, p, r);
} /* PartitionStochastique() */


/* =============================================================== */
void TriRapideStochastique_dec(uint32_t * A, int *I, int32_t p, int32_t r) {
/* =============================================================== */
/*
   trie les valeurs du tableau A de l'indice p (compris) a l'indice r (compris)
   par ordre decroissant
 */
	int32_t q;
	if (p < r) {
		q = PartitionStochastique_dec(A, I, p, r);
		TriRapideStochastique_dec(A, I, p, q);
		TriRapideStochastique_dec(A, I, q + 1, r);
	}
} /* TriRapideStochastique() */





/* =============================================================== */
int32_t Partitionner_inc(uint32_t *A, int * I, int32_t p, int32_t r) {
/* =============================================================== */
/*
   partitionne les elements de A entre l'indice p (compris) et l'indice r (compris)
   en deux groupes : ceux <= A[p] et les autres.
 */
	uint32_t t;
	int      t1;
	uint32_t x = A[p];
	int32_t  i = p - 1;
	int32_t  j = r + 1;
	while (1) {
		do
			j--;while (A[j] > x);
		do
			i++;while (A[i] < x);
		if (i < j) {
			t    = A[i];
			A[i] = A[j];
			A[j] = t;
			t1   = I[i];
			I[i] = I[j];
			I[j] = t1;
		} else
			return j;
	} /* while (1) */
}

/* =============================================================== */
int32_t PartitionStochastique_inc(uint32_t *A, int * I, int32_t p, int32_t r) {
/* =============================================================== */
/*
   partitionne les elements de A entre l'indice p (compris) et l'indice r (compris)
   en deux groupes : ceux <= A[q] et les autres, avec q tire au hasard dans [p,r].
 */
	uint32_t t;
	int      t1;
	int32_t  q;

	q    = p + (rand() % (r - p + 1));
	t    = A[p];    /* echange A[p] et A[q] */
	A[p] = A[q];
	A[q] = t;

	t1   = I[p];     /* echange I[p] et I[q] */
	I[p] = I[q];
	I[q] = t1;

	return Partitionner_inc(A, I, p, r);
}



/* =============================================================== */
void TriRapideStochastique_inc(uint32_t * A, int *I, int32_t p, int32_t r) {
/* =============================================================== */
/*
   trie les valeurs du tableau A de l'indice p (compris) a l'indice r (compris)
   par ordre croissant
 */
	int32_t q;
	if (p < r) {
		q = PartitionStochastique_inc(A, I, p, r);
		TriRapideStochastique_inc(A, I, p, q);
		TriRapideStochastique_inc(A, I, q + 1, r);
	}
} /* TriRapideStochastique() */

bool Score::operator<(const Score& score) const
{
	return (value < score.value);
}


/* =============================================================== */
void sortRange(float * F, int * Es, int M, bool reverse)
/* =============================================================== */
/* sort values of array F ascending (default), Es stores the edge nr. */
{
	std::vector<Score> data(M);

	for(int k = 0; k < M; k++) {
		Score entry = {F[k], Es[k]};
		data[k] = entry;
	}

	sort(data.begin(), data.end());

	if (reverse) {
		for (int k=0; k<M; k++) {
			F[k] = data[M - k - 1].value;
			Es[k] = data[M - k - 1].index;
		}
	} else {
		for (int k=0; k<M; k++) {
			F[k] = data[k].value;
			Es[k] = data[k].index;
		}
	}
}

}
