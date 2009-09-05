/////////////////////////////////////////////////////////////////////////////
// Name:        fams.h
// Purpose:     fast adaptive meanshift class and struct
// Author:      Ilan Shimshoni
// Modified by: Bogdan Georgescu
// Created:     08/14/2003
// Version:     v0.1
/////////////////////////////////////////////////////////////////////////////

#ifndef _FAMS_H
#define _FAMS_H

#ifndef UNIX
#define drand48()    (rand() * 1.0 / RAND_MAX)
#endif

#include <cmath>
#include "auxiliary.h"
#include "multi_img.h"
// Algorithm constants

/* Find K L */
// number of points on which test is run
#define FAMS_FKL_NEL      500
// number of times on which same test is run
#define FAMS_FKL_TIMES    10

/* FAMS main algorithm */
// maximum valid K
#define FAMS_MAX_K         70
// maximum valid L
#define FAMS_MAX_L         500
// first hash table block size
#define FAMS_BLOCKSIZE     4096
// second hash table block size
#define FAMS_BLOCKSIZE2    256
// do speedup or not
#define FAMS_DO_SPEEDUP    1
// maximum MS iterations
#define FAMS_MAXITER       100
// weight power
#define FAMS_ALPHA         1.0

/* Prune Modes */
// window size (in 2^16 units) in which modes are joined
#define FAMS_PRUNE_WINDOW    3000
// min number of points assoc to a reported mode
/*The original version had FAMS_PRUNE_MINN value 40. After testing
   it was observed that the value of 50 produces better results */
#define FAMS_PRUNE_MINN      50
// max number of modes
#define FAMS_PRUNE_MAXM      100
// max points when considering modes
#define FAMS_PRUNE_MAXP      10000

// divison of mode h
#define FAMS_PRUNE_HDIV      1
#define FAMS_FLOAT_SHIFT     100000.0

typedef struct fams_point {
	unsigned short *data_;
	unsigned short usedFlag_;
	unsigned int   window_;
	float          weightdp2_;
	fams_point& operator=(struct fams_point& d2) {
		usedFlag_  = d2.usedFlag_;
		weightdp2_ = d2.weightdp2_;
		window_    = d2.window_;
		data_      = d2.data_;
		return *this;
	};
} fams_point;
typedef fams_point*   fams_pointp;

typedef struct fams_cut {
	unsigned short which_, where_;
} fams_cut;


class fams_res_cont {
public:
int         nel_;
fams_pointp *vec_;
fams_res_cont(int n = 1) {
	nel_ = 0;
	vec_ = new fams_pointp[n];
};
~fams_res_cont() {
	delete[] vec_;
};
inline void push_back(fams_pointp in_el) {
	vec_[nel_++] = in_el;
};
inline void clear() {
	nel_ = 0;
};
inline int size() {
	return nel_;
};
};

typedef fams_cut   fams_partition[FAMS_MAX_K];

typedef struct fams_hash_entry {
	short       whichCut_;
	int         which2_;
	fams_pointp pt_;
} fams_hash_entry;

typedef struct fams_hash_entry2 {
	int            whichCut_;
	unsigned short **dp_;
} fams_hash_entry2;


const int Bs  = FAMS_BLOCKSIZE / sizeof(fams_hash_entry);
const int Bs2 = FAMS_BLOCKSIZE2 / sizeof(fams_hash_entry2);

typedef fams_hash_entry    block[Bs];
typedef fams_hash_entry2   block2[Bs2];

class FAMS
{
public:

FAMS(bool use_LSH);
~FAMS();

bool LoadPoints(char* filename);
bool ImportPoints(const multi_img& img);
void CleanPoints();
void CleanSelected();
void CleanPrunedModes();
void CleanHash();
void SelectMsPoints(double percent, int jump);

int RunFAMS(int K, int L, int k, double percent, int jump, float width,
			const std::string& outdir, const std::string& filename);
inline int RunFAMS(int K, int L, int k, float width,
				   const std::string& od, const std::string& fn) {
					return RunFAMS(K, L, k, 0., 1, width, od, fn); }
inline int RunFAMS(int K, int L, int k, int jump, float width,
				   const std::string& od, const std::string& fn) {
				 	return RunFAMS(K, L, k, 0., jump, width, od, fn); }
inline int RunFAMS(int K, int L, int k, double percent, float width,
				   const std::string& od, const std::string& fn) {
					return RunFAMS(K, L, k, percent, 1, width, od, fn); }
			
void MakeCuts(fams_partition* cuts);
void MakeCutL(fams_partition& cut);
void InitHash(int nk);
int HashFunction(int *cutVals, int whichPartition, int kk, int M = 0,
				 int *hjump = NULL);
void AddDataToHash(block HT[], int hs[], fams_point & pt, int where, int Bs,
				   int M, int which, int which2,
				   int hjump);
void ComputePilot(block *HT, int *hs, fams_partition *cuts,
					const std::string& outdir, const std::string& filename);
void GetNearestNeighbours(fams_point & who, block * HT, int *hs,
						  fams_partition * cuts, fams_res_cont & res,
						  int print,
						  int num_l[]);
void AddDataToRes(block HT[], int hs[], fams_res_cont & res, int where,
				  int Bs, int M, int which, unsigned short nnres, int which2,
				  int hjump);
unsigned short* FindInHash(block2* HT2, int *hs2, int where, int which, int M2,
						   int hjump);
void InsertIntoHash(block2* HT2, int *hs2, int where, int which,
					unsigned short **solution, int M, int hjump);
unsigned short* GetNearestNeighbours2H(unsigned short *who, block*HT, int *hs,
									   fams_partition* cuts,
									   fams_res_cont& res,
									   unsigned short **solution, block2 *HT2,
									   int *hs2);
void DoFAMS(block *HT, int *hs, fams_partition *cuts,
			block2* HT2, int *hs2);
unsigned int DoMeanShiftAdaptiveIteration(fams_res_cont& res,
										  unsigned short *old,
										  unsigned short *ret);
void SaveModes(const std::string& outdir, const std::string& filename);

void SaveMymodes(const std::string& outdir, const std::string& filename);
void SaveSegments(const std::string& outdir, const std::string& filename);
void CreatePpm(char *fn);
int LoadBandwidths(const char* fn);
void SaveBandwidths(const char* fn);
std::pair<int, int> FindKL(int Kmin, int Kmax, int Kjump, int Lmax, int k_neigh,
		   float width, float epsilon);
void ComputeRealBandwidths(unsigned int h);
double DoFindKLIteration(int K, int L, float* scores);
void ComputeScores(block *HT, int *hs, fams_partition *cuts, float* scores);
int PruneModes(int hprune, int npmin);
void SavePrunedModes(char* fn);



//Produce the boolean vector of a data point with a partition
inline void EvalCutRes(fams_point& in_pt, fams_partition& in_part,
					   int in_cut_res[]) {
	for (int in_i = 0; in_i < K_; in_i++)
		in_cut_res[in_i] = in_pt.data_[in_part[in_i].which_] >=
						   in_part[in_i].where_;
}

//Produce the boolean vector of a ms data point with a partition
inline void EvalCutRes(unsigned short *in_dat, fams_partition& in_part,
					   int in_cut_res[]) {
	for (int in_i = 0; in_i < K_; in_i++)
		in_cut_res[in_i] = in_dat[in_part[in_i].which_] >=
						   in_part[in_i].where_;
}


// return a prime number greater than minp
static int GetPrime(int minp) {
	int i, j;
	for (i = minp % 2 == 0 ? minp + 1 : minp;; i += 2) {
		int sqt = (int)sqrt(i);
		if (i % 2 == 0)
			continue;
		for (j = 3; j < sqt; j += 2) {
			if (i % j == 0)
				break;
		}
		if (j >= sqt)
			return i;
	}
	return -1;
};


// distance in L1 between two data elements
inline unsigned int DistL1(fams_point& in_pt1, fams_point& in_pt2) {
	int          in_i;
	unsigned int in_res = 0;
	for (in_i = 0; in_i < d_; in_i++)
		in_res += abs(in_pt1.data_[in_i] - in_pt2.data_[in_i]);
	return in_res;
}

/*
   a boolean function which computes the distance if it is less than dist into
   dist_res.
   It returns a boolean value
 */

inline bool DistL1Data(unsigned short* in_d1, fams_point& in_pt2,
					   double in_dist,
					   double& in_res) {
	in_res = 0;
	for (int in_i = 0; in_i < d_ && (in_res < in_dist); in_i++)
		in_res += abs(in_d1[in_i] - in_pt2.data_[in_i]);
	return(in_res < in_dist);
}

//Compare a pair of L binary vectors
inline int CompareCutRes(int in_cr1[FAMS_MAX_L][FAMS_MAX_K],
						 int in_cr2[FAMS_MAX_L][FAMS_MAX_K]) {
	for (int in_i = 0; in_i < L_; in_i++)
		for (int in_j = 0; in_j < K_; in_j++)
			if (in_cr1[in_i][in_j] != in_cr2[in_i][in_j])
				return 1;
//      if (memcmp(in_cr1[in_i], in_cr2[in_i], (K_*sizeof(int))) != 0)
//         return 1;
	return 0;
}

inline bool NotEq(unsigned short* in_d1, unsigned short* in_d2) {
	for (int in_i = 0; in_i < d_; in_i++)
		if (in_d1[in_i] != in_d2[in_i])
			return true;
	return false;
}

// interval of input data
float minVal_, maxVal_;

// input points
fams_point    *points_;
unsigned short*data_;
int           hasPoints_;
int           n_, d_, w_, h_; // number of points, number of dimensions
int           dataSize_;
double        *rr_;           //temp work

// selected points on which MS is run
int           *psel_;
int           nsel_;
unsigned short*modes_;
unsigned int  *hmodes_;
int           npm_;
unsigned short*prunedmodes_;
int           *nprunedmodes_;
float         *mymodes;
int           *indmymodes;
float         *testmymodes;
float         *tmpmymodes;
// alg params
int K_, L_, k_;
bool use_LSH_;

// hash table data
int M_, M2_;
int * hashCoeffs_;

// temporary
int            t_cut_res_[FAMS_MAX_L][FAMS_MAX_K];
int            t_old_cut_res_[FAMS_MAX_L][FAMS_MAX_K];
int            t_old_m_[FAMS_MAX_L];
int            t_m_[FAMS_MAX_L];
int            t_m2_[FAMS_MAX_L];
int            t_hjump_[FAMS_MAX_L];
unsigned short nnres1_;
unsigned short nnres2_;
};

#endif




