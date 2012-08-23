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

#include "progress_observer.h"

#include <cmath>
// for segment image
#include <cv.h>
#include "meanshift_config.h"
#include "auxiliary.h"
#include "multi_img.h"
#include "lsh.h"
// Algorithm constants

/* Find K L */
// number of points on which test is run
#define FAMS_FKL_NEL      500
// number of times on which same test is run
#define FAMS_FKL_TIMES    10

/* FAMS main algorithm */
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
// now set at runtime to allow using meanshift for post-processing with very few points
//#define FAMS_PRUNE_MINN      50
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
	// size of ms window around this point (L1)
	unsigned int   window_;
	double         weightdp2_;
	fams_point& operator=(struct fams_point& d2) {
		usedFlag_  = d2.usedFlag_;
		weightdp2_ = d2.weightdp2_;
		window_    = d2.window_;
		data_      = d2.data_;
		return *this;
	};
} fams_point;
typedef fams_point*   fams_pointp;

class FAMS
{
public:

FAMS(bool use_LSH);
~FAMS();

vole::ProgressObserver *progressObserver;

bool LoadPoints(char* filename);
bool ImportPoints(const multi_img& img);
void CleanPoints();
void CleanSelected();
void CleanPrunedModes();
void SelectMsPoints(double percent, int jump);

int RunFAMS(const vole::MeanShiftConfig &config, double percent, int jump, float width,
			vector<double> *bandwidths = NULL);
inline int RunFAMS(const vole::MeanShiftConfig &config, float width, vector<double> *bandwidths) {
					return RunFAMS(config, 0., 1, width, bandwidths); }
inline int RunFAMS(const vole::MeanShiftConfig &config, int jump, float width) {
					return RunFAMS(config, 0., jump, width); }
inline int RunFAMS(const vole::MeanShiftConfig &config, double percent, float width) {
					return RunFAMS(config, percent, 1, width); }
			
bool ComputePilot(LSH &lsh);
bool DoFAMS(LSH &lsh);
unsigned int DoMeanShiftAdaptiveIteration(const std::vector<unsigned int>& res,
										  unsigned short *old,
										  unsigned short *ret);
void SaveModes(const std::string& filebase);

void SaveMymodes(const std::string& filebase);
void CreatePpm(char *fn);
int LoadBandwidths(const char* fn);
void SaveBandwidths(const char* fn);
std::pair<int, int> FindKL(int Kmin, int Kmax, int Kjump, int Lmax, int k_neigh,
		   float width, float epsilon);
void ComputeRealBandwidths(unsigned int h);
double DoFindKLIteration(int K, int L, float* scores);
void ComputeScores(float* scores, LSH &lsh);
int PruneModes(int hprune, int npmin);
void SavePrunedModes(const std::string& filebase);

// returns 2D, 1 color image that uses color markers to show image segments
cv::Mat1s segmentImage(bool normalize = false);

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

private:
bool progressUpdate(int percent);
};

#endif




