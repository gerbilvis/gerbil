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
#include "meanshift_config.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <opencv2/core/core.hpp> // for segment image & timer functionality
#include <multi_img.h>
#include <lsh.h>
#include <lshreader.h>
#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/mutex.h>
#include <emmintrin.h>

// Algorithm constants

/* Find K L */
// number of points on which test is run
#define FAMS_FKL_NEL      500
// number of times on which same test is run
#define FAMS_FKL_TIMES    10

/* FAMS main algorithm */
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
// now runtime setting to allow meanshift post-processing with very few points
//#define FAMS_PRUNE_MINN      50
// max number of modes
#define FAMS_PRUNE_MAXM      100
// max points when considering modes
#define FAMS_PRUNE_MAXP      10000

// divison of mode h
#define FAMS_PRUNE_HDIV      1
#define FAMS_FLOAT_SHIFT     100000.0

struct fams_point {
	/*fams_point& operator=(const fams_point& d2) {
		weightdp2_ = d2.weightdp2_;
		window_    = d2.window_;
		data_      = d2.data_;
		return *this;
	}*/ // useless?!

	unsigned short *data_;
	// size of ms window around this point (L1)
	unsigned int   window_;
	double         weightdp2_;
};

class FAMS
{
public:

	struct ComputePilotPoint {
		ComputePilotPoint(FAMS& master, vector<double> *weights = NULL)
			: fams(master), weights(weights), dbg_acc(0.), dbg_noknn(0) {}
		ComputePilotPoint(ComputePilotPoint& other, tbb::split)
			: fams(other.fams), weights(other.weights),
			  dbg_acc(0.), dbg_noknn(0) {}
		void operator()(const tbb::blocked_range<int> &r);
		void join(ComputePilotPoint &other)
		{
			dbg_acc += other.dbg_acc;
			dbg_noknn += other.dbg_noknn;
		}

		FAMS& fams;
		vector<double> *weights;
		double dbg_acc; // double, as it can go over limit of 32 bit integer
		unsigned int dbg_noknn;
	};

	struct MeanShiftPoint {
		MeanShiftPoint(FAMS& master)
			: fams(master) {}
		void operator()(const tbb::blocked_range<int> &r) const;

		FAMS& fams;
	};

	friend struct ComputePilotPoint;
	friend struct MeanShiftPoint;

	FAMS(const vole::MeanShiftConfig& config, vole::ProgressObserver* po=NULL);
	~FAMS();

	int getDimensionality() const { return d_; }
	const fams_point* getPoints() const { return points_; }
	const int* getModePerPixel() const { return indmymodes; }

	bool LoadPoints(char* filename);
	bool ImportPoints(const multi_img& img);
	void CleanPoints();
	void CleanPrunedModes();
	void SelectMsPoints(double percent, int jump);
	void ImportMsPoints(std::vector<fams_point> &points);

	/** optional argument bandwidths provides pre-calculated
	 *  per-point bandwidth
	 */
	bool PrepareFAMS(vector<double> *bandwidths = NULL);
	bool FinishFAMS();
	void PruneModes();
	void SaveModeImg(const std::string& filebase,
					 const std::vector<multi_img::BandDesc>& ref);
	void SavePrunedModeImg(const std::string& filebase,
					 const std::vector<multi_img::BandDesc>& ref);
	void DbgSavePoints(const std::string& filebase,
							 const std::vector<fams_point> points,
							 const std::vector<multi_img::BandDesc>& ref);
	void SaveModes(const std::string& filebase);

	void SaveMymodes(const std::string& filebase);
	void CreatePpm(char *fn);
	std::pair<int, int> FindKL();
	void ComputeRealBandwidths(unsigned int h);
	int64 DoFindKLIteration(int K, int L, float* scores);
	void ComputeScores(float* scores, LSHReader &lsh, int L);
	void SavePrunedModes(const std::string& filebase);

	// returns 2D intensity image containing segment indices
	cv::Mat1s segmentImage();

	// distance in L1 between two data elements
	inline unsigned int DistL1(fams_point& in_pt1, fams_point& in_pt2) const
	{
		int i = 0;
		unsigned int ret = 0;
		__m128i vret = _mm_setzero_si128(), vzero = _mm_setzero_si128();
		for (; i < d_ - 8; i += 8) {
			__m128i vec1 = _mm_loadu_si128((__m128i*)&in_pt1.data_[i]);
			__m128i vec2 = _mm_loadu_si128((__m128i*)&in_pt2.data_[i]);
			__m128i v1i1 = _mm_unpacklo_epi16(vec1, vzero);
			__m128i v1i2 = _mm_unpackhi_epi16(vec1, vzero);
			__m128i v2i1 = _mm_unpacklo_epi16(vec2, vzero);
			__m128i v2i2 = _mm_unpackhi_epi16(vec2, vzero);
			__m128i diff1 = _mm_sub_epi32(v1i1, v2i1);
			__m128i diff2 = _mm_sub_epi32(v1i2, v2i2);
			__m128i mask1 = _mm_srai_epi32(diff1, 31); // shift 32-1 bits
			__m128i mask2 = _mm_srai_epi32(diff2, 31);
			__m128i abs1 = _mm_xor_si128(_mm_add_epi32(diff1, mask1), mask1);
			__m128i abs2 = _mm_xor_si128(_mm_add_epi32(diff2, mask2), mask2);
			vret = _mm_add_epi32(abs1, _mm_add_epi32(abs2, vret));
		}
		ret += *((unsigned int*)&vret + 0);
		ret += *((unsigned int*)&vret + 1);
		ret += *((unsigned int*)&vret + 2);
		ret += *((unsigned int*)&vret + 3);
		for (; i < d_; i++) {
			ret += abs(in_pt1.data_[i] - in_pt2.data_[i]);
		}

		return ret;
	}

	/*
	   a boolean function which computes the distance if it is less than dist
	   into dist_res.
	   It returns a boolean value
	 */
	inline bool DistL1Data(unsigned short* in_d1, fams_point& in_pt2,
						   double in_dist,
						   double& in_res) const
	{
		in_res = 0;
		for (int in_i = 0; in_i < d_ && (in_res < in_dist); in_i++)
			in_res += abs(in_d1[in_i] - in_pt2.data_[in_i]);
		return (in_res < in_dist);
	}

	inline bool NotEq(unsigned short* in_d1, unsigned short* in_d2) const
	{
		for (int in_i = 0; in_i < d_; in_i++)
			if (in_d1[in_i] != in_d2[in_i])
				return true;
		return false;
	}

	inline static void bgLog(const char *varStr, ...)
	{
		//obtain argument list using ANSI standard...
		va_list argList;
		va_start(argList, varStr);

		//print the output string to stderr using
		vfprintf(stderr, varStr, argList);
		va_end(argList);
		fflush(stderr);
	}

	int n_, d_, w_, h_; // number of points, number of dimensions

protected:
	bool ComputePilot(vector<double> *weights = NULL);
	unsigned int DoMSAdaptiveIteration(
			const std::vector<unsigned int> *res,
			unsigned short *old, unsigned short *ret) const;

	// tells whether to continue, takes recent progress
	bool progressUpdate(float percent, bool absolute = true);

	// interval of input data
	float minVal_, maxVal_;

	// input points
	fams_point    *points_;
	unsigned short*data_;
	int           hasPoints_;
	int           dataSize_;

	// selected points on which MS is run
	std::vector<fams_point*> psel_;
	int           nsel_;
	std::vector<unsigned short> modes_;
	std::vector<unsigned int>  hmodes_;

	// HACK for superpixel size
public:
	mutable std::vector<int> spsizes;
protected:

	int           npm_;
	unsigned short*prunedmodes_;
	int           *nprunedmodes_;
	float         *mymodes;
	int           *indmymodes;
	float         *testmymodes;
	float         *tmpmymodes;

	// LSH used during ordinary run
	LSH *lsh_;
	// alg params
	const vole::MeanShiftConfig &config;

	// observer for progress tracking
	vole::ProgressObserver *progressObserver;
	float progress, progress_old;
	tbb::mutex progressMutex;

	tbb::task_scheduler_init tbbinit;
};


#endif




