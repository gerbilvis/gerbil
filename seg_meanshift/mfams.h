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

#include "meanshift_config.h"

#include <multi_img.h>
#include <progress_observer.h>

#include <lsh.h>
#include <lshreader.h>

#include <opencv2/core/core.hpp> // for segment image & timer functionality
#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/mutex.h>

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <limits>
#include <emmintrin.h>

namespace seg_meanshift {

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
// float shift used for dp2, no idea what it really is supposed to do
#define FAMS_FLOAT_SHIFT     100000.0

/* Prune Modes */
// window size (in 2^16 units) in which modes are joined
#define FAMS_PRUNE_WINDOW    3000
// min number of points assoc to a reported mode
/*The original version had FAMS_PRUNE_MINN value 40. After testing
   it was observed that the value of 50 produces better results */
// now runtime setting to allow meanshift post-processing with very few points
//#define FAMS_PRUNE_MINN      50
// max number of modes
#define FAMS_PRUNE_MAXM      200
// max points when considering modes
#define FAMS_PRUNE_MAXP      10000

// divison of mode h
#define FAMS_PRUNE_HDIV      1

class FAMS
{
public:

	struct Point {
		std::vector<unsigned short> *data;
		// size of ms window around this point (L1)
		unsigned int   window;
		double         weightdp2;
	};

	struct Mode {
		std::vector<unsigned short> data;
		unsigned int window;
	};

	// used for mode pruning, defined in mode_pruning.cpp
	struct MergedMode {
		MergedMode() {}
		MergedMode(const FAMS::Mode &d, int m, int spm);

		// compare sizes for DESCENDING sort
		static inline bool cmpSize(const MergedMode& a, const MergedMode& b)
		{	return (a.spmembers > b.spmembers);	}

		std::vector<unsigned short> normalized() const;
		double distTo(const FAMS::Mode &m) const;
		void add(const FAMS::Mode &m, int sp);
		bool invalidateIfSmall(int smallest);

		std::vector<float> data;
		int members;
		int spmembers;
		bool valid;
	};

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

	FAMS(const MeanShiftConfig &config, ProgressObserver *po = 0);
	~FAMS();

	const std::vector<Point>& getPoints() const { return datapoints; }
	const std::vector<int>& getModePerPixel() const { return prunedIndex; }

	bool loadPoints(char* filename);
	bool importPoints(const multi_img& img);
	void selectStartPoints(double percent, int jump);
	void importStartPoints(std::vector<Point> &points);

	/** optional argument bandwidths provides pre-calculated
	 *  per-point bandwidth
	 */
	bool prepareFAMS(vector<double> *bandwidths = NULL);
	bool finishFAMS();
	void pruneModes();

	/* save to file as %g: modes for all starting points or the pruned ones */
	void saveModes(const std::string& filename, bool pruned);
	/* save to multi_img: modes for all starting points or the pruned ones */
	void saveModeImg(const std::string& filebase, bool pruned,
					 const std::vector<multi_img::BandDesc>& ref);
	void dbgSavePoints(const std::string& filebase,
					   const std::vector<Point> &points,
					   const std::vector<multi_img::BandDesc>& ref);

	std::pair<int, int> FindKL();
	void ComputeRealBandwidths(unsigned int h);
	int64 DoFindKLIteration(int K, int L, float* scores);
	void ComputeScores(float* scores, LSHReader &lsh, int L);

	// returns 2D intensity image containing segment indices
	cv::Mat1s segmentImage() const;
	// returns a vector of pruned modes (sorted by size)
	std::vector<multi_img::Pixel> modeVector() const;

	// conversion functions
	inline multi_img::Value ushort2value(unsigned short in) const
	{
		return (multi_img::Value)in * (maxVal_ - minVal_) / 65535.f + minVal_;
	}
	template <typename T>
	inline T value2ushort(multi_img::Value in) const
	{
		multi_img::Value scale = 65535.f / (maxVal_ - minVal_);
		return (in - minVal_) / scale;
	}

	union m128i_uint {
		__m128i v;
		unsigned int i[4];
	};

	// distance in L1 between two data elements
	inline unsigned int DistL1(Point& in_pt1, Point& in_pt2) const
	{
		size_t i = 0;
		unsigned int ret = 0;
		if (in_pt1.data->size() > 7) {
			__m128i vret = _mm_setzero_si128(), vzero = _mm_setzero_si128();
			for (; i < in_pt1.data->size() - 8; i += 8) {
				unsigned short *p1 = &(*in_pt1.data)[i];
				unsigned short *p2 = &(*in_pt2.data)[i];
				__m128i vec1 = _mm_loadu_si128((__m128i*)p1);
				__m128i vec2 = _mm_loadu_si128((__m128i*)p2);
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
			m128i_uint *unpack = (m128i_uint*)&vret;
			ret += unpack->i[0];
			ret += unpack->i[1];
			ret += unpack->i[2];
			ret += unpack->i[3];
		}
		for (; i < in_pt1.data->size(); i++) {
			ret += abs((*in_pt1.data)[i] - (*in_pt2.data)[i]);
		}

		return ret;
	}

	/*
	   a boolean function which computes the distance if it is less than dist
	   into dist_res.
	   Not using SSE due to early abortion.
	 */
	inline bool DistL1Data(const std::vector<unsigned short> &in_d1,
						   const Point& in_pt2, double in_dist,
						   double& in_res) const
	{
		in_res = 0;
		for (size_t in_i = 0;
			 in_i < in_d1.size() && (in_res < in_dist); in_i++)
			in_res += abs(in_d1[in_i] - (*in_pt2.data)[in_i]);
		return (in_res < in_dist);
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

	unsigned int n_, d_, w_, h_; // number of points, number of dimensions

protected:
	bool ComputePilot(vector<double> *weights = NULL);
	unsigned int DoMSAdaptiveIteration(
			const std::vector<unsigned int> *res,
			const std::vector<unsigned short> &old,
			std::vector<unsigned short> &ret) const;

	// tells whether to continue, takes recent progress
	bool progressUpdate(float percent, bool absolute = true);

	// helper functions to pruneModes()
	static std::pair<double, int>
	findClosest(const Mode &mode, const std::vector<MergedMode> &foomodes);
	void trimModes(std::vector<MergedMode> &foomodes, int npmin, bool sp,
				   size_t allowance = std::numeric_limits<size_t>::max());

	// interval of input data
	float minVal_, maxVal_;

	// input points
	std::vector<Point> datapoints;

	// input data, in case we need to store it ourselves
	std::vector<std::vector<unsigned short> > dataholder;

	// selected points on which MS is run
	std::vector<Point*> startPoints;

	// modes derived for these points
	std::vector<Mode> modes;

	// final result of mode pruning
	std::vector<std::vector<unsigned short> > prunedModes;

	// index of each pixel regarding to prunedModes
	std::vector<int> prunedIndex;

	// HACK for superpixel size
public:
	mutable std::vector<int> spsizes;

protected:

	// LSH used during ordinary run
	LSH *lsh_;
	// alg params
	const MeanShiftConfig &config;

	// observer for progress tracking
	ProgressObserver *po;
	float progress, progress_old;
	tbb::mutex progressMutex;

	tbb::task_scheduler_init tbbinit;
};

}
#endif
