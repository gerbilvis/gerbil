#ifndef MEDIANSHIFT_H
#define MEDIANSHIFT_H

#include "medianshift_config.h"
#include "progress_observer.h"
#include <multi_img.h>
#include <lsh.h>

/// maximum number of shift iterations
#define MEDIANSHIFT_MAXITER 5

/// granularity of distance bucket-sort for adaptive window sizes
#define MEDIANSHIFT_DISTBUCKETS 700

/// maximum adaptive window size (percentage of max L1 distance)
#define MEDIANSHIFT_MAXWIN .1f

// number of random projects used for Tukey median
#define MEDIANSHIFT_TUKEY_NPROJ 100

using namespace cv;

namespace vole {

class MedianShift {
public:
	typedef float Value;

	MedianShift(const MedianShiftConfig& config) : config(config) {}

	cv::Mat1s execute(const multi_img& input, ProgressObserver *progress = NULL);

private:
	const MedianShiftConfig &config;

	ProgressObserver *progressObserver;

	/// pointer to interleaved data (as returned by multi_img::export_interleaved)
	unsigned short *data;

	/// return Tukey median within given set of points
	unsigned int getTukeyMedian(const multi_img &image, const vector<unsigned int> &points, int nprojections, const vector<unsigned int> &weights = vector<unsigned int>());
	unsigned int getTukeyMedian(const Mat_<Value> &points, int nprojections, const vector<unsigned int> &weights = vector<unsigned int>());

	/// TODO: merge both prune functions
	/// return all points within certain radius around certian center (L2 norm)
	vector<unsigned int> pruneByL2(const multi_img &image, const vector<unsigned int> &points, unsigned int center, double radius);

	/// return all points within certain radius around certian center (L1 norm)
	vector<unsigned int> pruneByL1(const multi_img &image, const vector<unsigned int> &points, unsigned int center, double radius);

	/// sort vector, remove duplicate values
	void removeDups(vector<unsigned int> &v);

	/// write L1 norm into dist, return true if it's less than cmp_dist
	inline bool distL1(const vector<Value> &a, const vector<Value> &b, double cmp_dist, double &dist) const
	{
		assert(a.size() == b.size());
		dist = 0;
		for (unsigned int i = 0; i < a.size(); ++i)
			dist += std::abs(a[i] - b[i]);
		return dist < cmp_dist;
	}

	/// write L2 norm into dist, return true if it's less than cmp_dist
	inline bool distL2(const vector<Value> &a, const vector<Value> &b, double cmp_dist, double &dist) const
	{
		dist = 0;
		for (unsigned int i = 0; i < a.size(); ++i)
			dist += (a[i] - b[i])*(a[i] - b[i]);
		dist = std::sqrt(dist);
		return dist < cmp_dist;
	}

	/// geodesic propagation of modes to whole image
	vector<unsigned int> propagateModes(LSH &lsh, const vector<unsigned int> &modes, const multi_img &image, const vector<double> &windowSizes);

	/// update progress if an observer exists
	bool progressUpdate(int percent);

	/// utility class for sorting in getTukeyMedian()
	class TukeyPred {
	private:
		const Mat_<float> &v;
	public:
		TukeyPred(const Mat_<float> &v) : v(v) {}

		bool operator()(const int a, const int b) const {
			return v(0, a) < v(0, b);
		}
	};

	/// utility container for geodesic propagation
	struct GpItem {
		float dist;
		unsigned int index;
		unsigned int mode;

		GpItem(float dist_, unsigned int index_, unsigned int mode_) :
		dist(dist_), index(index_), mode(mode_) {}

		bool operator<(const GpItem& other) const {
			return dist > other.dist;
		}
	};

};

}

#endif // MEDIANSHIFT_H
