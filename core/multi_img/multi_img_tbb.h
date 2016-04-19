#ifndef MULTI_IMG_TBB_H
#define MULTI_IMG_TBB_H

#include <multi_img.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>

// TODO doc
class RebuildPixels {
public:
	RebuildPixels(multi_img &multi) : multi(multi) {}
	void operator()(const tbb::blocked_range<size_t> &r) const;
	// second way to do it that can also be run on a specific region
	void operator()(const tbb::blocked_range2d<int> &r) const;
private:
	multi_img &multi;
};

// TODO doc
class ApplyCache {
public:
	ApplyCache(multi_img &multi) : multi(multi) {}
	void operator()(const tbb::blocked_range<size_t> &r) const;
	// second way to do it that can also be run on a specific region
	void operator()(const tbb::blocked_range2d<int> &r) const;
private:
	multi_img &multi;
};

// TODO doc
class DetermineRange {
public:
	DetermineRange(multi_img &multi)
		: multi(multi), min(multi_img::ValueMax), max(multi_img::ValueMin) {}
	DetermineRange(DetermineRange &toSplit, tbb::split)
		: multi(toSplit.multi), min(multi_img::ValueMax), max(multi_img::ValueMin) {}
	void operator()(const tbb::blocked_range<size_t> &r);
	void join(DetermineRange &toJoin);
	multi_img::Value GetMin() { return min; }
	multi_img::Value GetMax() { return max; }
private:
	multi_img &multi;
	multi_img::Value min;
	multi_img::Value max;
};

// TODO doc
class Xyz {
public:
	Xyz(multi_img_base &multi, cv::Mat_<cv::Vec3f> &xyz, multi_img::Band &band, int cie)
		: multi(multi), xyz(xyz), band(band), cie(cie) {}
	void operator()(const tbb::blocked_range2d<int> &r) const;
private:
	multi_img_base &multi;
	cv::Mat_<cv::Vec3f> &xyz;
	multi_img::Band &band;
	int cie;
};

// TODO doc
class Bgr {
public:
	Bgr(multi_img_base &multi, cv::Mat_<cv::Vec3f> &xyz, cv::Mat_<cv::Vec3f> &bgr, float greensum)
		: multi(multi), xyz(xyz), bgr(bgr), greensum(greensum) {}
	void operator()(const tbb::blocked_range2d<int> &r) const;
private:
	multi_img_base &multi;
	cv::Mat_<cv::Vec3f> &xyz;
	cv::Mat_<cv::Vec3f> &bgr;
	float greensum;
};

// TODO doc
class Grad {
public:
	Grad(multi_img &source, multi_img &target)
		: source(source), target(target) {}
	void operator()(const tbb::blocked_range<size_t> &r) const;

private:
	multi_img &source;
	multi_img &target;
};

// TODO doc
class Log {
public:
	Log(multi_img &source, multi_img &target)
		: source(source), target(target) {}
	void operator()(const tbb::blocked_range<size_t> &r) const;

private:
	multi_img &source;
	multi_img &target;
};

// TODO doc
class NormL2 {
public:
	NormL2(multi_img &source, multi_img &target)
		: source(source), target(target) {}
	void operator()(const tbb::blocked_range2d<int> &r) const;

private:
	multi_img &source;
	multi_img &target;
};

// TODO doc
class Clamp {
public:
	Clamp(multi_img &source, multi_img &target) : source(source), target(target) {}
	void operator()(const tbb::blocked_range<size_t> &r) const;
private:
	multi_img &source;
	multi_img &target;
};

// TODO doc
class Illumination {
public:
	Illumination(multi_img &source, multi_img &target, Illuminant& il, bool remove)
		: source(source), target(target), il(il), remove(remove) {}
	void operator()(const tbb::blocked_range<size_t> &r) const;
private:
	multi_img &source;
	multi_img &target;
	Illuminant &il;
	bool remove;
};

class PcaProjection {
public:
	PcaProjection(cv::Mat_<multi_img::Value> &source, multi_img &target, cv::PCA &pca)
		: source(source), target(target), pca(pca) {}
	void operator()(const tbb::blocked_range<size_t> &r) const;
private:
	cv::Mat_<multi_img::Value> &source;
	multi_img &target;
	cv::PCA &pca;
};


class MultiImg2BandMat {
public:
	MultiImg2BandMat(multi_img &source, cv::Mat_<multi_img::Value> &target)
		: source(source), target(target) {}
	void operator()(const tbb::blocked_range<size_t> &r) const;
private:
	multi_img &source;
	cv::Mat_<multi_img::Value> &target;
};

class Resize {
public:
	Resize(multi_img &source, multi_img &target, size_t newsize)
		: source(source), target(target), newsize(newsize) {}
	void operator()(const tbb::blocked_range2d<int> &r) const;
private:
	multi_img &source;
	multi_img &target;
	size_t newsize;
};

#endif // MULTI_IMG_TBB_H
