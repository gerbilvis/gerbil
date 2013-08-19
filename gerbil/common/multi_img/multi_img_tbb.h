#ifndef MULTI_IMG_TBB_H
#define MULTI_IMG_TBB_H


class RebuildPixels {
public:
	RebuildPixels(multi_img &multi) : multi(multi) {}
	void operator()(const tbb::blocked_range<size_t> &r) const;
private:
	multi_img &multi;
};

class ApplyCache {
public:
	ApplyCache(multi_img &multi) : multi(multi) {}
	void operator()(const tbb::blocked_range<size_t> &r) const;
private:
	multi_img &multi;
};

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

#endif // MULTI_IMG_TBB_H
