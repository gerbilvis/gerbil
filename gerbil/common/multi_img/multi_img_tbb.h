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

#endif // MULTI_IMG_TBB_H
