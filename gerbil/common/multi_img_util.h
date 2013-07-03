#ifndef MULTI_IMG_UTIL_H
#define MULTI_IMG_UTIL_H

/** Image Data Range. */
struct ImageDataRange {
	multi_img::Value min;
	multi_img::Value max;

	ImageDataRange()
		: min(0.), max(0.)
	{}

	explicit ImageDataRange(multi_img::Value minx, multi_img::Value maxx)
		: min(minx), max(maxx)
	{
	}

	explicit ImageDataRange(const std::pair<multi_img::Value,multi_img::Value> &minmax)
		: min(minmax.first), max(minmax.second)
	{
	}

	/** Returns (min,max) as a std::pair. */
	std::pair<multi_img::Value, multi_img::Value> asPair() {
		return std::pair<multi_img::Value, multi_img::Value>(min,max);
	}
};

// for debugging
inline std::ostream& operator<<(std::ostream& stream, const ImageDataRange& r) {
	stream << "[" << r.min << "," << r.max << "]";
}

#endif // MULTI_IMG_UTIL_H
