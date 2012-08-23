#ifndef SUPERPIXELS_SUPERPIXEL_H
#define SUPERPIXELS_SUPERPIXEL_H

#include <opencv2/core/core.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace superpixels {

/** Representation of a superpixel. */
class Superpixel {
public:
	Superpixel()
		: size( 0 )
	{}

	/** Coordinates of the pixels in the image belonging to the superpixel. */
	std::vector<cv::Point> coordinates;
	cv::Rect bbox;
	int size;
	
	void clear()
	{
		coordinates.clear();
		bbox.x = 0;
		bbox.y = 0;
		bbox.width = 0;
		bbox.height = 0;
		size = 0;
	}
	
	bool empty() const
	{
		return (size == 0 ? true : false);
	}
	
	// TODO damit Svens Code diese Klasse auch benutzen kann:
	//      befuellen in superpixelsWithMask; nimm code aus muil/segment.cpp
	/* cv::Mat_<unsigned char> mask_of_superpixels; // hier: 255:
	 *              // Pixel ist im Superpixel, 0: Pixel ist nicht im Superpixel;
	 * groesse der maske ist groesse des rectangles
	 * cv::Mat_<cv::Vec3b> source_image;
	 * cv::Rect source_region
	*/
	
private:
	friend class boost::serialization::access;
	// When the class Archive corresponds to an output archive, the
	// & operator is defined similar to <<.  Likewise, when the class Archive
	// is a type of input archive the & operator is defined similar to >>.
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & coordinates;
		ar & bbox.x;
		ar & bbox.y;
		ar & bbox.width;
		ar & bbox.height;
	}
};

} // namespace superpixels

#endif // SUPERPIXELS_SUPERPIXEL_H
