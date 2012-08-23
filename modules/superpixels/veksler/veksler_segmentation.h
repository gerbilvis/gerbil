#ifndef SUPERPIXELS_VEKSLER_VEKSLERSEGMENTATION_H
#define SUPERPIXELS_VEKSLER_VEKSLERSEGMENTATION_H

#include "commands/veksler_config.h"
#include "superpixel_segmentation.h"

namespace boost {
namespace serialization {

	template<class Archive>
	void serialize(Archive & ar, cv::Point & p, const unsigned int version)
	{
		ar & p.x;
		ar & p.y;
	}

} // namespace serialization
} // namespace boost

namespace superpixels {

class VekslerSegmentation : public SuperpixelSegmentation
{
public:
	VekslerSegmentation(vole::VekslerConfig config);
	VekslerSegmentation(int patchSize = 20, int type = 0, int iterations = 2, int lambda = 10);
	~VekslerSegmentation();

public:
	std::vector<Superpixel> superpixels(const cv::Mat_<cv::Vec3b>& image);
	std::vector<Superpixel> superpixels(const cv::Mat_<cv::Vec3d>& image);

	/*
	std::string getCacheIdentifier();
	std::string getStorageFilename(std::string dir, unsigned long int hashKey);
	*/
	void addCachingInfo(std::string input_file = std::string(), std::string input_file_16_bit = std::string(), std::string output_file = std::string(), bool isGraphical = false);

	virtual bool writeToCache(unsigned long int hashKey);
	virtual bool readFromCache(unsigned long int hashKey);

private:
	vole::VekslerConfig *localCfg;

	friend class boost::serialization::access;
	// When the class Archive corresponds to an output archive, the
	// & operator is defined similar to <<.  Likewise, when the class Archive
	// is a type of input archive the & operator is defined similar to >>.
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & height;
		ar & width;
		ar & localCfg; // FIXME add all information here that is required to restore the cache, as well as the hash number! 
		ar & _superpixels;
	}

};


} // namespace superpixels

#endif // SUPERPIXELS_VEKSLER_VEKSLERSEGMENTATION_H
