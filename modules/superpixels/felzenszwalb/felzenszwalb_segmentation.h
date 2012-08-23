#ifndef SUPERPIXELS_FELZENSZWALB_FELZENSWALBSEGMENTATION_H
#define SUPERPIXELS_FELZENSZWALB_FELZENSWALBSEGMENTATION_H

#include "superpixel_segmentation.h"
#include "commands/felzenszwalb_config.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace superpixels {

class FelzenszwalbSegmentation : public SuperpixelSegmentation
{
public:
	FelzenszwalbSegmentation(vole::FelzenszwalbConfig config);
	FelzenszwalbSegmentation(int min_size, int kThreshold = 500, double sigma = 0.5);
	~FelzenszwalbSegmentation();

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
	vole::FelzenszwalbConfig *localCfg;

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
		for (unsigned int i = 0; i < _superpixels.size(); ++i) {
			std::vector<cv::Point> &coord = _superpixels[i].coordinates;
			for (unsigned int j = 0; j < coord.size(); ++j) {
				ar & coord[j].x;
				ar & coord[j].y;
			}
		}
	}

};

} // namespace superpixels

#endif // SUPERPIXELS_FELZENSZWALB_FELZENSWALBSEGMENTATION_H
