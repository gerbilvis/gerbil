#ifndef SUPERPIXELS_PATCHES_PATCHESSEGMENTATION_H
#define SUPERPIXELS_PATCHES_PATCHESSEGMENTATION_H

#include "superpixel_segmentation.h"
#include "commands/patches_config.h"
//#include <boost/archive/text_oarchive.hpp>
//#include <boost/archive/text_iarchive.hpp>

namespace superpixels {

class PatchesSegmentation : public SuperpixelSegmentation
{
public:
	/** Constructor for a given config, a given mask, and a given exclution area. The caller must take care of the mask's and the excluded area's pointers.
	 * \param config a config of type vole::PatchesConfig that contains the patches specific options.
	 * \param mask a mask for pixels that shall be excluded from patch creation, e.g. too saturated or too dark pixels. Caller must take care of the given pointer.
	 * \param excluded_area an area marked for exclution, e.g. from a specific object in the image, like a ball. Caller must take care of the given pointer.
	 */
	PatchesSegmentation( vole::PatchesConfig config, const cv::Mat_<unsigned char>* mask = 0, const cv::Mat_< unsigned char >* excluded_area = 0 );
	/** Constructor for a given config, a given segmenter, a given mask, and a given exclution area. The caller must take care of the mask's and the excluded area's pointers.
	 * \param config a config of type vole::PatchesConfig that contains the patches specific options.
	 * \param segmenter a general, real segmentation method, e.g. Felzenszwalb or Veksler. The class will take care of this pointer, i.e. delete it in its destructor.
	 * \param mask a mask for pixels that shall be excluded from patch creation, e.g. too saturated or too dark pixels. Caller must take care of the given pointer.
	 * \param excluded_area an area marked for exclution, e.g. from a specific object in the image, like a ball. Caller must take care of the given pointer.
	 */
	PatchesSegmentation( vole::PatchesConfig config, SuperpixelSegmentation* segmenter, const cv::Mat_<unsigned char>* mask = 0, const cv::Mat_< unsigned char >* excluded_area = 0 );
	/** Destructor will delete the member variables, i.e. the given SuperpixelSegmentation* segmenter, provided in constructor or setSegmenter( SuperpixelSegmentation* segmenter );
	 */
	~PatchesSegmentation();

public:
	/** Function to set the internal concrete segmenter class, e.g. FelzenszwalbSegmentation or VekslerSegmentation.
	 * \param segmenter a pointer to an abstract SuperpixelSegmentation object. The class will take care of the pointer, i.e. delete it in its destructor.
	 */
	void setSegmenter( SuperpixelSegmentation* segmenter );

	std::vector<Superpixel> useSegmenter( const cv::Mat_<cv::Vec3b>& image );
	std::vector<Superpixel> useSegmenter( const cv::Mat_<cv::Vec3d>& image );

	std::vector<Superpixel> superpixels( const cv::Mat_<cv::Vec3b>& image );
	std::vector<Superpixel> superpixels( const cv::Mat_<cv::Vec3d>& image );

	int sizeSegmenterSuperpixels() const { return segmenter_superpixels_.size(); }

#define GPR_VERSION 1
#if GPR_VERSION==1
	void getPatchRandomly( int rows, int cols, Superpixel& patch );
#elif GPR_VERSION==2
	// DEBUG
	Superpixel getPatchRandomly();
	void setImage( cv::Mat_<cv::Vec3b> image ) { image_ = image; };
	cv::Mat_<cv::Vec3b> image_;
#else
	void getPatchRandomly( const cv::Mat_<cv::Vec3b>& image, Superpixel& patch );
#endif
	// DEBUG END

	/*
	std::string getCacheIdentifier();
	std::string getStorageFilename(std::string dir, unsigned long int hashKey);
	*/

	virtual bool writeToCache( unsigned long int hashKey );
	virtual bool readFromCache( unsigned long int hashKey );

private:
	SuperpixelSegmentation* segmenter_;
	bool has_segmenter_;
	std::vector< Superpixel > segmenter_superpixels_;
	bool has_segmentation_;
	vole::PatchesConfig *localCfg; // TODO: why this solution with a pointer and not like it's done for IEBV, e.g.?
	const cv::Mat_< unsigned char >* mask_;
	const cv::Mat_< unsigned char >* excluded_area_;

#if GPR_VERSION==1
	// DEBUG
	void intersect( int rows, int cols, Superpixel& patch );
	// DEBUG END
#else
	void intersect( const cv::Mat_< cv::Vec3b >& image, Superpixel& patch );
#endif

	friend class boost::serialization::access;
	// When the class Archive corresponds to an output archive, the
	// & operator is defined similar to <<.  Likewise, when the class Archive
	// is a type of input archive the & operator is defined similar to >>.
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & localCfg; // FIXME add all information here that is required to restore the cache, as well as the hash number! 
		for (unsigned int i = 0; i < _superpixels.size(); ++i) {
			std::vector<cv::Point> &coord = _superpixels[i].coordinates;
			for (unsigned int j = 0; j < coord.size(); ++j) {
				ar & coord[j].x;
				ar & coord[j].y;
			}
		}
		ar & segmenter_;
		ar & has_segmenter_;
		for (unsigned int i = 0; i < segmenter_superpixels_.size(); ++i) {
			std::vector<cv::Point> &coord = segmenter_superpixels_[i].coordinates;
			for (unsigned int j = 0; j < coord.size(); ++j) {
				ar & coord[j].x;
				ar & coord[j].y;
			}
		}
		ar & has_segmentation_;
		// FIXME the following two necessary?
//		ar & mask_;
//		ar & excluded_area_;
	}
};

} // namespace superpixels

#endif // SUPERPIXELS_PATCHES_PATCHESSEGMENTATION_H
