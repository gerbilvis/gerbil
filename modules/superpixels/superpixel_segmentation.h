#ifndef SUPERPIXELS_SUPERPIXELSEGMENTATION_H
#define SUPERPIXELS_SUPERPIXELSEGMENTATION_H

#include "superpixel.h"
#include "vole_config.h"
#include "vole_cache_item.h"
#include <opencv2/core/core.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <sstream>

namespace superpixels {

/** The SuperpixelSegmentation class provides a base class for superpixel segmentation algorithms. */
class SuperpixelSegmentation : public vole::CacheItem
{
private:

public:
public:
	SuperpixelSegmentation(std::string descriptor = "", int height = -1, int width = -1);
	SuperpixelSegmentation( const SuperpixelSegmentation& to_copy );
	virtual ~SuperpixelSegmentation();

public:
	vole::Config *getConfig();

public:
	/** Segment an image into superpixels.
	 *  \param image Image in BGR format.
	 *  \return Vector of superpixels.
	 */
	virtual std::vector<Superpixel> superpixels(const cv::Mat_<cv::Vec3b>& image) = 0;
	/** Segment an image into superpixels.
	 *  \param image Image in BGR format.
	 *  \return Vector of superpixels.
	 */
	virtual std::vector<Superpixel> superpixels(const cv::Mat_<cv::Vec3d>& image) = 0;
	/** Create an image with superpixels marked by random colors.
	 *  \param image Image in BGR format.
	 *  \return Image with superpixels marked by random colors.
	 */
	cv::Mat_<cv::Vec3b> superpixelsImage(const cv::Mat_<cv::Vec3b>& image);
	/** Create an image with superpixels marked by random colors.
	 *  \param image Image in BGR format.
	 *  \return Image with superpixels marked by random colors.
	 */
	cv::Mat_<cv::Vec3b> superpixelsImage(const cv::Mat_<cv::Vec3d>& image);
	/** Create an image with superpixels marked by integer label numbers.
	 *  \param image Image in BGR format.
	 *  \return Image with superpixels marked by integer label numbers.
	 */
	cv::Mat_<int> superpixelsLabelImage(const cv::Mat_<cv::Vec3b>& image);
	/** Create an image with superpixels marked by integer label numbers.
	 *  \param image Image in BGR format.
	 *  \return Image with superpixels marked by integer label numbers.
	 */
	cv::Mat_<int> superpixelsLabelImage(const cv::Mat_<cv::Vec3d>& image);

	/* TODO damit Svens Code diese Klasse auch benutzen kann:
	 * virtual std::vector<Superpixel> superpixelsWithMask(const cv::Mat_<cv::Vec3b>& image) = 0;
	 */
 
//	virtual bool openCache(std::string cache_root_dir = std::string(), bool force_file_exists = false); // return always false in subclass if no caching is implemented
//	virtual bool writeCache(std::string cache_file = std::string()); // return always false in subclass if no caching is implemented
//	bool isCaching(); // keep caching_activated false if caching should be disabled at all
//	virtual std::string getCacheIdentifier() = 0;
//	virtual std::string getStorageFilename(std::string dir, unsigned long int hashKey) = 0;

//	virtual bool writeToCache(unsigned long int hashKey, std::vector<SuperpixelSegmentation::Superpixel> &superpixels, int rows, int cols);
//	virtual std::vector<SuperpixelSegmentation::Superpixel> readFromCache(unsigned long int hashKey);
	virtual bool writeToCache(unsigned long int hashKey) = 0;
	virtual bool readFromCache(unsigned long int hashKey) = 0;

public:
	/** Create an image with superpixels marked by random colors.
	 *  \param superpixels Vector of superpixels.
	 *  \param rows Number of rows of the image
	 *  \param cols Number of columns of the image
	 *  \return Image with superpixels marked by random colors.
	 */
	// TODO: comment on parameter collisionFree!
	// NOTE: Assumes that srand has been called if collisionFree == false
	cv::Mat_<cv::Vec3b> superpixelsImage(
		const std::vector<Superpixel>& superpixels,
		int rows,
		int cols,
		bool collisionFree = false);

	/** Create an image with superpixels marked by integer label numbers.
	 *  \param superpixels Vector of superpixels.
	 *  \param rows Number of rows of the image
	 *  \param cols Number of columns of the image
	 *  \return Image with superpixels marked by integer label numbers.
	 */
	cv::Mat_<int> & superpixelsLabelImage(const std::vector<Superpixel>& superpixels, int rows, int cols);

	// TODO: comment function!
	static void imageToSuperpixels(cv::Mat_<cv::Vec3b> &image, std::vector<Superpixel>& superpixels);

	/** Get the number of segments.
	 *  \return Number of segments, i.e. size of superpixels vector.
	 */
	int sizeSuperpixels() const { return _superpixels.size(); }

protected:
	vole::Config *config;

	// TODO: cleanup: why are height and width in this superclass? They fit more to
	//       patches than superpixels. And superpixels::Superpixel::bbox already 
	//       offers something similar.
	int height, width;
	std::vector<Superpixel> _superpixels;
	cv::Mat_<int> label_image_;

	bool caching_activated;
	std::string cache_dir;
	std::string cache_file;
	std::map<unsigned long int, std::string> cache;

	friend class boost::serialization::access;
	// When the class Archive corresponds to an output archive, the
	// & operator is defined similar to <<.  Likewise, when the class Archive
	// is a type of input archive the & operator is defined similar to >>.
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		// TODO: check if anything is missing here
		ar & config;
		ar & width;
		ar & height;
		for (unsigned int i = 0; i < _superpixels.size(); ++i) {
			std::vector<cv::Point> &coord = _superpixels[i].coordinates;
			for (unsigned int j = 0; j < coord.size(); ++j) {
				ar & coord[j].x;
				ar & coord[j].y;
			}
		}
		ar & label_image_;
		ar & caching_activated;
		ar & cache_dir;
		ar & cache_file;
		// FIXME add all information here that is required to restore the cache, as well as the hash number!
//		ar & cache;
	}
};

} // namespace superpixels

#endif // SUPERPIXELS_SUPERPIXELSEGMENTATION_H
