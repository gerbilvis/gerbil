#include "superpixel_segmentation.h"
#include "color.h"
#include <opencv2/highgui/highgui.hpp>
#include <boost/foreach.hpp>
#include <boost/version.hpp>
#include <iostream>
#include <sstream>

// boost property_tree exists only from boost 1.41 on; don't support earlier
// versions, as even debian squeeze contains boost >= 1.41
#if BOOST_VERSION >= 104100
#define SUPERPIXEL_SEGMENTATION_WITH_CACHING
#endif 

#ifdef SUPERPIXEL_SEGMENTATION_WITH_CACHING
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#endif // SUPERPIXEL_SEGMENTATION_WITH_CACHING

namespace superpixels {

SuperpixelSegmentation::SuperpixelSegmentation(std::string descriptor, int height, int width) :
	vole::CacheItem(descriptor), height(height), width(width)
{
	config = NULL;
	caching_activated = false;
}

SuperpixelSegmentation::SuperpixelSegmentation( const SuperpixelSegmentation& cpy )
	: vole::CacheItem( cpy.cache_identifier ), height( cpy.height ), width( cpy.width ), _superpixels( cpy._superpixels ), label_image_( cpy.label_image_ )
{}

SuperpixelSegmentation::~SuperpixelSegmentation()
{
	delete config;
	config = 0;
}

cv::Mat_<cv::Vec3b> SuperpixelSegmentation::superpixelsImage(const cv::Mat_<cv::Vec3b>& image)
{
	return superpixelsImage(superpixels(image), image.rows, image.cols);
}

cv::Mat_<cv::Vec3b> SuperpixelSegmentation::superpixelsImage(const cv::Mat_<cv::Vec3d>& image)
{
	return superpixelsImage(superpixels(image), image.rows, image.cols);
}

cv::Mat_<int> SuperpixelSegmentation::superpixelsLabelImage(const cv::Mat_<cv::Vec3b>& image)
{
	return superpixelsLabelImage(superpixels(image), image.rows, image.cols);
}

cv::Mat_<int> SuperpixelSegmentation::superpixelsLabelImage(const cv::Mat_<cv::Vec3d>& image)
{
	return superpixelsLabelImage(superpixels(image), image.rows, image.cols);
}

vole::Config *SuperpixelSegmentation::getConfig()
{
	return config;
}

/*
bool SuperpixelSegmentation::openCache(std::string cache_file, bool force_file_exists)
{
// no caching/ return
#ifndef SUPERPIXEL_SEGMENTATION_WITH_CACHING
	return false;
#else // SUPERPIXEL_SEGMENTATION_WITH_CACHING
	if (cache_file.length() > 0) {
		this->cache_file = cache_file;
	}
	using boost::property_tree::ptree;
	using namespace boost::property_tree;
	using namespace boost::property_tree::xml_parser;
	ptree pt;
	bool is_initialized = false;
	try {
		read_xml(this->cache_file, pt);
	} catch (const xml_parser_error &e) {
		if (force_file_exists) {
			std::cerr << "Error when reading file " << this->cache_file << ":" << std::endl << "  ";
			std::cerr << e.what() << std::endl;
			return false;
		} else {
			// initialize empty cache: (it is empty, nothing to do)
			is_initialized = true;
		}
	}
	if (!is_initialized) {
		caching_activated = true;
		BOOST_FOREACH(const ptree::value_type &k, pt) {
			unsigned long int hashkey;
			std::stringstream s;
			std::string key = k.first.data();
			s << key.substr(key.find_last_of('.')+1);
			s >> hashkey;
			cache[hashkey] = k.second.data();
		}
	}
	return true;
#endif // SUPERPIXEL_SEGMENTATION_WITH_CACHING
}
*/

/*
bool SuperpixelSegmentation::writeCache(std::string cache_file)
{
// no caching/ return
#ifndef SUPERPIXEL_SEGMENTATION_WITH_CACHING
	return false;
#else // SUPERPIXEL_SEGMENTATION_WITH_CACHING
	std::string write_file = this->cache_file;
	if (cache_file.length() > 0) write_file = cache_file;
	using boost::property_tree::ptree;
	using namespace boost::property_tree;
	using namespace boost::property_tree::xml_parser;
	ptree pt;

	typedef std::pair<unsigned long int, std::string> cache_entry;
	BOOST_FOREACH(const cache_entry &k, cache) {
		std::stringstream s; s << "cache." << getCacheIdentifier() << "." << k.first;
		pt.push_back(ptree::value_type(s.str(), ptree(k.second)));
	}

	// Write the property tree to the XML file.
	write_xml(write_file, pt);
	return true;
#endif // SUPERPIXEL_SEGMENTATION_WITH_CACHING
}

bool SuperpixelSegmentation::isCaching()
{
	return caching_activated;
}
*/

/*
bool SuperpixelSegmentation::writeToCache(unsigned long int hashKey, std::vector<SuperpixelSegmentation::Superpixel> &superpixels, int rows, int cols)
{
// no caching/ return
#ifndef SUPERPIXEL_SEGMENTATION_WITH_CACHING
	return false;
#else // SUPERPIXEL_SEGMENTATION_WITH_CACHING
	std::string file_name = getStorageFilename(cache_dir, hashKey);
	if (file_name.length() < 1) return false;

	// convert image:
	cv::Mat_<cv::Vec3b> img = superpixelsImage(superpixels, rows, cols, true);
	if (img.data == NULL) return false;
	cv::imwrite(file_name, img);
	cache[hashKey] = file_name;
	return true;
#endif // SUPERPIXEL_SEGMENTATION_WITH_CACHING
}

std::vector<SuperpixelSegmentation::Superpixel> SuperpixelSegmentation::readFromCache(unsigned long int hashKey)
{
// no caching/ return
#ifndef SUPERPIXEL_SEGMENTATION_WITH_CACHING
	return std::vector<SuperpixelSegmentation::Superpixel>();
#else // SUPERPIXEL_SEGMENTATION_WITH_CACHING
	// look for the key
	if (cache.find(hashKey) == cache.end()) return std::vector<SuperpixelSegmentation::Superpixel>(); // no entry :(

	std::string image_file_name = cache[hashKey];
	cv::Mat_<cv::Vec3b> img = cv::imread(image_file_name);

	if (img.data == NULL) return std::vector<SuperpixelSegmentation::Superpixel>(); // weird error - cache outdated?

	std::vector<SuperpixelSegmentation::Superpixel> tmp;
	imageToSuperpixels(img, tmp);
	return tmp;
#endif // SUPERPIXEL_SEGMENTATION_WITH_CACHING
}
*/

cv::Mat_<cv::Vec3b> SuperpixelSegmentation::superpixelsImage(
	const std::vector<Superpixel>& superpixels,
	int rows,
	int cols,
	bool collisionFree)
{
	cv::Mat_<cv::Vec3b> superpixelsImage(rows, cols);

	rbase::Color c;
	for (unsigned int i = 0; i < superpixels.size(); i++) {
		Superpixel superpixel = superpixels[i];

		if (!collisionFree) {
			// NOTE: Assumes that srand has been called!
			c.initRandomly();
		} else {
			c.initFromId(i);
		}

		for (unsigned int j = 0; j < superpixel.coordinates.size(); j++) {
			cv::Point coord = superpixel.coordinates[j];

			superpixelsImage(coord.y, coord.x)[0] = c.r;
			superpixelsImage(coord.y, coord.x)[1] = c.g;
			superpixelsImage(coord.y, coord.x)[2] = c.b;
		}
	}

	return superpixelsImage;
}

cv::Mat_< int > & SuperpixelSegmentation::superpixelsLabelImage(
	const std::vector<Superpixel>& superpixels,
	int rows,
	int cols )
{
	if( !label_image_.empty() ){
		return label_image_;
	}

//	label_image_ = cv::Mat_<int>(rows, cols);
	label_image_.create( rows, cols );

	for( unsigned int i = 0; i < superpixels.size(); i++ ){
		for( unsigned int j = 0; j < superpixels[i].coordinates.size(); j++ ){
			cv::Point coord = superpixels[i].coordinates[j];
			// we can use i as our label number
			label_image_(coord.y, coord.x) = i;
		}
	}

	return label_image_;
}

// FIXME inefficient implementation!
void SuperpixelSegmentation::imageToSuperpixels(
	cv::Mat_<cv::Vec3b> &image,
	std::vector<Superpixel>& superpixels)
{
	std::map<unsigned long int, int> index;
	superpixels.clear();
	int count = 0;
	for (int y = 0; y < image.rows; ++y) {
		for (int x = 0; x < image.cols; ++x) {
			unsigned long int cur = rbase::Color::rgbToId(image[y][x][2], image[y][x][1], image[y][x][0]);
			if (index.find(cur) == index.end()) {
				index[cur] = count;
				superpixels.push_back(Superpixel());
				count++;
			}
			superpixels[index[cur]].coordinates.push_back(cv::Point(x, y));
		}
	}
}

} // namespace superpixels
