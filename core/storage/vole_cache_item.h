#ifndef VOLE_CACHE_ITEM_H
#define VOLE_CACHE_ITEM_H

#include <string>
#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace vole {
	class CacheItem {
		// must be privately declared for every subclass that implements 
		friend class boost::serialization::access;

		public:
			CacheItem(std::string cache_identifier, bool cache_enabled = true);
			virtual ~CacheItem(); 

			bool isCaching();
			std::string getCacheIdentifier();

		protected:

			void enableCaching();
			void disableCaching();
			virtual bool openCache(std::string cache_root = std::string(), bool force_dir_exists = false);
			virtual bool closeCache(std::string cache_file = std::string());

			virtual bool writeToCache(unsigned long int hashKey) = 0;
			virtual bool readFromCache(unsigned long int hashKey) = 0;
			virtual std::string getStorageFilename(std::string dir, unsigned long int hashKey, std::string suffix = ".cch", std::string desc = std::string());

			std::string cache_identifier;
			bool is_caching;
			std::string cache_root;
			bool cache_enabled;

			std::ifstream in;
			std::ofstream out;
	};


}
#endif // VOLE_CACHE_ITEM_H
