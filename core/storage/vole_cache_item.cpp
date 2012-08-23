#include "vole_cache_item.h"

#include <sstream>
#include <iostream>
#include <boost/version.hpp>

#if BOOST_VERSION >= 104500
	#ifndef BOOST_FILESYSTEM_VERSION
    	#define BOOST_FILESYSTEM_VERSION 3
	#endif
#else
	#ifndef BOOST_FILESYSTEM_VERSION
		#define BOOST_FILESYSTEM_VERSION 2
	#endif
	#error detected boost version is lower than 1.45, I think I will die.
#endif

#include <boost/filesystem.hpp>

using namespace boost::filesystem;

namespace vole {

	CacheItem::CacheItem(std::string cache_identifier, bool cache_enabled)
		: cache_identifier(cache_identifier), is_caching(false), cache_enabled(cache_enabled)
	{
	}

	CacheItem::~CacheItem()
	{
	}

	void CacheItem::enableCaching()
	{
		cache_enabled = true;
	}

	void CacheItem::disableCaching()
	{
		cache_enabled = false;
	}

	bool CacheItem::isCaching()
	{
		if (!cache_enabled) return false;
		return is_caching;
	}

	std::string CacheItem::getCacheIdentifier()
	{
		return cache_identifier;
	}

	bool CacheItem::openCache(std::string cache_root_dir, bool force_dir_exists)
	{
		if (!cache_enabled) return false;
		if (is_caching) return true;

		if (cache_root.length() > 0) {
			this->cache_root = cache_root;
		}
		// find your directory
		path p (cache_root);
		p /= cache_identifier;

		try
		{
			if (!exists(p))    // does p actually exist?
			{
				if (force_dir_exists)
				{
					std::cerr << "ERROR: Caching directory " << cache_root << " does not exist, aborted." << std::endl;
					return false;
				} else {
					if (!create_directories(p))
					{
						std::cerr << "ERROR: Caching directory " << cache_root << " does not exist and can not be created, aborted." << std::endl;
						return false;
					}
				}
			}

			if (!is_directory(p))
			{
				std::cerr << "ERROR: Caching directory " << cache_root << " is not a directory, aborted." << std::endl;
				return false;
			}
		}
		catch (const filesystem_error& ex)
		{
			std::cout << ex.what() << '\n';
			return false;
		}
		// OK, directory is there; note that it is not known if we can read from it or write to it
		is_caching = true;
		return true;
	}

	bool CacheItem::closeCache(std::string cache_file)
	{
		if (!cache_enabled) return false;
		if (!is_caching) return false;
		// method becomes important when an explicit list of cached items is
		// introduced. Up to now, it is only an entry point to writing the
		// files
		is_caching = false;
		return true;
	}

	std::string CacheItem::getStorageFilename(std::string dir, unsigned long int hashKey, std::string suffix, std::string desc)
	{
		std::stringstream s;
		path p(dir);
		p /= cache_identifier;

		s << hashKey;
		if (desc.length() > 0) {
			s << "_" << desc;
		}
		s << suffix;
		p /= s.str();
		return p.string();
	}

}

