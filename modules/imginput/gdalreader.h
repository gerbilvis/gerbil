#ifdef WITH_GDAL

#ifndef GDALREADER_H
#define GDALREADER_H

#include <string>
#include <multi_img.h>
#include "imginput.h"
#include "imginput_config.h"

namespace vole {

class GdalReader {
public:
	GdalReader(const ImgInputConfig& config)
		: config(config) { }

	multi_img::ptr readFile();

private:
	const ImgInputConfig &config;

	static bool tryConvert(std::string const&, float&);
};

} // namespace

#endif // GDALREADER_H

#endif // WITH_GDAL
