#ifndef IMGINPUT_H
#define IMGINPUT_H

#include "imginput_config.h"
#include <multi_img.h>
#include <vector>

namespace imginput {

class ImgInput {
public:
	ImgInput(const ImgInputConfig& config) : config(config) { }

	// Wrapper around readFile()
	multi_img::ptr execute();

	static bool parseROIString(const std::string &str, std::vector<int> &vals);

private:
	const ImgInputConfig &config;

	void applyROI(multi_img &img, std::vector<int> &vals);

	void cropSpectrum(multi_img &img);

	void applyIllumination(multi_img& img_ptr);
};

} // namespace

#endif // IMGINPUT_H
