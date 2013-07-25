#ifndef IMGINPUT_H
#define IMGINPUT_H

#include "imginput_config.h"
#include <multi_img.h>
#include <vector>

namespace vole {

class ImgInput {
public:
	ImgInput(const ImgInputConfig& config) : config(config) { }

	// Wrapper around readFile()
	multi_img::ptr execute();

	static bool parseROIString(const std::string &str, std::vector<int> &vals);

private:
	const ImgInputConfig &config;

	void applyROI(multi_img::ptr &img_ptr, std::vector<int> &vals);

	void cropSpectrum(multi_img::ptr &img_ptr);
};

} // namespace

#endif // IMGINPUT_H
