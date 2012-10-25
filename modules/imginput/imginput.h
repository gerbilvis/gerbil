#ifndef IMGINPUT_H
#define IMGINPUT_H

#include "imginput_config.h"
#include <multi_img.h>

namespace vole {

class ImgInput {
public:
	ImgInput(const ImgInputConfig& config) : config(config) {}

	multi_img execute();

	// provide both processed and original image (on same ROI)
	// first: processed, second: original
	std::pair<multi_img, multi_img> both();

private:
	const ImgInputConfig &config;

	void applyROI(multi_img &img);
};

} // namespace

#endif // IMGINPUT_H
