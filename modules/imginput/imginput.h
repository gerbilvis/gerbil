#ifndef IMGINPUT_H
#define IMGINPUT_H

#include "imginput_config.h"
#include <multi_img.h>
#include <cv.h>

namespace vole {

class ImgInput {
public:
	ImgInput(const ImgInputConfig& config) : config(config) {}

	multi_img execute();

private:
	const ImgInputConfig &config;

	void applyROI(multi_img &img);
};

} // namespace

#endif // IMGINPUT_H
