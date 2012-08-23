#include <imginput.h>
#include <multi_img.h>
#include <string.h>
#include <vector>

using std::vector;

namespace vole {

multi_img ImgInput::execute() {
	multi_img img(config.file);

	// return empty image on failure
	if (img.empty())
		return img;

	// apply ROI
	if (!config.roi.empty())
		applyROI(img);

#ifdef WITH_GERBIL_COMMON
	// compute gradient
	if (config.gradient) {
		img.apply_logarithm();
		img = img.spec_gradient();
	}

	// reduce number of bands
	if (config.bands > 0 && config.bands < (int) img.size()) {
		img = img.spec_rescale(config.bands);
	}
#endif

	return img;
}

void ImgInput::applyROI(multi_img &img) {
	vector<int> vals;
	const std::string &str = config.roi;
	std::string::size_type prev_pos = 0, pos = 0;
	while ((pos = str.find(':', pos)) != std::string::npos) {
		vals.push_back(atoi(str.substr(prev_pos, pos - prev_pos).c_str()));
		prev_pos = ++pos;
	}
	vals.push_back(atoi(str.substr(prev_pos, pos - prev_pos).c_str()));

	if (vals.size() != 4) {
		std::cerr << "Ignoring invalid ROI specification" << std::endl;
		return;
	}

	img = multi_img(img, cv::Rect(vals[0], vals[1], vals[2], vals[3]));
}

} //namespace
