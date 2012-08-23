#ifndef SUPERPIXELS_COMMAND_OVERLAY_SEGMENTATION_CONFIG_H
#define SUPERPIXELS_COMMAND_OVERLAY_SEGMENTATION_CONFIG_H

#include "vole_config.h"

namespace vole {

class OverlaySegmentationConfig : public Config {
public:
	OverlaySegmentationConfig(const std::string& prefix = std::string());

public:
	virtual std::string getString() const;

protected:
	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST

public:
	std::string input_file;
	std::string input_file_16_bit;
	double max_intensity; // maximum intensity for a pixel
	double min_intensity; // minimum intensity for a pixel

	std::string segmentation_file;
	std::string annotation_color; // color to use to draw a superpixel
	std::string output_file;
	

private:
};

} // vole

#endif // SUPERPIXELS_COMMAND_OVERLAY_SEGMENTATION_CONFIG_H
