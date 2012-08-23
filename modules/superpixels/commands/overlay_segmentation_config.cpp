#include "overlay_segmentation_config.h"

#ifdef WITH_BOOST_PROGRAM_OPTIONS
using namespace boost::program_options;
#endif // WITH_BOOST_PROGRAM_OPTIONS

namespace vole {

OverlaySegmentationConfig::OverlaySegmentationConfig(const std::string& prefix) :
	Config(prefix)
{
	#ifdef WITH_BOOST
		initBoostOptions();
	#endif // WITH_BOOST
}

std::string OverlaySegmentationConfig::getString() const
{
	std::stringstream s;

	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else {
		s << "input=" << input_file << " # Image to process" << std::endl
		  << "i16=" << input_file_16_bit << " # 16 bit RGB input image" << std::endl
		  << "max_intensity=" << max_intensity << " # Maximum intensity for a pixel" << std::endl
		  << "min_intensity=" << min_intensity << " # Minimum intensity for a pixel" << std::endl
		  << "segmentation=" << segmentation_file << " # Segmentation image (as written by superpixelsegmentation)" << std::endl
		  << "output=" << output_file << " # segmentation output image" << std::endl
		;
	}
	s << "annotation" << annotation_color << " # color of the annotations, encoded as R,G,B string (w/o spaces)." << std::endl
	;

	return s.str();
}

#ifdef WITH_BOOST
void OverlaySegmentationConfig::initBoostOptions()
{
	if (!prefix_enabled) {
		options.add_options()
			(key("input,I"), value(&input_file)->default_value(""), "Image to process")
			(key("i16,J"), value(&input_file_16_bit)->default_value(""), "16 bit RGB input image")
			(key("max_intensity"), value(&max_intensity)->default_value(4095, "4095"), "Maximum intensity for a pixel")
			(key("min_intensity"), value(&min_intensity)->default_value(0, "0"), "Minimum intensity for a pixel")
			(key("segmentation,S"), value(&segmentation_file)->default_value(""), "Segmentation image (as written by superpixelsegmentation)")
			(key("output,O"), value(&output_file)->default_value(""), "segmentation output image")
		;
	}
	options.add_options()
		(key("annotation,A"), value(&annotation_color)->default_value(""), "color of the annotations, encoded as R,G,B string (w/o spaces).")
	;

}
#endif // WITH_BOOST

} // vole
