#include "veksler_config.h"

#ifdef WITH_BOOST_PROGRAM_OPTIONS
using namespace boost::program_options;
#endif // WITH_BOOST_PROGRAM_OPTIONS

namespace vole {

VekslerConfig::VekslerConfig(const std::string& prefix) :
	Config(prefix)
{
	#ifdef WITH_BOOST
		initBoostOptions();
	#endif // WITH_BOOST
}

std::string VekslerConfig::getString() const
{
	std::stringstream s;

	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else {
		s << "input=" << input_file << " # Image to process" << std::endl;
		s << "i16=" << input_file_16_bit << " # 16 bit RGB input image" << std::endl;
		s << "max_intensity=" << max_intensity << " # Maximum intensity for a pixel" << std::endl;
		s << "min_intensity=" << min_intensity << " # Minimum intensity for a pixel" << std::endl;
		s << "output=" << output_file << " # segmentation output image" << std::endl;
	}

	s << "graphical=" << isGraphical << " # Show any graphical output during runtime" << std::endl;
	s << "deterministic_coloring=" << deterministic_coloring
		<< " # output image encodes segment numbers (pro: reconstruction of segments, con: visually not appealing)"
		<< std::endl;
	s << "size=" << patchSize << " # Edge length of patches in pixels" << std::endl;
	s << "type=" << type << " # Segmentation type" << std::endl;
	s << "iterations=" << iterations << " # Number of iterations" << std::endl;
	s << "lambda=" << lambda << " # Parameter of energy function" << std::endl;

	return s.str();
}

#ifdef WITH_BOOST
void VekslerConfig::initBoostOptions()
{
	if (!prefix_enabled) {
		options.add_options()(key("input,I"), value(&input_file)->default_value(""), "Image to process");
		options.add_options()(key("i16,J"), value(&input_file_16_bit)->default_value(""), "16 bit RGB input image");
		options.add_options()(key("max_intensity"), value(&max_intensity)->default_value(4095, "4095"), "Maximum intensity for a pixel");
		options.add_options()(key("min_intensity"), value(&min_intensity)->default_value(0, "0"), "Minimum intensity for a pixel");
		options.add_options()(key("output,O"), value(&output_file)->default_value(""), "segmentation output image");
	}

	options.add_options()(key("graphical"), bool_switch(&isGraphical)->default_value(false), "Show any graphical output during runtime");
	options.add_options()(key("deterministic_coloring"), bool_switch(&deterministic_coloring)->default_value(false),
		"output image encodes segment numbers (pro: reconstruction of segments, con: visually not appealing)");

	options.add_options()(key("size"),value(&patchSize)->default_value(20), "Edge length of patches in pixels");
	options.add_options()(key("type"),value(&type)->default_value(0), "Segmentation type");
	options.add_options()(key("iterations"), value(&iterations)->default_value(2), "Number of iterations");
	options.add_options()(key("lambda"), value(&lambda)->default_value(10), "Parameter of energy function");
}
#endif // WITH_BOOST

} // vole
