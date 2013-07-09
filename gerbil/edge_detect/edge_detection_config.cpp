#include "edge_detection_config.h"
#include <time.h>

#ifdef WITH_BOOST
using namespace boost::program_options;
#endif

namespace vole {

EdgeDetectionConfig::EdgeDetectionConfig(const std::string& p)
	: Config(p), similarity(prefix + "similarity")
{
	som_file = "";
	hack3d = false;
	width = 32;
	height = 32;
	maxIter = 40000;
	learnStart = 0.1;
	learnEnd = 0.001;
	radiusStart = 4.;
	radiusEnd = 1.;
	seed = time(NULL);

	#ifdef WITH_BOOST
		initBoostOptions();
	#endif // WITH_BOOST
}

#ifdef WITH_BOOST
void EdgeDetectionConfig::initBoostOptions() {
	options.add_options()
		(key("som_input"), value(&som_file)->default_value(som_file),
			 "When set, read given multispectral image file to initialize SOM"
			 " instead of training")
		(key("hack3d"), bool_switch(&hack3d)->default_value(hack3d),
			 "Use hack to have 3D som of size width a*a*a, specify "
			 "width=a, height=1")
		(key("width"), value(&width)->default_value(width),
			"Width of the SOM")
		(key("height"), value(&height)->default_value(height),
			"Height of the SOM")
		(key("maxIter"), value(&maxIter)->default_value(maxIter),
			"Number of training iterations for the SOM")
		(key("learnStart"), value(&learnStart)->default_value(learnStart),
			"Learning rate at the beginning")
		(key("learnEnd"), value(&learnEnd)->default_value(learnEnd),
			"Learning rate at the end of the training process")
		(key("radiusStart"), value(&radiusStart)->default_value(radiusStart),
			"Initial neighborhood radius")
		(key("radiusEnd"), value(&radiusEnd)->default_value(radiusEnd),
			"Neighborhood radius at the end of the training process")
		(key("seed"), value(&seed)->default_value(seed),
			"Seed value of random number generators")
		;
	options.add(similarity.options);

	if (prefix_enabled)	// skip input/output options
		return;

	options.add_options()
		(key("input,I"), value(&input_file)->default_value("input.png"),
		 "Image file to process")
		(key("output,O"), value(&output_dir)->default_value("/tmp/"),
		 "Output directory")
		(key("output_som"), bool_switch(&output_som)->default_value(false),
			 "Output trained SOM as a multispectral image")
		;
}
#endif // WITH_BOOST

std::string EdgeDetectionConfig::getString() const {
	std::stringstream s;	// TODO
	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else {
		s << "input=" << input_file << " # Image to process" << std::endl
		  << "output=" << output_dir << " # Working directory" << std::endl
		;
	}
	s	<< "som_input=" << som_file << " # SOM image file instead of training" << std::endl
		<< "width=" << width << " # Width of the SOM" << std::endl
		<< "height=" << height << " # Height of the SOM" << std::endl
		<< "hack3d=" << hack3d << " # use hack for 3D SOM" << std::endl
		<< "maxIter=" << maxIter << " # Number of training iterations for the SOM" << std::endl
		<< "learnStart=" << learnStart << " # Start value for the learning rate in SOM" << std::endl
		<< "learnEnd=" << learnEnd << " # End value for the learning rate in SOM" << std::endl
		<< "radiusStart=" << radiusStart << " # Start value for the radius of the neighborhood function in SOM" << std::endl
		<< "radiusEnd=" << radiusEnd << " # End value for the radius of the neighborhood function in SOM" << std::endl
		<< "seed=" << seed << " # Seed value of random number generators " << std::endl
        << similarity.getString();
	;
	return s.str();
}

}
