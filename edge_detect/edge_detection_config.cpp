#include "edge_detection_config.h"
#include <time.h>

using namespace boost::program_options;

namespace vole {

EdgeDetectionConfig::EdgeDetectionConfig(const std::string& p)
	: Config(p), similarity(prefix + "similarity")
{

	#ifdef WITH_BOOST
		initBoostOptions();
	#endif // WITH_BOOST
}

#ifdef WITH_BOOST
void EdgeDetectionConfig::initBoostOptions() {
	options.add_options()
		(key("hack3d"), bool_switch(&hack3d)->default_value(false),
			 "Use hack to have 3D som of size width a*a*a, specify "
			 "width=a, height=1")
		(key("somWidth"), value(&width)->default_value(32),
			"Width of the SOM")
		(key("somHeight"), value(&height)->default_value(32),
			"Height of the SOM")
		(key("somMaxIter"), value(&maxIter)->default_value(40000),
			"Number of training iterations for the SOM")
		(key("somLearnStart"), value(&learnStart)->default_value(0.1),
			"Learning rate at the beginning")
		(key("somLearnEnd"), value(&learnEnd)->default_value(0.001),
			"Learning rate at the end of the training process")
		(key("somRadiusStart"), value(&radiusStart)->default_value(4.),
			"Initial neighborhood radius")
		(key("somRadiusEnd"), value(&radiusEnd)->default_value(1.),
			"Neighborhood radius at the end of the training process")
		(key("seed"), value(&seed)->default_value(time(NULL)),
			"Seed value of random number generators")
		;
	options.add(similarity.options);

	if (prefix_enabled)	// skip input/output options
		return;

	options.add_options()
		(key("input,I"), value(&input_file),
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
	s	<< "somWidth=" << width << " # Width of the SOM" << std::endl
		<< "somHeight=" << height << " # Height of the SOM" << std::endl
		<< "hack3d=" << hack3d << " # use hack for 3D SOM" << std::endl
		<< "somMaxIter=" << maxIter << " # Number of training iterations for the SOM" << std::endl
		<< "somLearnStart=" << learnStart << " # Start value for the learning rate in SOM" << std::endl
		<< "somLearnEnd=" << learnEnd << " # End value for the learning rate in SOM" << std::endl
		<< "somRadiusStart=" << radiusStart << " # Start value for the radius of the neighborhood function in SOM" << std::endl
		<< "somRadiusEnd=" << radiusEnd << " # End value for the radius of the neighborhood function in SOM" << std::endl
		<< "seed=" << seed << " # Seed value of random number generators " << std::endl
        << similarity.getString();
	;
	return s.str();
}

}
