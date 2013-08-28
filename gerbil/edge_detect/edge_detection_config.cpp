#include "edge_detection_config.h"
#include <time.h>

#ifdef WITH_BOOST
using namespace boost::program_options;
#endif

namespace vole {

#ifdef WITH_BOOST
ENUM_MAGIC(somtype)
#endif

EdgeDetectionConfig::EdgeDetectionConfig(const std::string& p)
	: Config(p), similarity(prefix + "similarity")
{
	som_file = "";
	sidelength = 32;
	granularity = 0.06; // 1081 neurons
	type = SOM_SQUARE;
	maxIter = 40000;
	learnStart = 0.1;
	learnEnd = 0.001; // TODO: we stop updating, when weight is < 0.01 !!!
	sigmaStart = 4.; // ratio sigmaStart : sigmaEnd should be about 4 : 1
	sigmaEnd = 1.;
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
		(key("sidelength"), value(&sidelength)->default_value(sidelength),
			"Sidelength of line / square / cube of the SOM")
		(key("granularity"), value(&granularity)->default_value(granularity),
			"Distance between neurons inside the cone")
		(key("type"), value(&type)->default_value(type),
			"Layout of the neurons in the SOM: line, square, cube, cone")
		(key("maxIter"), value(&maxIter)->default_value(maxIter),
			"Number of training iterations for the SOM")
		(key("learnStart"), value(&learnStart)->default_value(learnStart),
			"Learning rate at the beginning")
		(key("learnEnd"), value(&learnEnd)->default_value(learnEnd),
			"Learning rate at the end of the training process")
		(key("sigmaStart"), value(&sigmaStart)->default_value(sigmaStart),
			"Initial neighborhood radius")
		(key("sigmaEnd"), value(&sigmaEnd)->default_value(sigmaEnd),
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
		<< "sidelength=" << sidelength << " # Sidelength of line / square / cube of the SOM" << std::endl
		<< "granularity=" << granularity << " # Distance between neurons inside the cone" << std::endl
		<< "type=" << type << " # Layout of the neurons in the SOM" << std::endl
		<< "maxIter=" << maxIter << " # Number of training iterations for the SOM" << std::endl
		<< "learnStart=" << learnStart << " # Start value for the learning rate in SOM" << std::endl
		<< "learnEnd=" << learnEnd << " # End value for the learning rate in SOM" << std::endl
		<< "sigmaStart=" << sigmaStart << " # Start value for sigma of the gaussian-like neighborhood function that describes the decrease of the learning rate" << std::endl
		<< "sigmaEnd=" << sigmaEnd << " # End value for sigma of the gaussian-like neighborhood function that describes the decrease of the learning rate" << std::endl
		<< "seed=" << seed << " # Seed value of random number generators " << std::endl
        << similarity.getString();
	;
	return s.str();
}

}
