#include "quickshift_config.h"

#include <iostream>
#include <fstream>
#include <ctime> 
#include <cstdlib>

#ifdef VOLE_GUI
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#endif // VOLE_GUI

#ifdef WITH_BOOST_PROGRAM_OPTIONS
using namespace boost::program_options;
#endif // WITH_BOOST_PROGRAM_OPTIONS

QuickshiftConfig::QuickshiftConfig(const std::string& prefix)
 : Config(prefix) {
	#ifdef WITH_BOOST
		initBoostOptions();
	#endif // WITH_BOOST
}

#ifdef VOLE_GUI
// TODO dummy method
QWidget *QuickshiftConfig::getConfigWidget() {
	this->initConfigWidget();
	return configWidget;
}

// TODO dummy method
void QuickshiftConfig::updateValuesFromWidget() {
}

#endif// VOLE_GUI

#ifdef WITH_BOOST
void QuickshiftConfig::initBoostOptions() {
	options.add_options()
		(key("use_chroma"), bool_switch(&use_chroma)->default_value(false),
			"operate on a 2D-chromaticity space")
		(key("kernel_size"), value(&kernel_size)->default_value(3),
			 "kernel size for quickshift")
		(key("max_dist_multiplier"), value(&max_dist_multiplier)->default_value(3),
		                   "multiplier for the computation of max_dist = kernel_size * multipl")
		(key("merge_th"), value(&merge_threshold)->default_value(4),
		                   "merge threshold, choose between 1 and 19")
		;
	
	if (prefix_enabled) // disable input/output parameters
		return;

	options.add_options()
		(key("input,I"), value(&input_file)->default_value("input.png"),
		 "Image to process")
		(key("output,O"), value(&output_file)->default_value("/tmp/parents.png"),
		 "Output file for the samples")
		;
}
#endif // WITH_BOOST


std::string QuickshiftConfig::getString() const {
	std::stringstream s;
	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else {
		s << "input=" << input_file << " # Image to process" << std::endl
		  << "output=" << output_file << " # Working directory" << std::endl
		;
	}
	s << "kernel_size=" << kernel_size << " # kernel size for quickshift" << std::endl
	  << "max_dist_multiplier=" << max_dist_multiplier
	  << " # multiplier for the computation of max_dist = kernel_size * multipl" << std::endl
	  << "merge_th=" << merge_threshold
	  << " # merge threshold, choose between 1 and 19" << std::endl
		;
	return s.str();
}


