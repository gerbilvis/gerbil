#include "examples_config.h"

#ifdef VOLE_GUI
#include <QVBoxLayout>
#endif // VOLE_GUI

using namespace boost::program_options;

namespace vole {
	namespace examples {

ExamplesConfig::ExamplesConfig() : Config() {
	#ifdef WITH_BOOST
		initBoostOptions();
	#endif // WITH_BOOST
}

#ifdef VOLE_GUI
QWidget *ExamplesConfig::getConfigWidget() {
	this->initConfigWidget();
	QVBoxLayout *data_access_config = new QVBoxLayout();
	// (..)
	configWidget->setLayout(layout);
	layout->addLayout(data_access_config);
	configWidget->setLayout(layout);
	return configWidget;
}

void ExamplesConfig::updateValuesFromWidget() {
//	{ std::stringstream s; s << edit_min_ev_ratio->text().toStdString();
//		s >> minimum_eigenvalue_ratio; }
}
#endif// VOLE_GUI

#ifdef WITH_BOOST
void ExamplesConfig::initBoostOptions() {
	options.add_options()
		// global section
		("graphical", bool_switch(&isGraphical)->default_value(false),
			 "Show graphical output during runtime")
		("input,I", value(&input_file)->default_value(""),
			 "Input data file")
		("output,O", value(&output_directory)->default_value("/tmp/"),
		 "Output directory")
		;
}
#endif // WITH_BOOST

std::string ExamplesConfig::getString() const {
	std::stringstream s;
	s << "[examples]" << std::endl
		<< "verbose=" << verbosity
			<< " # verbosity level: 0 = silent, 1 = normal, 2 = much output, 3 = insane"
			<< std::endl
//		<< "command=" << command << " # the executed command" << std::endl
		<< "graphical=" << isGraphical << " # Show any graphical output during runtime" << std::endl
		<< "input=" << input_file << " # Argus left image" << std::endl
		<< "output=" << output_directory << " # Working directory" << std::endl
	;
	return s.str();
}



	} // namespace examples
} // namespace vole
