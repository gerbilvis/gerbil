#include "similarity_measure_config.h"

#ifdef VOLE_GUI
#include <QVBoxLayout>
#endif // VOLE_GUI

using namespace boost::program_options;

namespace vole {

SimilarityMeasureConfig::SimilarityMeasureConfig(const std::string &prefix) : Config(prefix) {
	#ifdef WITH_BOOST
		initBoostOptions();
	#endif // WITH_BOOST
}

#ifdef VOLE_GUI
QWidget *SimilarityMeasureConfig::getConfigWidget() {
	this->initConfigWidget();
	QVBoxLayout *data_access_config = new QVBoxLayout();
	// (..)
	configWidget->setLayout(layout);
	layout->addLayout(data_access_config);
	configWidget->setLayout(layout);
	return configWidget;
}

void SimilarityMeasureConfig::updateValuesFromWidget() {
//	{ std::stringstream s; s << edit_min_ev_ratio->text().toStdString();
//		s >> minimum_eigenvalue_ratio; }
}
#endif// VOLE_GUI

#ifdef WITH_BOOST
void SimilarityMeasureConfig::initBoostOptions() {
	options.add_options()
		// global section
		(key("graphical"), bool_switch(&isGraphical)->default_value(false),
			 "Show graphical output during runtime")
		(key("metrics,M"), value(&selected_metrics)->default_value("ms"),
			"\",\"-separated list of metrics to use for the template matching, possible values are ms,mrsd,ncc,cch,nmi,msh,mih,emd,gd.")
	;
	if (prefix_enabled)
		return;

	options.add_options()
		(key("input1,I"), value(&input_file1)->default_value(""), "Input data file")
		(key("input2,J"), value(&input_file2)->default_value(""), "Input data file 2")
		(key("win1,W"), value(&win1)->default_value(""),
			 "\",\"-separated list of coordinates of window1. win1 and win2 must have the same size")
		(key("win2,X"), value(&win2)->default_value(""),
			 "\",\"-separated list of coordinates of window2. win1 and win2 must have the same size")
		(key("output,O"), value(&output_directory)->default_value("/tmp/"), "Output directory")
	;
}
#endif // WITH_BOOST

std::string SimilarityMeasureConfig::getString() const {
	std::stringstream s;
	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else { // no prefix, i.e. we handle I/O parameters
		s << "input1=" << input_file1 << " # Image to process" << std::endl
		  << "input2=" << input_file2 << " # Input data file 2" << std::endl
		  << "win1=" << win1 << " # \",\"-separated list of coordinates of window1 (x0, y0, x1, y1). win1 and win2 must have the same size" << std::endl
		  << "win2=" << win2 << " # \",\"-separated list of coordinates of window2 (x0, y0, x1, y1). win1 and win2 must have the same size" << std::endl
		  << "output=" << output_directory << " # Working directory" << std::endl
		;
	}
	s << "metrics=" << selected_metrics << " # \",\"-separated list of metrics to use for the template matching, possible values are ms,mrsd,ncc,cch,nmi,msh,mih,emd,gd.";
	return s.str();
}


} // namespace vole

