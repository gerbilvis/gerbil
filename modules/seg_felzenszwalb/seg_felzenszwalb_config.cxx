#include "seg_felzenszwalb_config.h"

#include <iostream>
#include <fstream>
#include <ctime> 
#include <cstdlib>
#include <sstream>

#ifdef WITH_QT
#include <QVBoxLayout>
#include <QLabel>
#include <QCheckBox>
#endif // WITH_QT

using namespace boost::program_options;

namespace vole {

SegFelzenszwalbConfig::SegFelzenszwalbConfig(const std::string& prefix)
 : Config(prefix) {
	initBoostOptions();
	configWidget = NULL;
}

SegFelzenszwalbConfig::~SegFelzenszwalbConfig() {}


std::string SegFelzenszwalbConfig::getString() const {
	std::stringstream s;
	
	if (prefix_enabled) {
		s << "[" << prefix << "]" << std::endl;
	} else { // no prefix, i.e. we handle I/O parameters
		s << "input=" << input_file << " # Image to process" << std::endl
		  << "output=" << output_directory << " # Working directory" << std::endl;
	}

	s << "sigma=" << sigma << " # std. dev. for Gaussian pre-smoothing" << std::endl
	<< "k=" << k_threshold
		<< " # threshold for the pixel similarity (lower k = smaller superpixel)" << std::endl
	<< "min_size=" << min_size << " # minimum number of pixels per superpixel" << std::endl
	<< "chroma_img=" << (chroma_img ? "true" : "false")
		<< " # segment on chroma image (instead of intensity image)" << std::endl
	;
	return s.str();
}

#ifdef WITH_QT
void SegFelzenszwalbConfig::initConfigWidget() {
	if (configWidget != NULL) return;
	configWidget = new QWidget();
	// append everything that could possibly be configured
	layout = new QHBoxLayout();
	QVBoxLayout *globalValues = new QVBoxLayout();
	QLabel *verbosity = new QLabel("verbosity");
	globalValues->addWidget(verbosity);
	globalValues->addStretch();
	updateComputation = new QPushButton("Update!");
	globalValues->addWidget(updateComputation);
	layout->addLayout(globalValues);
	// TODO: in subclasses, you have to set the layout:
	// configWidget->setLayout(layout);
}

QWidget *SegFelzenszwalbConfig::getConfigWidget() {
	this->initConfigWidget();
	QVBoxLayout *fhs_config = new QVBoxLayout();
	fhs_config->addWidget(new QLabel("sigma"));
	edit_sigma = new QLineEdit();
	{
		edit_sigma->setInputMask("9.9000; ");
		std::stringstream s; s.precision(5); s << sigma;
		edit_sigma->setText(QString(s.str().c_str()));
	}
	fhs_config->addWidget(edit_sigma);

	fhs_config->addWidget(new QLabel("similarity thresh. k"));
	edit_k_threshold = new QLineEdit();
	{
		edit_k_threshold->setInputMask("0099; ");
		std::stringstream s; s << k_threshold;
		edit_k_threshold->setText(QString(s.str().c_str()));
	}
	fhs_config->addWidget(edit_k_threshold);

	fhs_config->addWidget(new QLabel("min. #pxls/seg."));
	edit_min_size = new QLineEdit();
	{
		edit_min_size->setInputMask("00099; ");
		std::stringstream s; s << min_size;
		edit_min_size->setText(QString(s.str().c_str()));
	}
	fhs_config->addWidget(edit_min_size);
	chk_chroma_img = new QCheckBox("use norm.RGB?");
	chk_chroma_img->setCheckState((chroma_img ? Qt::Checked : Qt::Unchecked));
	fhs_config->addWidget(chk_chroma_img);
	layout->addLayout(fhs_config);
	configWidget->setLayout(layout);
	return configWidget;
}

void SegFelzenszwalbConfig::updateValuesFromWidget() {
	{ std::stringstream s; s << edit_sigma->text().toStdString();
		s >> sigma; }
	{ std::stringstream s; s << edit_k_threshold->text().toStdString();
		s >> k_threshold; }
	{ std::stringstream s; s << edit_min_size->text().toStdString();
		s >> min_size; }
	chroma_img = (chk_chroma_img->checkState() == Qt::Checked);
}
#endif // WITH_QT

#ifdef WITH_BOOST
void SegFelzenszwalbConfig::initBoostOptions() {
	options.add_options()
		// TODO da gibt es wohl auch Werte, die im Code nie benutzt werden; die
		// sollten entfernt werden.
		(key("graphical"), bool_switch(&isGraphical)->default_value(false),
			 "Show any graphical output during runtime")
		(key("sigma"), value(&sigma)->default_value(0.5),
		                   "std. dev. for Gaussian pre-smoothing")
		(key("k"), value(&k_threshold)->default_value(500),
		                   "threshold for the pixel similarity (lower k = smaller superpixel)")
		(key("min_size"), value(&min_size)->default_value(20),
		                   "minimum number of pixels per superpixel")
		(key("chroma_img"), value(&chroma_img)->default_value(false),
		                   "segment on chroma image (instead of intensity image)")
		;
	
	if (prefix_enabled)
		return;
		
	options.add_options()	// input/output options
		(key("input,I"), value(&input_file)->default_value("input.png"),
		 "Image to process")
		(key("output,O"), value(&output_directory)->default_value("/tmp/"),
		 "Working directory")
		;
}
#endif // WITH_BOOST

}


