#ifndef VOLE_QUICKSHIFT_CONFIG_H
#define VOLE_QUICKSHIFT_CONFIG_H

#include "vole_config.h"

#include "cv.h"

#include <iostream>
#include <vector>

#ifdef VOLE_GUI
#include <QWidget>
#include <QLineEdit>
#include <QCheckBox>
#endif // VOLE_GUI

/**
 * Configuration parameters for the illuminant estimation by voting, to be
 * included by all classes that need to access it.
 */
class QuickshiftConfig : public vole::Config {
public:
	QuickshiftConfig(const std::string& prefix = std::string());

	// global vole stuff
	/// input file name
	std::string input_file;
	/// directory for all intermediate files
	std::string output_file;

	double kernel_size;
	double max_dist_multiplier;
	double merge_threshold;

	bool use_chroma;

	#ifdef VOLE_GUI
		virtual QWidget *getConfigWidget();
		virtual void updateValuesFromWidget();
	#endif// VOLE_GUI

	std::string getString() const;

protected:

	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST

	#ifdef VOLE_GUI
		// define gui elements here
	#endif // VOLE_GUI
};


#endif // VOLE_QUICKSHIFT_CONFIG_H
