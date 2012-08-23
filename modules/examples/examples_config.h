#ifndef VOLE_EXAMPLES_CONFIG_H
#define VOLE_EXAMPLES_CONFIG_H

#include <iostream>
#include <string>

#include "vole_config.h"

namespace vole {
	namespace examples {

	class ExamplesConfig : public Config {
		public:
		ExamplesConfig();

		// graphical output on runtime?
		bool isGraphical;
		// input data
		std::string input_file;
		// working directory
		std::string output_directory;
		// subcommand (if required for a command)
		int verbosity;

		virtual std::string getString() const;

		#ifdef VOLE_GUI
			virtual QWidget *getConfigWidget();
			virtual void updateValuesFromWidget();
		#endif// VOLE_GUI

		protected:

		#ifdef WITH_BOOST
			virtual void initBoostOptions();
		#endif // WITH_BOOST

		#ifdef VOLE_GUI
		// qt data structures 
		#endif // VOLE_GUI
	};

	}
}

#endif // VOLE_EXAMPLES_CONFIG_H
