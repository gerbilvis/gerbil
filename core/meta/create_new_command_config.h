#ifndef VOLE_CREATE_NEW_COMMAND_CONFIG_H
#define VOLE_CREATE_NEW_COMMAND_CONFIG_H

#include "vole_config.h"

namespace vole {

class CreateNewCommandConfig : public Config {
public:
	CreateNewCommandConfig(const std::string &prefix = std::string());
	virtual std::string getString();

	// global vole stuff
	/// may iebv create graphical output (i.e. open graphical windows)?
	bool isGraphical;
	/// input file name
	std::string input_file;
	/// directory for all intermediate files
	std::string output_directory;

	/// add hook for shell command to existing module
	bool add_command;
	/// add hook for graphical interface to existing module
	bool add_gui;

	/// create the command interactively
	bool interactive;

	/// component where this command should belong to
	std::string component_name;
	/// unit where this command should belong to. If left blank, a new unit is created
	std::string unit_name;
	/// name of the module.
	std::string module_name;
	/// root directory of the command. It is possible to follow the
	//recommendations that are generated from the build tool
	std::string root_directory;


protected:
	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST
	
};

}

#endif // VOLE_CREATE_NEW_COMMAND_CONFIG_H
