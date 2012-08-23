#include "create_new_command_config.h"

namespace vole {

CreateNewCommandConfig::CreateNewCommandConfig(const std::string &prefix) : Config(prefix) {
	#ifdef WITH_BOOST
		initBoostOptions();
	#endif // WITH_BOOST
}

std::string CreateNewCommandConfig::getString() {
	std::stringstream s;

	s << "[newcommand]" << std::endl
		<< "verbose=" << verbosity
			<< " # Verbosity Level: 0 = silent, 1 = normal, 2 = much output, 3 = insane" << std::endl
		<< "graphical=" << isGraphical
			<< " # Show any graphical output during runtime" << std::endl
		<< "input=" << input_file << " # Image to process" << std::endl
		<< "output=" << output_directory << " # Working directory" << std::endl
		<< "interactive=" << interactive << " # create the command interactively" << std::endl
		<< "component=" << component_name << " # component where this command should belong to" << std::endl
		<< "unit=" << unit_name
			<< " # unit where this command should belong to. If left blank, a new unit called 'name' is created"
			<< std::endl
		<< "name=" << module_name << " # name of the module" << std::endl
		<< "root=" << root_directory << " # root directory of the command" << std::endl
		<< "add_gui=" << add_gui << " # add hook for graphical interface to existing module" << std::endl
		<< "add_command=" << add_command << " # add hook for shell command to existing module" << std::endl
	;
	return s.str();
}


#ifdef WITH_BOOST
void CreateNewCommandConfig::initBoostOptions() {
	options.add_options()
		// global section
		// TODO da gibt es wohl auch Werte, die im Code nie benutzt werden; die
		// sollten entfernt werden.
		(key("graphical"), bool_switch(&isGraphical)->default_value(false),
			 "Show any graphical output during runtime")
		(key("input,I"), value(&input_file)->default_value("input.png"),
		 "Image to process")
		(key("output,O"), value(&output_directory)->default_value("/tmp/"),
		 "Working directory")

		(key("interactive,i"), bool_switch(&interactive)->default_value(false),
			"create the command interactively")
		(key("component"), value(&component_name)->default_value(""),
			"component where this command should belong to")
		(key("unit"), value(&unit_name)->default_value(""),
			"unit where this command should belong to. If left blank, a new unit called 'name' is created")
		(key("name"), value(&module_name)->default_value(""),
			"name of the module")
		(key("root"), value(&root_directory)->default_value(""),
			"root directory of the command")
		(key("add_gui"), bool_switch(&add_gui)->default_value(false),
			"add hook for graphical interface to existing module")
		(key("add_command"), bool_switch(&add_command)->default_value(false),
			"add hook for shell command to existing module")
	;
}
#endif // WITH_BOOST



}
