#include "create_new_command.h"
#include "filesystem_helpers.h"

namespace vole {

CreateNewCommand::CreateNewCommand()
	: Command("vole_new_command",
		config,
		"Christian Riess",
		"christian.riess@informatik.uni-erlangen.de")
{
}

int CreateNewCommand::execute()
{
	if (config.interactive) return manageUserInteraction();
	return doCreation();
}

int CreateNewCommand::doCreation() {
	// do everything offline
	if (!config.add_command && !config.add_gui) {
		if (FilesystemHelpers::file_exists(config.root_directory)) {
			std::cout << "good, root dir exists" << std::endl;
		} else {
			std::cout << "oops, root dir does not exist" << std::endl;
		}
	
		// verify non-existence of the module
		// print svn add string
	} else {
		// verify existence of the module
		if (config.add_command) {
			// verify non-existence of the command class & hook
		} else {
			// verify non-existence of the gui class & hook
		}
		// print svn add string
	}
	return 0;
}

int CreateNewCommand::manageUserInteraction() {
	return 0;
}

void CreateNewCommand::printShortHelp() const
{
	std::cout << "creates a new vole command" << std::endl;
}

void CreateNewCommand::printHelp() const
{
	std::cout << "creates a new vole command." << std::endl;
	std::cout << "Basically, the template in vole_core/meta/template is copied" << std::endl;
	std::cout << "with the classnames etc. changed to the settings for the new command." << std::endl;
}

}
