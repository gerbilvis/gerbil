/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "vole_user_paths.h"
#include "filesystem_helpers.h"

#include <sstream>
#include <iostream>
#include <cerrno>

#include <cstdlib> // for getenv

// FIXME
#ifdef __unix__
#include <unistd.h> // for getcwd
#endif

#ifdef WITH_BOOST_PROGRAM_OPTIONS
using namespace boost::program_options;
#endif // WITH_BOOST_PROGRAM_OPTIONS

namespace vole {

UserPaths::UserPaths() : Config()
{
	#ifdef WITH_BOOST_PROGRAM_OPTIONS
		initBoostOptions();
	#endif // WITH_BOOST_PROGRAM_OPTIONS

	// try to read the data from the home directory
	// FIXME this is most likely _not_ portable
	char *home_dir = getenv("HOME");
	std::string backup;
	if (home_dir == NULL) { // happens with WIN32
		backup = "C:/temp/";
		home_dir = const_cast<char *>(backup.c_str());
	}
		
	std::stringstream vole_config_file;
	vole_config_file << home_dir << "/.vole.cfg";

	std::string cfg_file = vole_config_file.str();

	if (FilesystemHelpers::file_exists(cfg_file)) {
		readConfig(cfg_file.c_str());
	} else {
		std::cerr << "vole global config file " << cfg_file << " not found, created standard config file; please check the paths if they suit your needs" << std::endl;
		// FIXME again highly unportable
// FIXME
#ifdef __unix__
		char *working_dir = get_current_dir_name();
		verbosity = 1;
		gvole_config_root = std::string(working_dir);
		gvole_data_root = "/disks/data1/riess/gvole_data";
		if (!storeConfig(cfg_file.c_str()) || (errno != 0)) {
			std::stringstream error_msg_stream;
			error_msg_stream << "unable to create config file " << cfg_file
				<< ", no config written" << std::endl;
			std::string error_msg = error_msg_stream.str();
			std::cerr << error_msg;
			free(working_dir);
			throw error_msg;
		}
		free(working_dir);
		std::exit(1);
// #else
		verbosity = 1;
		std::cout << "working_dir not used, creating .vole.cfg in (home?? ) directory " << home_dir << std::endl;
		gvole_config_root = std::string(home_dir);
		gvole_data_root = "/disks/data1/riess/gvole_data";
		if (!storeConfig(cfg_file.c_str()) || (errno != 0)) {
			std::stringstream error_msg_stream;
			error_msg_stream << "unable to create config file " << cfg_file
				<< ", no config written" << std::endl;
			std::string error_msg = error_msg_stream.str();
			std::cerr << error_msg;
			throw error_msg;
		}
#else
#endif
	}
}

std::string UserPaths::getString() const {
	std::stringstream s;
	s	<< "verbose=" << verbosity
			<< " # verbosity level: 0 = silent, 1 = normal, 2 = much output, 3 = insane" << std::endl
		<< "[vole]" << std::endl
		<< "gvole_config_root=" << gvole_config_root
			<< " # root directory for config files in gvole (typically the installation root)" << std::endl
		<< "gvole_data_root=" << gvole_data_root
			<< " # root directory for data files in gvole" << std::endl
	;
	return s.str();
}

#ifdef WITH_BOOST_PROGRAM_OPTIONS
void UserPaths::initBoostOptions() {
	options.add_options()
		("vole.gvole_config_root", value(&gvole_config_root)->default_value(""),
			 "root directory for config files in gvole (typically the installation root)")
		("vole.gvole_data_root", value(&gvole_data_root)->default_value("/disks/data1/riess/gvole_data"),
			 "root directory for data files in gvole")
	;
}
#endif // WITH_BOOST_PROGRAM_OPTIONS

}
