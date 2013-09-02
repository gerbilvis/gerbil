/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>
	and Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "vole_config.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

#ifdef WITH_BOOST_PROGRAM_OPTIONS
using namespace boost::program_options;
#endif // WITH_BOOST_PROGRAM_OPTIONS

namespace vole {

Config::Config(const std::string& p) :
	prefix_enabled(!p.empty()),
	prefix(prefix_enabled ? p + "." : p),
	verbosity(0)
{
#ifdef WITH_BOOST_PROGRAM_OPTIONS
	initMandatoryBoostOptions();
#endif // WITH_BOOST_PROGRAM_OPTIONS
}

bool Config::readConfig(const char *filename) {
#ifdef WITH_BOOST_PROGRAM_OPTIONS
	return parseOptionsDescription(filename);
#endif // WITH_BOOST_PROGRAM_OPTIONS

	std::cerr << "No options reader implemented - big mayhem in core/vole_config.cpp" << std::endl;
	return false; 
}

const char* Config::key(const char *k) const {
	if (!prefix_enabled)
		return k;
	/* note: the boost parser enables shortcut keys (single character) by adding
	   ",<keychar>" to the key description. It does not support these with
	   a prefix. So they are stripped here. */
	std::string full = prefix, key = k;
	full += key.substr(0, key.find_first_of(','));
	return strdup(full.c_str());
	/* note this is a waste of memory, but only a small one and happens only
	   once. we have no other good option with add_options() only taking char* */
}

bool Config::storeConfig(const char *filename) {
	std::ofstream out;
	out.open(filename);
	// TODO ordentliche Fehlerabfrage
	if (!out.good())
		return false;
	out << getString() << std::endl;
	out.close();
	return true;
}

unsigned long int Config::configHash(vole::HashMethod method) {
	// collect string from relevant parameters, hash it
	std::string current_config = getString();
	return Hashes::getHash(current_config.c_str(), method);
}

#ifdef WITH_BOOST_PROGRAM_OPTIONS
void Config::initMandatoryBoostOptions() {
	options.add_options()
		(key("verbose,V"), value(&verbosity)->default_value(1),
		 "Verbosity Level: 0 = silent, 1 = normal, 2 = much output, 3 = insane")
	;
}

bool Config::parseOptionsDescription(const char *filename, variables_map *vm)
{
	std::ifstream file(filename, std::ios_base::in);
	if (!file.good()) {
		std::cerr << "*** Error: File " << filename
		          << " could not be read!" << std::endl;
		return false;
	}

	try {
		bool vm_created = false;
		if (vm == NULL) {
			vm = new variables_map();
			vm_created = true;
		}
		store(parse_config_file(file, options), *vm);
		file.close();
		notify(*vm);
		if (vm_created)
			delete vm;
		return true;
	} catch (std::exception& e) {
		std::cerr << "*** Error reading configuration:\n    " << e.what();
		std::cerr << std::endl << std::endl;
		return false;
	}

}
#endif // WITH_BOOST_PROGRAM_OPTIONS


}
