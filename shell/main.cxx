#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include "modules.h"
#include "command.h"

using namespace std;
using namespace boost::program_options;

/* NOTE: use #define SINGLE to build the single-command version! */

void printVoleOnce() {
	static bool already_printed = false;
	if (already_printed)
		return;
	already_printed = true;

	cout << "\n"
			"            .s_,  ._ssaoXXXZqaa,.              \n"
			"           _mmZmaoZXSnooooXXZmWWma.            \n"
			"         _mQWmZZoo2ooonnnnnXZ#mWWBma.          \n"
			"        <QWQB#ZXnnoo2onvnnnXX#mBmWWWm,.        \n"
			"       =WWmW#ZenvnoonI|+|+{nX##WBWBWBWWga,     \n"
			"       ???Y?\"---<vIl|i=====I3X#mWmBm###:?Wc    \n"
			"             )nnii\"----   ---*YX##!~-   .mk    \n"
			"              -           :iiv?!~-   . .j#~    \n"
			"                                     _saZ!`     \n"
#ifdef __unix__
			"  \x1b[1mG E R B I L\x1b[0m        -{I;_asssas%IY!!^         \n"
#else
			"  G E R B I L        -{I;_asssas%IY!!^         \n"
#endif
			"\n" << endl;
}

void printGeneralHelp(const char *program_name, bool withLogo, bool single) {
	if (withLogo)
		printVoleOnce();
	cout << "Usage: " << program_name << " [--help]" << (single ? " " : " command ");
	cout << "[configfile] [options ...]" << endl << endl;
	cout << "All options can also be given in the specified config file." << endl;
	cout << "Options given in the command line will overwrite options from the file.";
	cout << endl << endl;
}

void printAvailableCommands(const vole::Modules &m) {
	cout << "Available commands:" << endl;
	for (vole::Modules::const_iterator it = m.begin(); it != m.end(); it++) {
		cout << "  " << it->first << ":\t";
		it->second->printShortHelp();
	}
}

void printSpecificHelp(vole::Command *cmd, bool single) {
	cout << "-------------------------------------------------------------------------------" << endl;
	cmd->printHelp();
	cout << "-------------------------------------------------------------------------------" << endl;
	cout << endl;
	if (!single)
		cout << "Options for " << cmd->getName() << ":" << endl;
	cout << cmd->getConfig().options;
}

vole::Command* grab_command(int argc, char** argv, const vole::Modules &m) {
	bool firsthelp = false;
	if (argc > 1)
		firsthelp = ((strncmp(argv[1], "--help", 6) == 0) || (strncmp(argv[1], "-H", 2) == 0));
	// catch --help and no command at all
	if ((argc == 1) || ((argc == 2) && firsthelp)) {
		printGeneralHelp(argv[0], true, false);
		printAvailableCommands(m);
		return NULL;
	}
	string command = (firsthelp ? argv[2] : argv[1]);

	// fetch command object
	vole::Command *cmd = NULL;
	vole::Modules::const_iterator it = m.find(command);
	if (it != m.end())
		cmd = it->second;

	if (!cmd) {
		cerr << "*** Error: Command " << command << " is unknown to me!" << endl << endl;
		printAvailableCommands(m);
	}
	return cmd;
}

bool parse_opts(int argc, char** argv, vole::Command *cmd, bool single) {
	string configfile, placeholder;
	vole::Config& config = cmd->getConfig();

	try {
		variables_map vm;

		// get options from command
		options_description cmdline;
		positional_options_description p;
		cmdline.add(config.options);
		cmdline.add_options()
			("help,H", "Give help (on command)")
			("configfile", value(&configfile)->default_value(""), "configuration file")
		;
		if (!single) { // needed such that given command does not confuse parser 
			cmdline.add_options()
				("command", value(&placeholder)->default_value(""), "command to execute");
			p.add("command", 1);
		}
		p.add("configfile", 1);

		// parse commandline
		/* The first store is the preferred one!
		   Therefore the command line arguments overrule the ones given in the config file */
		store(command_line_parser(argc, argv).options(cmdline).positional(p).run(), vm);
		notify(vm);

		if (vm.count("help")) {
//			printGeneralHelp(argv[0], config.verbosity > 0, single);
			printSpecificHelp(cmd, single);
			return false;
		}

		/* Add input from config file if applicable */
		if (!configfile.empty()) {
			// always keep same variables map (vm), otherwise earlier changes are shadowed
			if (!config.parseOptionsDescription(configfile.c_str(), &vm)) {
				cerr << "*** Error reading config file: " << configfile << endl;
				return false;
			}
		}
		return true;
	} catch (exception& e) {
		cerr << "*** Error reading configuration:\n    " << e.what() << endl << endl;
		cout << "Run " << argv[0] << " --help for help!" << endl;
		return false;
	}
}

int main(int argc, char *argv[]) {
#ifdef SINGLE
	bool single = true;
#else
	bool single = false;
#endif
	vole::Modules m;
	assert(!m.empty());
	vole::Command *c = (single ? m.begin()->second : grab_command(argc, argv, m));
	if (!c || !parse_opts(argc, argv, c, single)) return 1;
	if (c->getConfig().verbosity > 0)
		printVoleOnce();
	return c->execute();	// all command destructors are called here
}

