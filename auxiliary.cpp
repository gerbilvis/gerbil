#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <ctime>
#include "auxiliary.h" // for fun

using namespace std;
using namespace boost::program_options;

ENUM_MAGIC(ms_points)

param::param() : global("General options") {
	global.add_options()
		("help,H", "Give help (on command)")
		("input,I", value(&inputfile)->default_value("input.png"),
		 "Image to process")
		("output,O", value(&outputdir)->default_value("/tmp/"),
		 "Working directory")
		("batch,B", bool_switch(&batch)->default_value(false),
		 "batch mode: output end product only") // TODO: default to false is ugly
	;
	all.add(global);
}

param_mfams::param_mfams() : mfams("Mean shift options") {
	mfams.add_options()
		("lsh.enabled", bool_switch(&use_LSH)->default_value(false),
		 "use locality-sensitive hashing") // TODO: default to false is ugly
		("lsh.K", value(&K)->default_value(30),
		 "K for LSH")
		("lsh.L", value(&L)->default_value(30),
		 "L for LSH")
		("pilot.k", value(&k)->default_value(200),
		 "number of neighbors used in the construction of the pilot density")
		("init.method", value(&starting)->default_value(ALL),
		 "start mean shift from all points (ALL), every Xth point (JUMP), "
		 "or a random selection of points (PERCENT)")
		("init.jump", value(&jump)->default_value(2),
		 "use points with indices 1+(jump*[1..infty])")
		("init.percent", value(&percent)->default_value(50.f),
		 "randomly select given percentage of points")
		("bandwidth", value(&bandwidth)->default_value(1.f),
		 "use fixed bandwidth*dimensionalty for mean shift window")
		("findKL.enabled", value(&findKL)->default_value(false),
		 "empirically determine optimal K, L values (1 < L < lsh.L)")
		("findKL.Kmin", value(&Kmin)->default_value(1),
		 "minimum value of K to be tested (parameter LSH.K for max)")
		("findKL.Kjump", value(&Kjump)->default_value(5),
		 "test every Kmin:Kjump:K values for K")
		("findKL.epsilon", value(&epsilon)->default_value(0.05f),
		 "allowed error introduced by LSH")
	;
	all.add(mfams);
}

bool param_mfams::parse(int argc, char** argv) {

  
	// do the needlework
	try {
		/* command line only options */
		options_description cmdline;
		//string command
		string configfile;
		cmdline.add_options()
		//("command", value(&command)->default_value(""), "command to execute")
		("configfile", value(&configfile)->default_value(""), "configuration file")
		;
		positional_options_description p;
		//p.add("command", 1);
		p.add("configfile", 1);

		/* wrap it up to start parsing */
		variables_map vm;
		cmdline.add(all); // all is member variable

		/* The first store is the preferred one!
		   Therefore the command line arguments overrule the ones given in the config file */
		store(command_line_parser(argc, argv).options(cmdline).positional(p).run(), vm);
		notify(vm);

		/* Add input from config file if applicable */
		if (!configfile.empty()) {
		        ifstream file(configfile.c_str(), ios_base::in);
		        if (!file.good()) {
		                cout << "*** Error: File " << configfile
		                          << " could not be read!" << endl;
		                return NULL;
		        }
		        store(parse_config_file(file, all), vm);
		        notify(vm);
		        file.close();
		}

		/* print help if we got nuthin' better to do */
        if (vm.count("help")) {
	        cout << "Usage: " << argv[0] << " [configfile] [options ...]" << endl << endl;
	        cout << "All options can also be given in the specified config file." << endl;
	        cout << "Options given in the command line will"
	        		"overwrite options from the file." << endl << endl;
			cout << all;
			cout << endl;
			return false;
		}
        return true;
    }
    catch(exception& e) {
        cerr << "*** Error reading configuration:\n    " << e.what() << endl << endl;
        cout << "Run " << argv[0] << " --help for help!" << endl;
        return false;
    }


}

void bgLog(const char *varStr, ...) {
	//obtain argument list using ANSI standard...
	va_list argList;
	va_start(argList, varStr);

	//print the output string to stderr using
	vfprintf(stdout, varStr, argList);
	va_end(argList);
}

time_t timestart, timeend;

void timer_start() {
	timestart = clock();
}

double timer_stop() {
	timeend = clock();
	unsigned long seconds, milliseconds;
	seconds      = (timeend - timestart) / CLOCKS_PER_SEC;
	milliseconds =
		((1000 * (timeend - timestart)) / CLOCKS_PER_SEC) - 1000 * seconds;
	return seconds + milliseconds / 1000.0;
}

double timer_elapsed(int prnt) {
	timeend = clock();
	unsigned long hours = 0, minutes = 0, seconds = 0, milliseconds = 0;
	seconds      = (timeend - timestart) / CLOCKS_PER_SEC;
	milliseconds =
		((1000 * (timeend - timestart)) / CLOCKS_PER_SEC) - 1000 * seconds;
	minutes = seconds / 60;
	if (minutes == 0) {
		if (prnt) {
			printf("elapsed %lu.%03lu seconds. \n", seconds, milliseconds);
		}
	} else {
		hours   = minutes / 60;
		seconds = seconds - minutes * 60;
		if (hours == 0) {
			if (prnt)
				printf("elapsed %lum%lus%lums\n", minutes,
					   seconds,
					   milliseconds);
		} else {
			minutes = minutes - hours * 60;
			if (prnt)
				printf("elapsed %luh%lum%lus%lums\n", hours,
					   minutes, seconds,
					   milliseconds);
		}
	}
	return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0;
}
