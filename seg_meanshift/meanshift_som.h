#ifndef MEANSHIFT_SOM_H
#define MEANSHIFT_SOM_H

#include "meanshift_config.h"
#include "meanshift.h"
#include <command.h>

#ifdef WITH_SOM
namespace som {
	class GenSOM;
	class SOMClosestN;
}
#endif

namespace seg_meanshift {

class MeanShiftSOM : public shell::Command {

public:
	struct Result : public MeanShift::Result {
#ifdef WITH_SOM
		boost::shared_ptr<som::GenSOM> som;
		boost::shared_ptr<som::SOMClosestN> lookup;
#endif
	};

	MeanShiftSOM();
	~MeanShiftSOM();
	int execute();
	Result execute(multi_img::ptr input);

	void printShortHelp() const;
	void printHelp() const;

	MeanShiftConfig config;
};

}

#endif
