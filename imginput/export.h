#ifndef EXPORT_H
#define EXPORT_H

#include "imginput_config.h"
#include <command.h>

namespace imginput {

class Export : public shell::Command {
public:
	Export() : Command("export", config) {}

	int execute();

	void printShortHelp() const;
	void printHelp() const;

public:
	ImgInputConfig config;
};

}

#endif // EXPORT_H
