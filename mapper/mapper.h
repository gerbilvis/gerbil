#ifndef MAPPER_H
#define MAPPER_H

#include "mapper_config.h"
#include <command.h>

namespace mapper {

class Mapper : public shell::Command {
public:
	Mapper() : Command("mapper", config) {}

	int execute();

	void printShortHelp() const;
	void printHelp() const;

public:
	MapperConfig config;
};

}

#endif // MAPPER_H
