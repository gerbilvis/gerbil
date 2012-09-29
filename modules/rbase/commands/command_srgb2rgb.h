#ifndef RBASE_COMMAND_SRGB_TO_RGB_H
#define RBASE_COMMAND_SRGB_TO_RGB_H

#include "command.h"
#include "srgb2rgb_config.h"

namespace rbase {

class CommandSrgb2Rgb : public vole::Command
{
public:
	CommandSrgb2Rgb();
	int execute();

	void printShortHelp() const;
	void printHelp() const;

	Srgb2RgbConfig config;

private:


};


}

#endif // RBASE_COMMAND_SRGB_TO_RGB_H
