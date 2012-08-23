#ifndef VOLE_QUICKSHIFT_DEMO_H
#define VOLE_QUICKSHIFT_DEMO_H

#include <iostream>
#include "command.h"

#include "quickshift_config.h"

#include "cv.h"


class quickshiftDemo : public vole::Command {
public:
	quickshiftDemo();
	~quickshiftDemo();
	int execute();

	void printShortHelp() const;
	void printHelp() const;

	QuickshiftConfig config;

private:

};

#endif // VOLE_QUICKSHIFT_DEMO_H
