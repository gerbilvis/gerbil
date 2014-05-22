#ifndef SOM_TEST_H
#define SOM_TEST_H

#include "som_config.h"
#include <imginput_config.h>
#include <command.h>


class SOMTestConfig : public vole::Config
{
public:
	SOMTestConfig(const std::string& p = "");
	virtual std::string getString() const;

#ifdef WITH_BOOST
	virtual void initBoostOptions();
#endif // WITH_BOOST

	vole::ImgInputConfig imgInput;
	SOMConfig som;

	std::string output_file;
};

class SOMTest : public vole::Command {
public:
	SOMTest();
	~SOMTest();
	int execute();
	
	void printShortHelp() const;
	void printHelp() const;

	SOMTestConfig config;
};

#endif
