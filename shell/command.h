#ifndef COMMAND_H
#define COMMAND_H

#include <vector>
#include <iostream>
#include <boost/program_options.hpp>
#include "vole_config.h"
#include "progress_observer.h"

namespace vole {


class Command {

public:
	Command(const std::string &name, Config& config,
	        const std::string &contributor_name = "",
	        const std::string &contributor_mail = "")
	 : name(name), contributor_name(contributor_name),
	   contributor_mail(contributor_mail), abstract_config(config)  {}

	virtual ~Command() {}
	virtual const std::string& getName() const { return name; }
	virtual const std::string& getContributorName() const { return contributor_name; }
	virtual const std::string& getContributorMail() const { return contributor_mail; }
	virtual Config& getConfig() { return abstract_config; }
	virtual int execute() = 0;
	virtual void abort() {}
	virtual std::map<std::string, boost::any> execute(std::map<std::string, boost::any> &input, ProgressObserver *progress = NULL) { /*dummy*/ assert(false); std::map<std::string, boost::any> a; return a; }
	virtual void printShortHelp() const = 0;
	virtual void printHelp() const = 0;

protected:
	std::string name;
	std::string contributor_name;
	std::string contributor_mail;

private:
	Config& abstract_config;

};

}

#endif // COMMAND_H
