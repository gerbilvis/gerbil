#ifndef COMMAND_H
#define COMMAND_H

#include <vector>
#include <iostream>
// Workaround for Qt 4 moc bug: moc fails to parse boost headers.
// Fixed in Qt 5.
// https://bugreports.qt-project.org/browse/QTBUG-22829
#ifndef Q_MOC_RUN
#include <boost/program_options.hpp>
#endif // Q_MOC_RUN
#include "vole_config.h"
#include "progress_observer.h"

class CommandRunner;

namespace shell {

class Command {

public:
	Command(const std::string &name, Config& config,
	        const std::string &contributor_name = "",
			const std::string &contributor_mail = "",
			ProgressObserver *po = NULL)
	 : name(name),
	   contributor_name(contributor_name),
	   contributor_mail(contributor_mail),
	   po(NULL),
	   abstract_config(config)
	{}

	virtual ~Command() {}
	virtual const std::string& getName() const { return name; }
	virtual const std::string& getContributorName() const { return contributor_name; }
	virtual const std::string& getContributorMail() const { return contributor_mail; }
	virtual Config& getConfig() { return abstract_config; }
	virtual int execute() = 0;
	virtual std::map<std::string, boost::any> execute(std::map<std::string, boost::any> &input, ProgressObserver *progress = NULL) { /*dummy*/ assert(false); std::map<std::string, boost::any> a; return a; }
	virtual void printShortHelp() const = 0;
	virtual void printHelp() const = 0;

protected:
	ProgressObserver *progressObserver() { return po; }
	ProgressObserver const* progressObserver() const { return po; }
	void setProgressObserver(ProgressObserver *po) { this->po = po; }

	/** Return true if a ProgressObserver is set and it has the abort flag set.
	 */
	bool isAborted() const {
		return progressObserver() && progressObserver()->isAborted();
	}

	std::string name;
	std::string contributor_name;
	std::string contributor_mail;

private:
	ProgressObserver *po;
	Config& abstract_config;

	// setProgressObserver()
	friend class ::CommandRunner;
};

}

#endif // COMMAND_H
