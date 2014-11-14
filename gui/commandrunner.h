#ifndef COMMANDRUNNER_H
#define COMMANDRUNNER_H

#include <gerbil_cplusplus.h>

#include <QThread>
#include <vole_config.h>
#include <command.h>
#include <progress_observer.h>

class CommandRunner : public QThread, public ProgressObserver
{
    Q_OBJECT
public:
	// FIXME: Missing parent parameter
	CommandRunner();
	~CommandRunner();
	bool update(float report, bool incremental = false);
	void run();

	shell::Command *cmd;
	std::map<std::string, boost::any> input;
	std::map<std::string, boost::any> output;
	const std::string base;

signals:
	void progressChanged(int percent);
	void success(std::map<std::string, boost::any> output);
	void failure();

public slots:
	void abort() GBL_OVERRIDE;
	void deleteLater();

private:
	// progress cached for incremental updates (not threadsafe)
	float progress;
	// progress in percent
	int percent;
};

#endif // COMMANDRUNNER_H
