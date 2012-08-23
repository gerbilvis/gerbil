#ifndef COMMANDRUNNER_H
#define COMMANDRUNNER_H

#include <QThread>
#include <vole_config.h>
#include <command.h>
#include <progress_observer.h>

class CommandRunner : public QThread, public vole::ProgressObserver
{
    Q_OBJECT
public:
	CommandRunner();
	~CommandRunner();
	bool update(int percent);
	void run();

	bool abort;
	vole::Command *cmd;
	std::map<std::string, boost::any> input;
	std::map<std::string, boost::any> output;
	const std::string base;

signals:
	void progressChanged(int percent);
	void success(std::map<std::string, boost::any> output);
	void failure();

public slots:
	void terminate();
};

#endif // COMMANDRUNNER_H
