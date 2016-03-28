#include "commandrunner.h"

#include <app/gerbilapplication.h>

#define GGDBG_MODULE
#include "gerbil_gui_debug.h"

CommandRunner::CommandRunner()
	: cmd(0), progress(0.f), percent(0)
{
	connect(this, SIGNAL(exception(std::exception_ptr, bool)),
	        GerbilApplication::instance(),
	        SLOT(handle_exception(std::exception_ptr, bool)),
	        Qt::BlockingQueuedConnection);
}

CommandRunner::~CommandRunner()
{
	GGDBGM("CommandRunner object " << this << endl);
	if (cmd) {
		delete cmd;
	}
}

bool CommandRunner::update(float report, bool incremental)
{
	//GGDBGM("CommandRunner object " << this << endl);
	progress = (incremental ? progress + report : report);

	// propagate if percentage changed, don't heat up GUI unnecessarily
	if ((progress * 100) > percent) {
		percent = progress * 100;
		emit progressChanged(percent);
	}
	return !isAborted();
}

void CommandRunner::run() {
	assert(cmd != NULL);

	emit progressChanged(0);
	try {
		output = cmd->execute(input, this);
	} catch (std::exception &) {
		emit exception(std::current_exception(), false);
		abort();
	}

	if (isAborted())
		emit failure();
	else
		emit progressChanged(100);
		emit success(output);
	//GGDBGM("CommandRunner object " << this << " return" << endl);
}

void CommandRunner::setCommand(shell::Command *cmd)
{
	cmd->setProgressObserver(this);
	this->cmd = cmd;
}

void CommandRunner::abort() {
	// QThread::terminate() is dangerous and doesn't clean up anything
	// this is the safe way to abort.
	ProgressObserver::abort();
	GGDBGM("CommandRunner aborting" << endl);
}

void CommandRunner::deleteLater()
{
	GGDBGM("CommandRunner object " << this << endl);
	QThread::deleteLater();
}
