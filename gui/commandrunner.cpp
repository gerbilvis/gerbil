#include "commandrunner.h"

#define GGDBG_MODULE
#include "gerbil_gui_debug.h"

CommandRunner::CommandRunner()
	: cmd(0), abort(false), progress(0.f), percent(0) {}

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
	return !abort;
}

void CommandRunner::run() {
	assert(cmd != NULL);

	emit progressChanged(0);
	output = cmd->execute(input, this);
	emit progressChanged(100);

	if (abort)
		emit failure();
	else
		emit success(output);
	//GGDBGM("CommandRunner object " << this << " return" << endl);
	return;
}

void CommandRunner::terminate() {
	// QThread::terminate() is dangerous and doesn't clean up anything
	// this is the safe way to abort
	abort = true;
	//std::cerr << "CommandRunner aborting" << std::endl;
}

void CommandRunner::deleteLater()
{
	//GGDBGM("CommandRunner object " << this << endl);
	QThread::deleteLater();
}
