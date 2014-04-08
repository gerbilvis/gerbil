#include "commandrunner.h"

//#include "gerbil_gui_debug.h"

CommandRunner::CommandRunner()
	: abort(false), cmd(NULL) {}

CommandRunner::~CommandRunner()
{
	//GGDBGM("CommandRunner object " << this << endl);
	delete cmd;
}

bool CommandRunner::update(int percent)
{
	//GGDBGM("CommandRunner object " << this << endl);
	emit progressChanged(percent);
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
	if(cmd) {
		cmd->abort();
	}
	//std::cerr << "CommandRunner aborting" << std::endl;
}

void CommandRunner::deleteLater()
{
	//GGDBGM("CommandRunner object " << this << endl);
	QThread::deleteLater();
}
