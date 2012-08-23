#include "commandrunner.h"

CommandRunner::CommandRunner()
	: abort(false), cmd(NULL) {}

CommandRunner::~CommandRunner()
{
	delete cmd;
}

bool CommandRunner::update(int percent)
{
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
}

void CommandRunner::terminate() {
	// QThread::terminate() is dangerous and doesn't clean up anything
	// this is the safe way to abort
	abort = true;
}
