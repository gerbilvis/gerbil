/*	
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifdef WITH_BOOST_THREAD
#include "background_task.h"

void BackgroundTask::update(int percent)
{
#ifdef WITH_QT
	emit progress(description, percent);
#endif
}

void BackgroundTask::done(bool success)
{
	Lock lock(guard);
	terminated = true;
	this->success = success;
	lock.unlock(); // Unlock to prevent deadlock when signalling the condition.
	future.notify_all();
#ifdef WITH_QT
	emit progress(description, 100);
	emit finished(success);
#endif
}

bool BackgroundTask::wait()
{
	Lock lock(guard);
	if (!terminated) {
		future.wait(lock); // Yields lock until signalled.
	}
	return success;
}

cv::Rect BackgroundTask::roi()
{
	return targetRoi;
}

#endif
