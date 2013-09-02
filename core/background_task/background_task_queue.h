/*	
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef BACKGROUND_TASK_QUEUE_H
#define BACKGROUND_TASK_QUEUE_H

#ifdef WITH_BOOST_THREAD
#include <deque>
#include <boost/thread/mutex.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/condition_variable.hpp>
#include <opencv2/core/core.hpp>
#include "background_task.h"

class BackgroundTaskQueue {

public:
	BackgroundTaskQueue() : halted(false), cancelled(false) {}

	/** Any tasks in the queue? */
	bool isIdle();

	/** Flush all queued tasks and terminate worker thread. */
	void halt();
	/** Put task into queue for later calculation. */
	void push(BackgroundTaskPtr &task);
	/** Cancel all tasks associated with the given ROI. */
	// TODO FIXME This is a major PITA,
	// since everybody who wants to cancel tasks needs to know the ROI.
	void cancelTasks(const cv::Rect &roi = cv::Rect());

	/** Background worker thread's main(). */
	void operator()(); 

protected:
	/** Fetch task from queue or passivelly wait on empty queue. */
	bool pop();

private:
	// do not implement
	BackgroundTaskQueue(const BackgroundTaskQueue &other);
	// do not implement
	BackgroundTaskQueue &operator=(const BackgroundTaskQueue &other);

	typedef boost::mutex Mutex;
	typedef boost::unique_lock<Mutex> Lock;

	/** Used for background thread termination. */
	bool halted; 
	/** Discards results of the currently calculated task. */
	bool cancelled; 
	/** Wakes sleeping worker thread. */
	boost::condition_variable future; 
	/** Serializes thread access to the queue. */
	Mutex mutex;
	/** Currently calculated task. */
	BackgroundTaskPtr currentTask; 
	/** Task queue which is consumed one-by-one by worker thread. */
	std::deque<BackgroundTaskPtr> taskQueue;
};

#endif
#endif
