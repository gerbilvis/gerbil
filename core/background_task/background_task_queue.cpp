/*	
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifdef WITH_BOOST_THREAD

//#define BACKGROUND_TASK_QUEUE_DEBUG

#include "background_task_queue.h"
#include <iostream>
#include <iomanip>

bool BackgroundTaskQueue::isIdle()
{
	Lock lock(mutex);
	return (taskQueue.empty()) && (!currentTask);
}

void BackgroundTaskQueue::halt()
{
	Lock lock(mutex);
	halted = true;
	taskQueue.clear(); // Flush the queue, so there is nothing else to pop.
	if (currentTask) {
		cancelled = true;
		currentTask->cancel();
	}
	lock.unlock(); // Unlock to prevent deadlock when signalling the condition.
	future.notify_all(); // In case the thread is sleeping.
}

bool BackgroundTaskQueue::pop() 
{
	Lock lock(mutex);
	while (taskQueue.empty()) {
		if (halted) {
			return false; // Thread will terminate.
		} else {
			future.wait(lock); // Yields lock until signalled.
		}
	}
	currentTask = taskQueue.front(); // Fetch the task.
	taskQueue.pop_front();
#ifdef BACKGROUND_TASK_QUEUE_DEBUG
	std::cout << "BackgroundTaskQueue pop():" << std::endl;
	print();
#endif /* BACKGROUND_TASK_QUEUE_DEBUG */
	cancelled = false;
	return true;
}

void BackgroundTaskQueue::print()
{
	int row = 0;
	for(std::deque<BackgroundTaskPtr>::const_iterator it = taskQueue.begin();
		it != taskQueue.end();
		++it, ++row)
	{
		std::string name = typeid(**it).name();
		std::cout << std::setw(4) << row << " " << name << std::endl;
	}
}

void BackgroundTaskQueue::push(BackgroundTaskPtr &task) 
{
	Lock lock(mutex);
	taskQueue.push_back(task);
#ifdef BACKGROUND_TASK_QUEUE_DEBUG
	std::cout << "BackgroundTaskQueue push():" << std::endl;
	print();
#endif /* BACKGROUND_TASK_QUEUE_DEBUG */
	lock.unlock(); // Unlock to prevent deadlock when signalling the condition.
	future.notify_all();
}

void BackgroundTaskQueue::cancelTasks()
{
	Lock lock(mutex);
	taskQueue.clear();
	if (currentTask) {
		cancelled = true;
		currentTask->cancel();
	}
}

void BackgroundTaskQueue::operator()() 
{
#ifdef WITH_QT
	try {
#else
	{
#endif
		while (true) {
			if (!pop()) {
				break; // Thread termination.
			}
			bool success = currentTask->run();
			{
				Lock lock(mutex);
				currentTask->done(!cancelled && success);
				currentTask.reset();
			}
		}
#ifdef WITH_QT
	} catch (std::exception &) {
		emit exception(std::current_exception(), true);
#endif
	}
}

#endif
