/*	
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef SHARED_DATA_H
#define SHARED_DATA_H

#include <multi_img.h>
#include <utility>
#include <boost/shared_ptr.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/thread/locks.hpp>
#include <opencv2/core/core.hpp>

#ifdef WITH_QT
#include <QImage>
#endif

/** Optimally, there should be boost::shared_mutex to allow more than one
    thread to hold data concurrently. Because shared_mutex is not recursive
	(not even for read access), it would however need fundamental refactoring
	of currently existing classes to avoid deadlocks. Since it is not expected
	there to be more than two foreground threads holding data (e.g. one for GUI
	event loop and second for OpenGL rendering), usage of recursive_mutex seems
	to be reasonable tradeoff between low footprint on existing code and
	concurrency guarantees. */
typedef boost::recursive_mutex SharedDataGuard;
/** Lock that should be used by foreground threads (GUI, OpenGL) to prevent
    background worker thread to swap embedded raw pointer. Note that the lock 
	does not prevent other parties to modify the data in-place, that is if you 
	need mutual exclusion on the targeted data itself, you should equip the 
	targeted data by a mutex of your choice and properly lock on it depending 
	on a particular scenario. Often, the possible in-place modification of
	data by foreground threads can be avoided by temporarily disabling user
	input in the corresponding part of the GUI. */
typedef boost::unique_lock<SharedDataGuard> SharedDataHold;
/** Lock that should be used by background worker thread to swap embedded raw
    pointer once the calculation of new version of data is finished. Note that
	this should enforce usage of a simple variant of read-copy-update pattern.
	Simple in a sense that there can be only one swapper thread, which is
	expected to be background worker thread dispatching background tasks
	from the queue. Also note that it is perfectly fine if background worker
	modify locked data in-place instead of using RCU pattern - this might be 
	reasonable if it involves assignment of only a couple of primitive values. */
typedef boost::unique_lock<SharedDataGuard> SharedDataSwap;

/** Templated wrapper class for any data shared between foreground presentation
    threads (GUI, OpenGL) and background worker thread. All data members of 
	BackgroundTask that are expected to be accessed by presentation thread while
	worker thread is calculating new version of that data shall	be wrapped in 
	this class. Synchronization is based on recursive mutex and read-copy-update
	pattern (read comments above for details). Locking must be done explicitly 
	by code accessing the internal data. Wrapper sharing between threads must 
	be done via pointer to avoid having two or more wrappers of the same data. 
	To simplify this scenario, wrapper is expected to be further wrapped into 
	shared pointer in which case the access to the internal data must be done 
	by double deference. Wrapping the wrapper into shared pointer also avoids 
	ownership assignment to one of the threads. */
template<class T>
class SharedData {
public:
	SharedDataGuard lock;

	/** Construct empty wrapper. */
	SharedData() : data(NULL) {}
	/** Construct data wrapper. Wrapper becomes owner of data, so the
	    raw pointer to data should not be used anywhere else. */
	SharedData(T *data) : data(data) {}
	/** Destruct wrapper alongside with the internal data. */
	virtual ~SharedData() { delete data; }
	/** Yield ownership of current data and become owner of new data. 
	    Raw pointer to old data is returned to caller which is responsible
		for deallocation. */
	T *swap(T *newData) { T *oldData = data; data = newData; return oldData; }
	
	T &operator*() { return *data; }
	T *operator->() { return data; }

protected:
	T *data;

private:
	SharedData(const SharedData<T> &other); // avoid copying of the wrapper
	SharedData<T> &operator=(const SharedData<T> &other); // avoid copying of the wrapper
};

typedef boost::shared_ptr<SharedData<multi_img_base> > multi_img_base_ptr;
typedef boost::shared_ptr<SharedData<multi_img> > multi_img_ptr;
typedef boost::shared_ptr<SharedData<cv::Mat_<cv::Vec3f> > > mat_vec3f_ptr;
typedef boost::shared_ptr<SharedData<std::pair<multi_img::Value, multi_img::Value> > > data_range_ptr;

#ifdef WITH_QT
typedef boost::shared_ptr<SharedData<QImage> > qimage_ptr;
#endif

#endif
