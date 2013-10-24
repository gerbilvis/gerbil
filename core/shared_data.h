/*	
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef SHARED_DATA_H
#define SHARED_DATA_H

#ifdef WITH_BOOST_THREAD
#include <multi_img.h>
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
typedef boost::recursive_mutex SharedDataMutex;


// FIXME 2013-04-11 georg altmann:
// I do not see the point of the two different lock types. We are doing
// exclusive locking on the mutex always. There is no gradual
// read-/write-locking taking place anywhere AFAIK.
// Also it is not necessary to use unique_lock, lock_guard should do, since
// the typedefs are always used in that way and I can think of a use case
// where the delayed locking of unique_lock would be useful.

/** Lock that should be used by foreground threads (GUI, OpenGL) to prevent
    background worker thread to swap embedded raw pointer. Note that the lock 
	does not prevent other parties to modify the data in-place, that is if you 
	need mutual exclusion on the targeted data itself, you should equip the 
	targeted data by a mutex of your choice and properly lock on it depending 
	on a particular scenario. Often, the possible in-place modification of
	data by foreground threads can be avoided by temporarily disabling user
	input in the corresponding part of the GUI. */
typedef boost::unique_lock<SharedDataMutex> SharedDataLock;
/** Lock that should be used by background worker thread to swap embedded raw
    pointer once the calculation of new version of data is finished. Note that
	this should enforce usage of a simple variant of read-copy-update pattern.
	Simple in a sense that there can be only one swapper thread, which is
	expected to be background worker thread dispatching background tasks
	from the queue. Also note that it is perfectly fine if background worker
	modify locked data in-place instead of using RCU pattern - this might be 
	reasonable if it involves assignment of only a couple of primitive values. */
typedef boost::unique_lock<SharedDataMutex> SharedDataSwapLock;

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
	SharedDataMutex mutex;

	/** Construct empty wrapper. */
	SharedData() : data(NULL) {}
	/** Construct data wrapper. Wrapper becomes owner of data, so the
		raw pointer to data should not be used anywhere else. */
	SharedData(T *data) : data(data) {}
	/** Destruct wrapper alongside with the internal data. */
	virtual ~SharedData() { delete data; }
	/** De-allocate data and replace with new data */
	void replace(T *newData) {
		if (data != newData)
			delete data;
		data = newData;
	}
	/** Release data ownership */
	void release() { data = NULL; }
	
	T &operator*() { return *data; }
	T *operator->() { return data; }

	// implement the boost::Lockable concept
	void lock() { mutex.lock(); }
	bool try_lock() { return mutex.try_lock(); }
	void unlock() { mutex.unlock(); }

protected:
	T *data;
private:
	SharedData(const SharedData<T> &other); // avoid copying of the wrapper
	SharedData<T> &operator=(const SharedData<T> &other); // avoid copying of the wrapper
};

// Specialization of SharedData for multi_img_base / multi_img
//
// Most tasks in gerbil need to operate on a multi_img object. For this reason
// SharedData<multi_img_base> casts to multi_img in most access functions. If
// client code can restrict itself to multi_img_base functionality, it should
// use getBase() and overloaded swap(multi_img_base*).
//
// Notice: The SharedData concept is not suited for covariant type casting
// as implemented in boost::shared_ptr. As a consequence it was
// decided to have implicit casting to multi_img in
// SharedData<multi_img_base>. Client code has to be aware of the consequences
// of the implicit casts.

// Added functionality: The object can now also be initialized with a
// multi_img::ptr. It will keep the shared_ptr as shared pointers never give
// up ownership.

// possible optimization: use an extra multi_img* member to store the result
// of the cast.
template<>
class SharedData<multi_img_base> {
public:
	SharedDataMutex mutex;

	SharedData(multi_img_base *data) : data(data) {}
	SharedData(multi_img::ptr ptr) : data(ptr.get()), owner(ptr) {}

	void replace(multi_img_base *newData) {
		if (data == newData)
			return;

		if (owner.get()) // data is owned by a shared pointer
			owner.reset();
		else
			delete data;
		data = newData;
	}
	// throws bad_cast, if the encapsulated pointer points to multi_img_base
	// object
	multi_img &operator*() { return dynamic_cast<multi_img&>(*data); }
	// does not throw, may return NULL
	multi_img *operator->() {
		multi_img *ret = dynamic_cast<multi_img*>(data);
		return ret;
	}
	// Return base class image. Tasks that can restrict themselves to using
	// multi_img_base functionality should use this function.
	multi_img_base &getBase() {
		assert(data);
		return *data;
	}

	// implement the boost::Lockable concept
	void lock() { mutex.lock(); }
	bool try_lock() { return mutex.try_lock(); }
	void unlock() { mutex.unlock(); }

protected:
	multi_img_base *data;
	multi_img::ptr owner;
private:
	// non-copyable
	SharedData(const SharedData<multi_img_base> &other);
	SharedData<multi_img_base> &operator=(const SharedData<multi_img_base> &other); 
};

typedef SharedData<multi_img_base> SharedMultiImgBase;
// Guard type that locks the SharedData mutex
typedef boost::lock_guard<SharedData<multi_img_base> > SharedMultiImgBaseGuard;

// pointer to SharedData<multi_img_base> which really behaves like SharedData<multi_img>, see above
/* TODO: we discussed that this could be a SharedMultiImgBase const* instead
 * to avoid confusion about the nature/functionality of the ptr */
typedef boost::shared_ptr<SharedMultiImgBase> SharedMultiImgPtr;

typedef boost::shared_ptr<SharedData<cv::Mat3f> > mat3f_ptr;
typedef boost::shared_ptr<SharedData<multi_img::Range> > SharedMultiImgRangePtr;

// BUG
// There is no reasonable way to actually get the pointer to the multi_img
// from SharedData.
// multi_img_ptr x;
// typeof(*x) == SharedData<multi_img> != multi_img*
// typeof(**x) == multi_img
// typeof(&(**x)) == multi_img*    [doh!]

#ifdef WITH_QT
typedef boost::shared_ptr<SharedData<QImage> > qimage_ptr;
#endif

#endif
#endif

