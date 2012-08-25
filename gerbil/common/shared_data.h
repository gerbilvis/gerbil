/*	
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef SHARED_DATA_H
#define SHARED_DATA_H

#include <boost/thread/shared_mutex.hpp>

/** Templated wrapper class for any data shared between GUI thread and
    background worker thread. All data members of BackgroundTask that could
	be potentially accessed concurrently by GUI and worker thread shall
	be wrapped in this class. Synchronization is based on read-write lock.
	Locking must be done explicitly by code accessing the internal data.
	Wrapper sharing between threads must be done via pointer to avoid
	having two or more wrappers of the same data. To simplify this scenario, 
	wrapper is expected to be further wrapped into shared pointer in which 
	case the access to the internal data must be done by double deference.
	Wrapping the wrapper into shared pointer also avoids ownership 
	assignment to one of the threads. */
template<class T>
class SharedData {
public:
	typedef boost::shared_mutex Guard;
	typedef boost::shared_mutex::scoped_lock Write; ///< write lock
	typedef boost::shared_mutex::scoped_lock_shared Read; ///< read lock
	Guard lock;

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

#endif
