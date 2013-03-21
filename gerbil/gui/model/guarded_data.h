#ifndef GUARDED_DATA_H
#define GUARDED_DATA_H


/*!
 * \brief RAII style guarded pointer for locked data structures.
 *
 * GuardedData provides a pointer to a data object of type T. On construction
 * GuardedData locks the boost::recursive_mutex mutex associated with the data
 * object. When the GuardedData pointer is destroyed the
 * boost::recursive_mutex is unlocked.  Whenever a GuardedData pointer is
 * copied another level of locking is imposed on the mutex. That is until all
 * GuardedData objects holding a reference to the data object are destroyed,
 * the mutex remains locked.
 */
template<typename T>
class GuardedData {
public:
	GuardedData(T &data, boost::recursive_mutex &mutex) : m_data(data), m_mutex(mutex)
	{
		mutex.lock();
	}

	GuardedData(const GuardedData &other)
		: m_data(other.m_data), m_mutex(other.m_mutex)
	{
		m_mutex.lock();
	}

	~GuardedData()
	{
		m_mutex.unlock();
	}

	GuardedData& operator=(const GuardedData &other)
	{
		m_mutex.unlock();
		this->m_data = other.m_data;
		this->m_mutex = other.m_mutex;
		m_mutex.lock();
		return *this;
	}

	T &operator*() { return m_data; }
	T *operator->() { return &m_data; }

private:
	T &m_data;
	boost::recursive_mutex &m_mutex;
};

/*!
 * \brief Specialization of GuardedData for multi_img_base.
 */
class GuardedMultiImgBase : public GuardedData<multi_img_base>
{
public:

	GuardedMultiImgBase(multi_img_base &d, boost::recursive_mutex &mutex)
		: GuardedData<multi_img_base>(d,mutex)
	{}

	// default copy constructors and assignment

	/*!
	 * \brief Cast the guarded multi_img_base reference to multi_img.
	 * \return Reference to multi_img object.
	 * \note If the underlying multi_img_base reference does not refer to a multi_img object bad_cast is thrown.
	 */
	multi_img &downCast() throw (std::bad_cast) {
		multi_img_base &img_base = this->operator*();
		return dynamic_cast<multi_img&>(img_base);
	}
};

#endif // GUARDED_DATA_H
