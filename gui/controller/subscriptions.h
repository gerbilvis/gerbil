//***************************************************************************
// Controller subscription helper classes.
//
// These were put in an extra file to not bloat the Controller class with all
// the template definitions.
//***************************************************************************
#ifndef SUBSCRIPTIONS_H
#define SUBSCRIPTIONS_H

#include <model/representation.h>
#include <model/falsecolor/falsecoloring.h>

#include <boost/tr1/unordered_set.hpp>
#include <boost/tr1/functional.hpp>

class QObject;

template<typename T>
std::size_t make_hash(const T& v)
{
	return std::tr1::hash<T>()(v);
}

/** Functor class that wraps make_hash() for std library applications.
 *  Defaults to  std::tr1::hash<T>(), specialize make_hash() as necessary.;
*/
template <typename T>
struct Hash {
	typedef T argument_type;
	typedef std::size_t result_type;

	result_type operator()(argument_type const& t) const {
		return make_hash<T>(t);
	}
};

/** Identify multi_img representation and band. */
struct ImageBandId {
	ImageBandId(representation::t repr,  int bandx)
		: repr(repr), bandx(bandx)
	{}
	ImageBandId(ImageBandId const& other)
		: repr(other.repr), bandx(other.bandx)
	{}

	bool operator==(ImageBandId const& band) const {
		return band.repr == repr && band.bandx == bandx;
	}

	// image representation type
	representation::t repr;
	// band number
	int bandx;
};

template <>
std::size_t make_hash(const ImageBandId& t)
{
	// No XOR here, repr and bandx are both small and likely to collide.
	return 65437 * // prime
			make_hash<int>(t.repr + 7) +
			make_hash<int>((t.bandx << 16) + 1031 );
}

template <typename IDTYPE>
struct SubscriptionHash;


/** Identifies a subscription by subscriber and target type IDTYPE.
 *
 * IDTYPE must implement operator==().
 */
template <typename IDTYPE>
struct Subscription {
	typedef IDTYPE IdType;
	// Set type using IDTYPE as value_type.
	typedef std::tr1::unordered_set<IDTYPE, Hash<IDTYPE> > IdTypeSet;
	// Set type using Subscription<IDTYPE> as value_type.
	typedef std::tr1::unordered_set<Subscription<IDTYPE>, SubscriptionHash<IDTYPE> > Set;

	Subscription(QObject *subscriber, IDTYPE const& subsid)
		: subscriber(subscriber), subsid(subsid)
	{}

	Subscription(Subscription const& other)
		: subscriber(other.subscriber), subsid(other.subsid)
	{}

	Subscription & operator=(Subscription const& other) {
		return Subscription(other);
	}

	bool operator==(Subscription const& other) const {
		return other.subscriber == subscriber &&
				other.subsid == subsid;
	}

	QObject *subscriber;
	IDTYPE subsid;
};

template <typename IDTYPE>
struct SubscriptionHash {
	typedef Subscription<IDTYPE> argument_type;
	typedef std::size_t result_type;

	result_type operator()(argument_type const& t) const {
		// IDTYPE will most likely not be QObject*. Thus we have disjunct hash
		// spaces we can safely XOR; see discussion at
		// http://stackoverflow.com/questions/5889238/why-is-xor-the-default-way-to-combine-hashes,
		// XOR vs. ADD.
		// We reserve 13 bits of at least 32 for the subscriber hash.
		return make_hash<void*>(t.subscriber) ^
				(make_hash<IDTYPE>(t.subsid) << 13);
	}
};

struct Subscriptions {
	Subscription<ImageBandId>::Set			imageBand;
	Subscription<FalseColoring::Type>::Set  falseColor;
};

#endif // SUBSCRIPTIONS_H
