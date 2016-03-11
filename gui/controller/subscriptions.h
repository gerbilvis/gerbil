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

#include <unordered_set>
#include <functional>

class QObject;

/* needed fix for C++11 not being able to implicitely hash enum classes */
struct EnumHash {
    template <typename T>
    std::size_t operator()(T t) const {
        return static_cast<std::size_t>(t);
    }
};
template <typename Key>
using HashType = typename std::conditional<
std::is_enum<Key>::value, EnumHash, std::hash<Key>>::type;

/** Custom hash functor class for std library applications.
 *  Defaults to std::hash<T>(), specialize operator() as necessary.
*/
template <typename T>
struct GenericHash {
	size_t operator()(T const& v) const {
		return HashType<T>()(v);
	}
};

/** Identify representation and band pair. */
struct BandId {
	BandId(representation::t repr,  int band)
		: repr(repr), band(band)
	{}
	BandId(BandId const& other)
		: repr(other.repr), band(other.band)
	{}

	bool operator==(BandId const& other) const {
		return other.repr == repr && other.band == band;
	}

	// image representation type
	representation::t repr;
	// band number
	int band;
};

template <>
struct GenericHash<BandId> {
	size_t operator()(BandId const& v) const;
};

/** Identifies a subscription by subscriber and target type Target.
 *
 * IDTYPE must implement operator==().
 */
template <typename Target>
struct Subscription {

	struct Hash {
		std::size_t operator()(Subscription<Target> const& v) const {
			// IDTYPE will most likely not be a pointer. Thus we have disjunct hash
			// spaces we can safely XOR; see discussion at
			// http://stackoverflow.com/questions/5889238/why-is-xor-the-default-way-to-combine-hashes,
			// XOR vs. ADD.
			// We reserve 13 bits of at least 32 for the subscriber hash.
			return std::hash<void*>()((void *)v.subscriber) ^
					(GenericHash<Target>()(v.id) << 13);
		}
	};

	// Set type using IDTYPE as value_type.
	typedef std::unordered_set<Target, GenericHash<Target> > KeySet;

	// Set type using Subscription<IDTYPE> as value_type.
	/** unordered_set with enhanced functionality.
	 * Note: *never* introduce new class members to std containers
	 */
	class Set : public std::unordered_set<Subscription<Target>, Hash> {
	public:
		/* check if there is any subscription for a target */
		bool subscribed(Target t) {
			for (auto s : *this) {
				if (s.id == t)
					return true;
			}
			return false;
		}

		/** Add a subscriber.
		 * Returns true if this subscription did not exist yet.
		 */
		bool subscribe(QObject* subscriber, Target const& id) {
			auto result = this->insert(Subscription<Target>(subscriber, id));
			return result.second;
		}

		/** Remove a subscriber.
		 * Returns true if the specified subscription did exist.
		 */
		bool unsubscribe(QObject* subscriber, Target const& id)
		{
			return 0 < this->erase(Subscription<Target>(subscriber, id));
		}
	};

	Subscription(QObject const* subscriber, Target const& id)
		: subscriber(subscriber), id(id)
	{}

	Subscription(Subscription const& other)
		: subscriber(other.subscriber), id(other.id)
	{}

	Subscription& operator=(Subscription const& other) {
		return Subscription(other);
	}

	bool operator==(Subscription const& other) const {
		return other.subscriber == subscriber && other.id == id;
	}

	// The subscribing object.
	// WARNING: Never use this to access the pointed to object. The only purpose
	// of this pointer is to serve as an identification key.
	QObject const* const subscriber;
	const Target id;
};

struct Subscriptions {
	Subscription<BandId>::Set bands;
	Subscription<FalseColoring::Type>::Set colorings;
	Subscription<representation::t>::Set images;
};


#endif // SUBSCRIPTIONS_H
