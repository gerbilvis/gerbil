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

#include <tuple>
#include <utility>
#include <unordered_set>
#include <functional>

class QObject;

template<typename T>
std::size_t make_hash(const T& v)
{
    return std::hash<T>()(v);
}

/** General std::hash hash functor for std::pair.
 *
 * The first two template arguments denote the template arguments for
 * std::pair<First,Second>.  The last two template arguments denote define the
 * type of hash function to use for the pair types.
 *
 * Note that std::hash<> is only defined for primitive types and pointers.
*/
template<typename First, typename Second,
		 typename FirstHashT = First, typename SecondHashT = Second>
struct pair_hash {
	typedef std::pair<First,Second> argument_type;
	typedef std::size_t result_type;

	result_type operator()(argument_type const& p) const {
		return make_hash<FirstHashT>(p.first) ^
				(make_hash<SecondHashT>(p.second) << 1);
	}
};

/** General std::hash hash functor for std::tuple<First, Second, Third>
 * analogous to pair_hash. */
template<typename First, typename Second, typename Third,
		 typename FirstHashT = First, typename SecondHashT = Second,
		 typename ThirdHashT = Third>
struct triple_hash {
	typedef std::tuple<First,Second, Third> argument_type;
	typedef std::size_t result_type;

	result_type operator()(argument_type const& t) const {
		return make_hash<FirstHashT>(std::get<0>(t)) ^
				(make_hash<SecondHashT>(std::get<1>(t)) << 1) ^
				(make_hash<ThirdHashT>(std::get<2>(t)) << 3);
	}
};

/// Image Band
typedef std::tuple<QObject*, representation::t, int> ImageBandSubscription;

// Hash function for ImageBandSubscription
typedef triple_hash<QObject*, representation::t, int, QObject*, int, int>
	ImageBandSubscriptionHash;
typedef std::unordered_set<ImageBandSubscription, ImageBandSubscriptionHash>
		ImageBandSubscriptionHashSet;

/// False Color
typedef std::pair<QObject*, FalseColoring::Type> FalseColorSubscription;

// Hash function for FalseColorSubscription
typedef pair_hash<QObject*, FalseColoring::Type, QObject*, int> FalseColorSubscriptionHash;
typedef std::unordered_set<FalseColorSubscription, FalseColorSubscriptionHash>
		FalseColorSubscriptionHashSet;

struct Subscriptions {
	ImageBandSubscriptionHashSet   imageBand;
	FalseColorSubscriptionHashSet  falseColor;
};

#endif // SUBSCRIPTIONS_H
