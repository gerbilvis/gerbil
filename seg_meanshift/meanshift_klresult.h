#ifndef MEANSHIFT_KLRESULT_H
#define MEANSHIFT_KLRESULT_H

// These are for makeKeyValueMap() only. Maybe find better solution than
// adding these include dependencies.
#include <map>
#include <string>
#include <boost/any.hpp>

namespace seg_meanshift {

struct KLState {
	/** KLState flags. */
	enum t {
		Good	  = 0x0,   //! Good result, 0 for bitwise operations.
		Aborted   = 0x1,   //! The computation was aborted.
		NoneFound = 0x2    //! No solution found.
	};
};

class KLResult{
public:
	/** Create KLResult with flags s0 and s1 set. */ 
	KLResult(int K, int L,
			 KLState::t s0 = KLState::Good,
			 KLState::t s1 = KLState::Good
			)
		: K(K), L(L), state(KLState::t(s0 | s1))
	{}

	const int K;
	const int L;

	/** Returns true if flag s is set. */
	bool isState(KLState::t s) const {
		if (KLState::Good == s)
			return isGood();
		else
			return (state & s) != 0;
	}

	/** Returns true if findKL did not abort and found a valid result. */
	bool isGood() const {
		return state == KLState::Good;
	}

	/** Insert result and flags into key-value map.
	 *
	 * The inserted values are:
	 *
	 * 	*  findKL.K         int
	 *  *  findKL.L         int
	 *  *  findKL.aborted   bool
	 *  *  findKL.good      bool
	 */
	void  insertInto(std::map<std::string, boost::any> &dest);

private:
	std::map<std::string, boost::any> makeKeyValueMap() const;
	KLState::t state;
};

/** Print informational messages to cerr on KLResult state. */
void diagnoseKLResult(KLResult const& ret);

} // namespace seg_meanshift

#endif // MEANSHIFT_KLRESULT_H
