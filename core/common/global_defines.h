/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef GLOBAL_DEFINES_H
#define GLOBAL_DEFINES_H

#include <set>
#include <map>
#include <vector>
#include <cassert>

#include "vole_limits.h"

#ifdef WITH_OPENCV2
#include "cv.h"
#endif

#ifdef WITH_OPENCV2
namespace vole {
	typedef cv::Rect_<int> Recti;
	typedef cv::Rect_<double> Rectf;
	typedef cv::Point_<double> Point2f;
}
#endif

#ifndef VOLE_EPS
#define VOLE_EPS 0.00001
#endif

// TODO: remove me
#ifndef VOLE_TOO_LARGE
#define VOLE_TOO_LARGE 1000000
#endif

#ifndef VOLE_PI
	#ifdef M_PI
	#define VOLE_PI M_PI
	#else
	#define VOLE_PI		3.14159265358979323846
	#endif
#endif

#ifndef VOLE_PI_2
	#ifdef M_PI_2
	#define VOLE_PI_2 M_PI_2
	#else
	#define VOLE_PI_2     1.57079632679489661923
	#endif
#endif

#ifndef VOLE_PI_4
	#ifdef M_PI_4
	#define VOLE_PI_4 M_PI_4
	#else
	#define VOLE_PI_4     0.785398163397448309616
	#endif
#endif


#ifndef VOLE_INVALID_MEAN
#define VOLE_INVALID_MEAN -300
#endif

#ifndef VOLE_INVALID_VAR
#define VOLE_INVALID_VAR -1
#endif

class GlobalDefines {

	public:
	template<typename T>
	static T abs(T a) {
		if (a < 0) 
			return -a;
		else
			return a;
	}

	template<typename T>
	static T min(T a, T b) {
		if (a < b) 
			return a;
		else
			return b;
	}

	template<typename T>
	static T max(T a, T b) {
		if (a > b) 
			return a;
		else
			return b;
	}

	template<typename T>
	static int maxPos(std::vector<T> &vec, int start = 0, int end = VoleLimits::max<int>() ) {
		assert(vec.size() > 0);
		assert(end > 0);
		assert(start >= 0);
		assert(end - start >= 0);

		T maxVal = VoleLimits::min<T>();
		int maxPos = -1;
		int length = vec.size();
		if (end > length) end = length;
		for (int i = start; i < end; i++) {
				if (vec[i] > maxVal)  {
						maxVal = vec[i];
						maxPos = i;
				}
		}
		return maxPos;
	}


	template<typename T>
	static int minPos(std::vector<T> &vec, int start = 0, int end = VoleLimits::max<int>() ) {
		assert(vec.size() > 0);
		assert(end > 0);
		assert(start >= 0);
		assert(end - start >= 0);

		T minVal = VoleLimits::max<T>();
		int minPos = -1;
		int length = vec.size();
		if (end > length) end = length;
		for (int i = start; i < end; i++) {
				if (vec[i] < minVal)  {
						minVal = vec[i];
						minPos = i;
				}
		}
		return minPos;
	}

};


typedef std::map<const std::string, const std::string> TStrStrMap;
typedef std::pair<const std::string, const std::string> TStrStrPair;
typedef std::set<std::string> TStrSet;

#endif
