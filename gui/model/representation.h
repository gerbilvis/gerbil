#ifndef REPRESENTATION_H
#define REPRESENTATION_H

#include <QMap>

// FIXME These macro defs are a temporary workaround to get the upcoming relase
// functioning. We need to get task processing more efficient in the future.
//#define WITH_IMGPCA
//#define WITH_GRADPCA
#define REPSIZE 2 // depends on the variables above
struct representation {

	enum t {
		IMG = 0,
		GRAD = 1
#ifdef WITH_IMGPCA
		,IMGPCA = 2
#endif
#ifdef WITH_GRADPCA
		,GRADPCA = 3
#endif
		// if you add a repres., also change operator<< and private constructor!
	};

	// map of all representations for easy looping
	static const QMap<int, t>& all()
	{
		static representation r;
		return r.map;
	}

	// user-readable string representation (for debug output, see operator<<)
	static const char* str(t r)
	{
		if (r < 0 || r >= REPSIZE) {
			return "None";
		}
		const char * const str[] = {
			"Image Spectrum",
			"Spectral Gradient Spectrum",
			"Image PCA",
			"Spectral Gradient PCA"
		};
		return str[r];
	}

private:
	representation() {
		for (int i = 0; i < REPSIZE; ++i)
			map[i] = (t)i;
	}
	representation(const representation &);
	representation operator=(const representation &);

	/* for easy looping over all valid representations
	 * is a map to be ordered */
	QMap<int, t> map;
};

// representation in debug output
std::ostream &operator<<(std::ostream& os, const representation::t& r);

#endif // REPRESENTATION_H
