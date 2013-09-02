#ifndef MODEL_REPRESENTATION_H
#define MODEL_REPRESENTATION_H

#include <QMap>

struct representation {

	enum t {
		IMG = 0,
		GRAD = 1,
		IMGPCA = 2,
		GRADPCA = 3,
		REPSIZE = 4
		// if you add something, also change operator<< and private constructor!
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
		if (r < 0 || r >= representation::REPSIZE) {
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

#endif // MODEL_REPRESENTATION_H
