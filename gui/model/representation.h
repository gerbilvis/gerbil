#ifndef REPRESENTATION_H
#define REPRESENTATION_H

#include <QMap>

struct representation {

	enum t {
		// if you add a repres., also change operator<< and str()!
		IMG = 0, NORM, GRAD, IMGPCA, GRADPCA, REPSIZE
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
			"Original Image",
			"Normalized Image",
			"Spectral Gradient",
			"Image PCA",
			"Spectral Gradient PCA"
		};
		return str[r];
	}

	static bool valid(t r) { return 0 <= r && r < REPSIZE; }

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
