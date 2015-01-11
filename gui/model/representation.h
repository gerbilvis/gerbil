#ifndef REPRESENTATION_H
#define REPRESENTATION_H

#include <QList>
#include <QString>

struct representation {

	enum t {
		// If you add a repres., also change prettyString() and str().
		INVALID = 0, IMG, NORM, GRAD, IMGPCA, GRADPCA, REPSIZE
	};

	// Map enum string to t.
	static t fromStr(const QString& s);

	// QList of all representations without INVALID.
	static const QList<representation::t>& all()
	{
		static representation repr;
		return repr.allList;
	}

	// user-readable string representation (for debug output, see str)
	static QString prettyString(t repr);

	// Return enum identifier as string.
	static const QString str(t repr);

	static bool valid(t repr) { return 1 <= repr && repr < REPSIZE; }

private:
	representation();
	representation(const representation &);
	representation operator=(const representation &);

	// List of all representations without INVALID.
	QList<representation::t> allList;
};

// representation in debug output
std::ostream &operator<<(std::ostream& os, const representation::t& r);

#endif // REPRESENTATION_H
