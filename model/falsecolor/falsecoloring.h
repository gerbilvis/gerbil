#ifndef FALSECOLORING_H
#define FALSECOLORING_H

#include <cassert>
#include <ostream>
#include <QMetaClassInfo>

#include "../representation.h"

/** Encapsulated enum representing the different false coloring types. */
struct FalseColoring {
	/* if this is changed, also update static member FalseColoring::allList
	 * and prettyFalseColorNames in falsecolordock.cpp */
	enum Type {
		CMF=0,
		PCA,
		PCAGRAD,
		SOM,
		SOMGRAD
	};
	enum {SIZE=5};
	static bool isDeterministic(Type coloringType) {
		return !(coloringType == SOM || coloringType == SOMGRAD);
	}
	/** Returns true if the computation false coloring coloringType is based on
	 * image represesentation type and false otherwise.	 */
	static bool isBasedOn(Type coloringType, representation::t type) {
		switch (coloringType) {
		case CMF:
		case PCA:
		case SOM:
			return type == representation::IMG;
			break;
		case PCAGRAD:
		case SOMGRAD:
			return type == representation::GRAD;
			break;
		default:
            assert(false); // this should not happen
            return false; // prevent compiler warning
		}
	}

	static QList<Type> all() { return allList;	}
	static size_t size() { return SIZE; }
private:
	static QList<Type> allList;
};
Q_DECLARE_METATYPE(FalseColoring)
std::ostream &operator<<(std::ostream& os, const FalseColoring::Type& coloringType);

#endif // FALSECOLORING_H
