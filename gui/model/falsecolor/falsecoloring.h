#ifndef FALSECOLORING_H
#define FALSECOLORING_H

#include <cassert>
#include <ostream>
#include <stdexcept>
#include <QSet>
#include <QMetaClassInfo>

#include "../representation.h"

/** Encapsulated enum representing the different false coloring types. */
struct FalseColoring {
	/* This enum is fragile: If changed, update all static member functions,
	 * allList init andand prettyFalseColorNames in falsecolordock.cpp. */
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
			throw(std::runtime_error("FalseColoring::isBasedOn():"
									 "invalid FalseColoring::Type"));
		}
	}

	/** Maps a false coloring to the representation it is based on. */
	static representation::t basis(Type coloringType) {
		switch (coloringType) {
		case CMF:
		case PCA:
		case SOM:
			return representation::IMG;
			break;
		case PCAGRAD:
		case SOMGRAD:
			return representation::GRAD;
			break;
		default:
			throw(std::runtime_error("FalseColoring::basis():"
									 "invalid FalseColoring::Type"));
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
