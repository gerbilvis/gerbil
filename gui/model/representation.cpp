#include "representation.h"
#include <QMap>
#include <ostream>

static QMap<QString, representation::t> strToRepresentation;

representation::representation() {
	// skips INVALID
	for (int i = 1; i < REPSIZE; ++i) {
		allList.append(t(i));
	}

}

representation::t representation::fromStr(const QString &s)
{
	// lazy init
	if (strToRepresentation.empty()) {
		for (auto r : all()) {
			strToRepresentation[str(r)] = r;
		}
		representation::t invalid = representation::INVALID;
		strToRepresentation[str(invalid)] = invalid;
	}
	representation::t repr =
			strToRepresentation.value(s, representation::INVALID);
	return repr;

}

QString representation::prettyString(representation::t repr)
{
	if (!valid(repr)) {
		return "None";
	}
	const QString str[] = {
		"Invalid",
		"Original Image",
		"Normalized Image",
		"Spectral Gradient",
		"Image PCA",
		"Spectral Gradient PCA"
	};
	return str[repr];
}

const QString representation::str(representation::t repr) {
	if (!valid(repr)) {
		repr = INVALID;
	}
	const QString str[] = {
		"INVALID", "IMG", "NORM", "GRAD", "IMGPCA", "GRADPCA"
	};
	return str[repr];
}

std::ostream &operator<<(std::ostream& os, const representation::t& repr)
{
	os << representation::str(repr).toStdString().c_str();
	return os;
}

