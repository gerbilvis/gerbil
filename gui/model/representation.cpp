#include "representation.h"
#include <QMap>
#include <ostream>

representation::representation() {
	// skips INVALID
	for (int i = 1; i < REPSIZE; ++i) {
		allList.append(t(i));
	}
}

representation::t representation::fromStr(const QString &s)
{
	static QMap<QString, representation::t> mapping;
	if (mapping.empty()) { // lazy initialization
		for (auto r : all()) {
			mapping[str(r)] = r;
		}
	}
	return mapping.value(s, representation::INVALID);
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

