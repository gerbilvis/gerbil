#ifndef FALSECOLOR_H
#define FALSECOLOR_H

#include <rgb.h>
#include <multi_img.h>

#include <QImage>
#include <QMap>

class FalseColor
{
	enum coloring {
		CMF = 0,
		PCA = 1,
		SOM = 2,
		COLSIZE = 3
	};

	struct payload {
		gerbil::RGB rgb;
		QImage img;
	};

	typedef QList<payload*> PayloadList;
	typedef QMap<coloring, payload*> PayloadMap;

public:
	FalseColor(const multi_img& img);
	~FalseColor();

	// Resets current true / false color representations
	void resetCaches();

	// always calls resetCaches()
	void setMultiImg(const multi_img& img);

	QImage get(coloring type);

private:
	const multi_img *img;
	PayloadMap map;
};

#endif // FALSECOLOR_H
