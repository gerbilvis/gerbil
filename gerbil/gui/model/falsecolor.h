#ifndef FALSECOLOR_H
#define FALSECOLOR_H

#include <rgb.h>
#include <multi_img.h>

#include <QImage>
#include <QMap>
#include <QObject>

class FalseColor : public QObject
{
	Q_OBJECT

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

	// resets current true / false color representations
	// on the next request, the color images are recalculated with possibly new multi_img data
	void resetCaches();

	// always calls resetCaches()
	void setMultiImg(const multi_img& img);

public slots:
	void request(coloring type);

signals:
	void loadComplete(QImage img, coloring type, bool changed);

private:
	const multi_img *img;
	PayloadMap map;
};

#endif // FALSECOLOR_H
