#ifndef BLENDING_IMAGE_H
#define BLENDING_IMAGE_H

#include <QScrollArea>
#include <QScrollBar>
#include <QMenu>
#include <QList>

#include "image_plane.h"
#include "draw_operation.h"

namespace Ui
{
class BlendingImage;
}

class BlendingImage : public DrawOperation
{
	Q_OBJECT

public slots:
	void alphaChanged(int);

public:
	BlendingImage();
	BlendingImage(QImage blendedImage, qreal alpha = static_cast<qreal>(0.5));

	void deinit();
	void init(QImage blendedImage, qreal alpha = static_cast<qreal>(0.5));

	void setActive(bool isActive);

	void draw(QImage *, QPainter *);
	
	void setBlendedImage(QImage);

	virtual ~BlendingImage();

protected:
	QImage blendedImage;
	qreal alpha;
	bool isActive;
	bool locked;
};

#endif // BLENDING_IMAGE_H
