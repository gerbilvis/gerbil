#ifndef VOLE_DRAW_OPERATION_H
#define VOLE_DRAW_OPERATION_H

#include <QObject>
#include <QImage>
#include <QPainter>

class DrawOperation : public QObject {
	Q_OBJECT

public:

	virtual void draw(QImage *, QPainter *p) = 0;
	
	virtual ~DrawOperation() {}
};

#endif // VOLE_DRAW_OPERATION_H
