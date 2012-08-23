#ifndef MOUSE_OVER_AREA_H
#define MOUSE_OVER_AREA_H

#include <QImage>
#include <QEvent>

#include "image_plane.h"
#include "mouse_over_image_generator.h"

namespace Ui
{
class MouseOverArea;
}

class MouseOverArea : public ScrollArea
{
	Q_OBJECT

public:
	MouseOverArea(QWidget *parent = 0);
	void    setImageGenerator(MouseOverImageGenerator *generator) { this->generator = generator; }
	void    setOutputArea(ScrollArea *outputArea) { this->outputArea = outputArea; }	
	virtual ~MouseOverArea();

protected:
	bool    eventFilter(QObject*, QEvent*);

	MouseOverImageGenerator *generator;
	ScrollArea *outputArea;



private:

};

#endif // MOUSE_OVER_AREA_H
