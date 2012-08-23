#include "mouse_over_area.h"

MouseOverArea::MouseOverArea(QWidget *parent)
	: ScrollArea(parent), generator(NULL), outputArea(NULL)
{
}

bool MouseOverArea::eventFilter(QObject* obj, QEvent* ev)
{
	// pass the event on to the parent class?
	if (obj != this) { return QWidget::eventFilter(obj, ev); }
	
	if (ev->type() == QEvent::MouseMove) {
		// todo catch mouse movement here

		// FIND COORDINATES
		// QUERY GENERATOR
		// SEND OUTPUT TO SOMEBODY ELSE

		return QWidget::eventFilter(obj, ev); 
	}

	return QWidget::eventFilter(obj, ev); 
}

MouseOverArea::~MouseOverArea() {
}
