#ifndef VOLE_MOUSE_MOVE_OPERATION_H
#define VOLE_MOUSE_MOVE_OPERATION_H

#include <QMouseEvent>
#include "draw_operation.h"

class MouseMoveOperation : public DrawOperation {

public:
	virtual void mouseMoved(QWidget *, QMouseEvent *) = 0;

	virtual void mouseClicked(QWidget *, QMouseEvent *) { return; };
	virtual void mouseReleased(QWidget *, QMouseEvent *) { return; };
	
	virtual ~MouseMoveOperation() {}
};

#endif // VOLE_MOUSE_MOVE_OPERATION_H
