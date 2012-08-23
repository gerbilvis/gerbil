#include "rectangle.h"

#include <QPainter>
#include <QEvent>
#include <QMouseEvent>
#include <QDebug>

#include <iostream>

Rectangle::Rectangle(QWidget *parent, QRect *rect, int bdPix, QColor bdCol,
		   QColor bgCol, bool sq, int mouseGrabRange)
	: QLabel(parent), bdPix(bdPix), bdColor(bdCol), bgColor(bgCol), sqFlag(sq),
	  taskMoveRectangle(false), taskResizeRectangle(false), 
	  grabPos(::vole::NO_POSITION), mouseGrabRange(mouseGrabRange)
{
	// set geometry and style
	setMinimumSize(rect->size());
	setAutoFillBackground(false);
	move(rect->topLeft());
	resize(rect->size());
	activateWindow();
	setFrameStyle(QFrame::NoFrame);
}

// draw the widget
void Rectangle::paintEvent(QPaintEvent*) {
	QPainter paint(this);

	// set brush for drawing the background
	QBrush br(bgColor);
//    paint.setBrush(Qt::NoBrush);
	paint.setBrush(br);
	paint.setBackgroundMode(Qt::TransparentMode);

	// set the pen for drawing the rectangle border
	QPen pen;
	pen.setWidth(bdPix);
	pen.setStyle(Qt::SolidLine);
	pen.setColor(bdColor);
	paint.setPen(pen);
	paint.drawRect(QRect(0, 0, width() - 1, height() - 1));

	// draw the text of the label
	pen.setColor(Qt::white);
	paint.drawText(4, 12, text());

	// end the drawing
	paint.end();
}

// move and resize widget
bool Rectangle::eventFilter(QObject* obj, QEvent *ev) {

	// pass the event on to the parent class?
	if (obj != this) { return QWidget::eventFilter(obj, ev); }
	// repaint event ?
	if (ev->type() == QEvent::Paint) { return false; }
	if (ev->type() == QEvent::Enter) {
		setMouseTracking(true);
		return QWidget::eventFilter(obj, ev); 
	}
	if (ev->type() == QEvent::Leave) {
		unsetCursor();
		grabPos = ::vole::NO_POSITION;
		setMouseTracking(false);
		return QWidget::eventFilter(obj, ev);
	}
	if (ev->type() == QEvent::MouseButtonRelease) {
		std::cout << "release" << std::endl;
		taskMoveRectangle = false;
		taskResizeRectangle = false;
		unsetCursor();
		grabPos = ::vole::NO_POSITION;
		return QWidget::eventFilter(obj, ev);
	}

	if ((ev->type() == QEvent::MouseButtonPress)
		&& (static_cast<QMouseEvent*>(ev)->button() == Qt::LeftButton))
	{
		std::cout << "lmb pressed" << std::endl;
		lastMousePos = static_cast<QMouseEvent*>(ev)->pos(); // save mouse pos in rect
		// move or resize rectangle?
		if (!taskResizeRectangle && (grabPos == ::vole::CENTER)) 
			taskMoveRectangle = true; // save clicked event
		if (!taskMoveRectangle && (grabPos != ::vole::CENTER) && (grabPos != ::vole::NO_POSITION))
			taskResizeRectangle = true;
		return QWidget::eventFilter(obj, ev);
	}
	
	// only option left: the mouse has been moved
	if (ev->type() != QEvent::MouseMove) { return QWidget::eventFilter(obj, ev); }

	if (!taskMoveRectangle && !taskResizeRectangle) { // mouse button has not been pressed so far
		// global mouse position
		QPoint mpos = static_cast<QMouseEvent*>(ev)->pos();

		// create sensitive region for resizing the rectangle
		int   w   = frameSize().width();
		int   h   = frameSize().height();
		// mouse cursor grabbing regions for resizing
		QRegion t(0, 0, w, mouseGrabRange, QRegion::Rectangle);
		QRegion b(0, h - mouseGrabRange, w, mouseGrabRange, QRegion::Rectangle);
		QRegion l(0, 0, mouseGrabRange, h, QRegion::Rectangle);
		QRegion r(w - mouseGrabRange, 0, mouseGrabRange, h, QRegion::Rectangle);

		if (checkRegion(t.intersected(l), mpos, ::vole::TOP_LEFT)) goto done;
		if (checkRegion(t.intersected(r), mpos, ::vole::TOP_RIGHT)) goto done;
		if (checkRegion(b.intersected(l), mpos, ::vole::BOTTOM_LEFT)) goto done;
		if (checkRegion(b.intersected(r), mpos, ::vole::BOTTOM_RIGHT)) goto done;
		if (!sqFlag && checkRegion(t, mpos, ::vole::TOP)) goto done;
		if (!sqFlag && checkRegion(b, mpos, ::vole::BOTTOM)) goto done;
		if (!sqFlag && checkRegion(l, mpos, ::vole::LEFT)) goto done;
		if (!sqFlag && checkRegion(r, mpos, ::vole::RIGHT)) goto done;
		// else
		grabPos = ::vole::CENTER;
		unsetCursor();
done:;
		// FIXME return here?
	}
	if (taskMoveRectangle) {
		QPoint	mouseMoveVec = static_cast<QMouseEvent*>(ev)->pos() - lastMousePos;
		// new global position of left top of the rect
		QPoint	topLeft = this->parentWidget()->mapToParent(pos() + mouseMoveVec);
		QPoint bottomRight = topLeft + QPoint(width(), height());
		QPoint	pTopLeft, pBottomRight;

		// global pos of image widget
		pTopLeft     = this->parentWidget()->pos();
		pBottomRight = this->parentWidget()->pos() + QPoint(
			this->parentWidget()->width(), this->parentWidget()->height());

		int nx = pos().x(), ny = pos().y();

		// change in x possible ?
		if (topLeft.x() >= pTopLeft.x() && bottomRight.x() <= pBottomRight.x()) {
			nx = pos().x() + mouseMoveVec.x();
		}

		// change in y possible ?
		if (topLeft.y() >= pTopLeft.y() && bottomRight.y() <= pBottomRight.y()) {
			ny = pos().y() + mouseMoveVec.y();
		}

		// move the rect
		move(nx, ny);
	}
	
	if (taskResizeRectangle) {
		QPoint mousePos = static_cast<QMouseEvent*>(ev)->pos();
		QPoint currentPos = mapToParent(mousePos);
		if (currentPos.x() < 0) mousePos.setX(mapToParent(pos()).x() - currentPos.x());
		if (currentPos.y() < 0) mousePos.setY(mapToParent(pos()).y() - currentPos.y());
		if (currentPos.x() >= parentWidget()->width()) mousePos.setX(parentWidget()->width()-1);
		if (currentPos.y() >= parentWidget()->height()) currentPos.setY(parentWidget()->height()-1);

		QPoint newP1(pos());
		QPoint newP2(x() + width(), y() + height());
		// mouse movement changes one border
		QPoint delta(mousePos.x() - lastMousePos.x(), mousePos.y() - lastMousePos.y());
		// discard dimensions that do not fit.
		std::cout << "grabPos = " << grabPos << std::endl;
		if (sqFlag) { // or square? pick larger dimension
			if (abs(delta.x()) > abs(delta.y())) delta.setY(delta.x()); else delta.setX(delta.y());
		}


		std::cout << "resizing to " << newP1.x() << "," << newP1.y() << " - " << newP2.x() << "," << newP2.y() << ": width = " << newP2.x() - newP1.x() << "/" << newP2.y() - newP1.y() << std::endl;

//		ensureImageBoundaries(delta, newP1, newP2);
		ensureMinimumSize(delta);
		std::cout << "  resizing 2 " << newP1.x() << "," << newP1.y() << " - " << newP2.x() << "," << newP2.y() << ": width = " << newP2.x() - newP1.x() << "/" << newP2.y() - newP1.y() << std::endl;

		if (leftGrab())   { newP1.setX(newP1.x() + delta.x()); }
		if (topGrab())    { newP1.setY(newP1.y() + delta.y()); }
		if (rightGrab())  { newP2.setX(newP2.x() + delta.x()); }
		if (bottomGrab()) { newP2.setY(newP2.y() + delta.y()); }
		
		// limit the mouse, set the cursor explicitly
//		QCursor::setPos(this->mapToGlobal(QPoint(lastMousePos.x() + delta.x(), lastMousePos.y() + delta.y())));

		if ((delta.x() == 0) && (delta.y() == 0)) {
			std::cout << "aborted, no delta" << std::endl;
			return QWidget::eventFilter(obj, ev);
		}

		if (leftGrab() || topGrab()) { // first point is reference point for cursor
			move(newP1);
			if (isVerticalDirection()) {
				lastMousePos = mapFromParent(newP1);
			} else {
				if (isHorizontalDirection())
					lastMousePos = QPoint(mapFromParent(newP1).x(), lastMousePos.y());
				else
					lastMousePos = QPoint(lastMousePos.x(), mapFromParent(newP1).y());
			}
		} else { // second point is reference point for cursor
			if (isVerticalDirection()) {
				lastMousePos = mapFromParent(newP2);
			} else {
				if (isHorizontalDirection())
					lastMousePos = QPoint(mapFromParent(newP2).x(), lastMousePos.y());
				else
					lastMousePos = QPoint(lastMousePos.x(), mapFromParent(newP2).y());
			}
		}
		QCursor::setPos(this->mapToGlobal(lastMousePos));
		// resize the window.
		resize(newP2.x() - newP1.x(), newP2.y() - newP1.y());
//		lastMousePos = static_cast<QMouseEvent*>(ev)->pos();
		std::cout << "lastMousePos = " << lastMousePos.x() << "," << lastMousePos.y() << std::endl;
	}
	return QWidget::eventFilter(obj, ev);
}


bool Rectangle::checkRegion(
	QRegion reg,
	QPoint &mousePos,
	vole::RectangularPosition currentPosition)
{
	if (reg.contains(mousePos)) {
		grabPos = currentPosition;
		if ((currentPosition == ::vole::TOP_LEFT) || (currentPosition == ::vole::BOTTOM_RIGHT)) 
			setCursor(Qt::SizeFDiagCursor);
		if ((currentPosition == ::vole::BOTTOM_LEFT) || (currentPosition == ::vole::TOP_RIGHT)) 
			setCursor(Qt::SizeBDiagCursor);
		if ((currentPosition == ::vole::TOP) || (currentPosition == ::vole::BOTTOM))
			setCursor(Qt::SizeVerCursor);
		if ((currentPosition == ::vole::LEFT) || (currentPosition == ::vole::RIGHT))
			setCursor(Qt::SizeHorCursor);
		return true;
	}
	return false;
}


// checks if the mouse delta would run over a boundary in the parent widget and
// limits the change, such that the rectangle remains within the image.
// TODO: Most likely, this is suboptimal - consider the use of
// QRectangle.intersect(parent)
void Rectangle::ensureImageBoundaries(QPoint &delta, QPoint& p1, QPoint& p2) {
	QPoint posInParent1 = this->mapToParent(pos() + delta);
	QPoint posInParent2 = this->mapToParent(pos() + delta + QPoint(width(), height()));
	std::cout << "in parent: " << posInParent1.x() << "," << posInParent1.y() << " - " << posInParent2.x() << "," << posInParent2.y() << "  (delta ist " << delta.x() << "," << delta.y() << ", parentWidget dimensions = " << parentWidget()->width() << "," << parentWidget()->height() << ", minimumDim: " << minimumWidth() << "," << minimumHeight() << ")" << std::endl;
	if (topGrab() && (posInParent1.y() < 0)) { // adjust y1
		delta.setY(delta.y() - posInParent1.y()); // subtract overlap
		p1.setY(this->y() + delta.y());
	}
	if (leftGrab() && (posInParent1.x() < 0)) { // adjust x1
		delta.setX(delta.x() - posInParent1.x());
		p1.setX(this->x() + delta.x());
	}
	// exceeds bottom of parent widget?
	if (bottomGrab() && (posInParent2.y() > this->parentWidget()->height())) { 
		delta.setY(delta.y() - (posInParent2.y() - this->parentWidget()->height()));
		p2.setY(this->y() + this->height() + delta.y());
	}
	// exceeds right boundary of parent widget?
	if (rightGrab() && (posInParent2.x() > this->parentWidget()->width())) { 
		delta.setX(delta.x() - (posInParent2.x() - this->parentWidget()->width()));
		p2.setX(this->x() + this->width() + delta.x());
	}
	// square and we hit a widget boundary? make them equal, but this time
	// by picking the more limited dimension
	if (sqFlag && (delta.y() != delta.x())) { 
		if (abs(delta.x()) > abs(delta.y())) delta.setX(delta.y()); else delta.setY(delta.x());
	}
}

void Rectangle::ensureMinimumSize(QPoint &delta) {
	if (isVerticalDirection()) { // trim height
		if (topGrab()) {
			if (delta.y() - (height() - minimumHeight()) > 0) {
				delta.setY(height() - minimumHeight());
			}
		} else {
			if (height() + delta.y() - minimumHeight() < 0) {
				delta.setY(height() - minimumHeight());
			}
		}
	}
	if (isHorizontalDirection()) { // trim width
		std::cout << "isHorizontalDirection(), width() = " << width() << ", delta.x() = " << delta.x() << ", minimumWidth() = " << minimumWidth() << std::endl;
		if (leftGrab()) {
			if (delta.x() - (width() - minimumWidth()) > 0) {
				delta.setX(width() - minimumWidth());
			}
		} else { // rightGrab
			if (width() + delta.x() - minimumWidth() < 0) {
				delta.setX(width() - minimumWidth());
			}
		}
	}
	if (sqFlag) {
		if (delta.x() > delta.y()) { delta.setY(delta.x()); } else { delta.setX(delta.y()); } // pick smaller boundary: larger does not fit.
	}
}


bool Rectangle::isVerticalDirection() {
	return ((grabPos == ::vole::TOP)
		|| (grabPos == ::vole::BOTTOM)
		|| isDiagonalDirection());
}

bool Rectangle::isHorizontalDirection() {
	return ((grabPos == ::vole::LEFT)
		|| (grabPos == ::vole::RIGHT)
		|| isDiagonalDirection());
}

bool Rectangle::isDiagonalDirection() {
	return ((grabPos == ::vole::TOP_LEFT)
		|| (grabPos == ::vole::TOP_RIGHT)
		|| (grabPos == ::vole::BOTTOM_LEFT)
		|| (grabPos == ::vole::BOTTOM_RIGHT));
}

bool Rectangle::topGrab() {
	return ((grabPos == ::vole::TOP)
		   || (grabPos == ::vole::TOP_LEFT)
		   || (grabPos == ::vole::TOP_RIGHT));
}

bool Rectangle::bottomGrab() {
	return ((grabPos == ::vole::BOTTOM)
		   || (grabPos == ::vole::BOTTOM_LEFT)
		   || (grabPos == ::vole::BOTTOM_RIGHT));
}

bool Rectangle::leftGrab() {
	return ((grabPos == ::vole::LEFT)
		   || (grabPos == ::vole::BOTTOM_LEFT)
		   || (grabPos == ::vole::TOP_LEFT));
}

bool Rectangle::rightGrab() {
	return ((grabPos == ::vole::RIGHT)
		   || (grabPos == ::vole::BOTTOM_RIGHT)
		   || (grabPos == ::vole::TOP_RIGHT));
}



Rectangle::~Rectangle() {}


