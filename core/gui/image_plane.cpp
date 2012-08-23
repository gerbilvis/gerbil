#include <QDebug>
#include <QMouseEvent>
#include <QMessageBox>
#include <QTransform>

#include "image_plane.h"

#include <iostream>

ImagePlane::ImagePlane(QWidget *parent)
	: QLabel(parent), popMenu(NULL), keyCTRL(false),
	  scaFak(1.)
{
	setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
	setScaledContents(true);
	setVisible(true);
	setMouseTracking(true);	
	setAlignment(Qt::AlignCenter);
}


void ImagePlane::paintEvent(QPaintEvent *ev) {
	QPainter p;
	// FIXME does not work
//	QTransform t;
	p.begin(this);
//	p.setWorldTransform(t.scale(scaFak, scaFak));
//	p.setWorldMatrixEnabled(true);
	p.setCompositionMode(QPainter::CompositionMode_SourceOver);
	p.drawImage(QPoint(0,0), img);
	std::vector<DrawOperation *>::iterator it = drawOperations.begin();
	for (; it != drawOperations.end(); it++) (*it)->draw(&img, &p);
	p.end();
}


void ImagePlane::addPaintEventNotification(DrawOperation *op) {
	drawOperations.push_back(op);
}

void ImagePlane::removePaintEventNotification(DrawOperation *op) {
	std::vector<DrawOperation *>::iterator it = drawOperations.begin();
	while (it != drawOperations.end()) {
		if (op == *it) {
			it = drawOperations.erase(it);
		} else {
			it++;
		}
	}
}

void ImagePlane::addMouseMoveEventNotification(MouseMoveOperation *op) {
	mouseMoveEvents.push_back(op);
}

void ImagePlane::removeMouseMoveEventNotification(MouseMoveOperation *op) {
	std::vector<MouseMoveOperation *>::iterator it = mouseMoveEvents.begin();
	while (it != mouseMoveEvents.end()) {
		if (op == *it) {
			it = mouseMoveEvents.erase(it);
		} else {
			it++;
		}
	}
}

void ImagePlane::addMousePressEventNotification(MouseMoveOperation *op) {
	mousePressEvents.push_back(op);
}

void ImagePlane::removeMousePressEventNotification(MouseMoveOperation *op) {
	std::vector<MouseMoveOperation *>::iterator it = mousePressEvents.begin();
	while (it != mousePressEvents.end()) {
		if (op == *it) {
			it = mousePressEvents.erase(it);
		} else {
			it++;
		}
	}
}

void ImagePlane::addMouseReleaseEventNotification(MouseMoveOperation *op) {
	mouseReleaseEvents.push_back(op);
}

void ImagePlane::removeMouseReleaseEventNotification(MouseMoveOperation *op) {
	std::vector<MouseMoveOperation *>::iterator it = mouseReleaseEvents.begin();
	while (it != mouseReleaseEvents.end()) {
		if (op == *it) {
			it = mouseReleaseEvents.erase(it);
		} else {
			it++;
		}
	}
}

void ImagePlane::addMouseDragEventNotification(MouseMoveOperation *op) {
	mouseDragEvents.push_back(op);
}

void ImagePlane::removeMouseDragEventNotification(MouseMoveOperation *op) {
	std::vector<MouseMoveOperation *>::iterator it = mouseDragEvents.begin();
	while (it != mouseDragEvents.end()) {
		if (op == *it) {
			it = mouseDragEvents.erase(it);
		} else {
			it++;
		}
	}
}

void ImagePlane::mouseMoveEvent ( QMouseEvent * event ) {
	std::vector<MouseMoveOperation *>::iterator it = mouseMoveEvents.begin();
	for (; it != mouseMoveEvents.end(); it++) (*it)->mouseMoved(this, event);
}

void ImagePlane::mousePressEvent ( QMouseEvent * event ) {
	// right button?
	if (event->button() == Qt::RightButton) {
		// any entry?
		if (popMenu != NULL) {
			// inside of the image label?
			if (((QRect)geometry()).contains(event->pos())) {
				// show popmenu
				popMenu->popup(event->globalPos());
				// save the point
				mevPoint = mapFromParent(event->pos());
			}
		}
	}

	if (event->button() != Qt::LeftButton) return; // react only on lmb
	std::vector<MouseMoveOperation *>::iterator it = mousePressEvents.begin();
	for (; it != mousePressEvents.end(); it++) (*it)->mouseClicked(this, event);
}

void ImagePlane::mouseReleaseEvent ( QMouseEvent * event ) {
	if (event->button() != Qt::LeftButton) return; // react only on lmb
	std::vector<MouseMoveOperation *>::iterator it = mouseReleaseEvents.begin();
	for (; it != mouseReleaseEvents.end(); it++) (*it)->mouseReleased(this, event);
}

void ImagePlane::dragMoveEvent ( QMouseEvent * event ) {
	if (event->button() != Qt::LeftButton) return; // react only on lmb
	std::vector<MouseMoveOperation *>::iterator it = mouseDragEvents.begin();
	for (; it != mouseDragEvents.end(); it++) (*it)->mouseMoved(this, event);
}

bool ImagePlane::eventFilter(QObject *obj, QEvent *ev) {
	// pass the event on to the parent class?
	if (obj != this) { return QWidget::eventFilter(obj, ev); }
	return QWidget::eventFilter(obj, ev);
}

QList<Rectangle*> &ImagePlane::getRectList() { return rectList; }

void ImagePlane::setPopMenu(QMenu* menu) { popMenu = menu; }

void ImagePlane::setImage(QImage nImg) {
	img = nImg;

	setPixmap(QPixmap::fromImage(img));
	setVisible(true);

	scaFak  = 1.;
	keyCTRL = false;
	update();
}

QRect ImagePlane::getRect(int i) {
	QRect r;

	if (i >= 0 && i < rectList.count()) {
		r = QRect((rectList[i])->pos() / scaFak,
		          (rectList[i])->size() / scaFak);
	}

	return r;
}

void ImagePlane::addRectangle(QRect rSizePos, QString text, QString objName) {

	Rectangle *r = new Rectangle(this, &rSizePos, 1, Qt::white, QColor(0, 255, 0, 127), false);

	// FIXME consider scaling here
	r->setFrameStyle(QFrame::Box | QFrame::Plain);
	r->setText(text);
	r->setAlignment(Qt::AlignLeading | Qt::AlignTop | Qt::AlignLeft);
	r->installEventFilter(r);
	r->setObjectName(objName);

	rectList.append(r);
}

void ImagePlane::addRectangle(int type) {
	Rectangle *r = NULL;

	// FIXME consider scaling here
	// which type?
	switch (type) {
	case 0:
		r = new Rectangle(this, new QRect(mevPoint.x(), mevPoint.y(), 20, 20), 1,
								Qt::red, QColor(255, 0, 0, 127), false);
		r->setText("A");
		break;
	case 1:
		r = new Rectangle(this, new QRect(mevPoint.x(), mevPoint.y(), 20, 20), 1,
								Qt::green, QColor(0, 255, 0, 127), false);
		r->setText("B");
		break;
	case 2:
		r = new Rectangle(this, new QRect(mevPoint.x(), mevPoint.y(), 20, 20), 1,
								Qt::magenta, QColor(255, 0, 255, 127), false);
		r->setText("C");
		break;
	case 3:
		r = new Rectangle(this, new QRect(mevPoint.x(), mevPoint.y(), 20, 20), 1,
								Qt::cyan, QColor(0, 255, 255, 127), false);
		r->setText("D");
		break;
	default:
		return;
	}


	r->setFrameStyle(QFrame::Box | QFrame::Plain);
	r->setAlignment(Qt::AlignLeading | Qt::AlignTop | Qt::AlignLeft);
	r->installEventFilter(r);
	r->setVisible(true);

	// create a new selection rect
	rectList.append(r);
}

void ImagePlane::removeRectangle() {
	// which rect covers p?
	for (QList<Rectangle*>::iterator i = rectList.begin(); i != rectList.end(); ++i) {
		Rectangle& r = **i;
		if (r.geometry().contains(mevPoint)) {
			// delete rect
			delete *i;
			rectList.erase(i);
			break;
		}
	}
}

// FIXME diese Funktion funktioniert nicht :(
void ImagePlane::ScaleImage(double nScaFak) {
	QList<Rectangle*>::iterator i;

	// only resize if image present
	if (img.isNull())
		return;
		
	// scale the image
	scaFak *= nScaFak;

	std::cout << "scaling from " << img.size().width() << "," << img.size().height() <<
		" to " << (scaFak * img.size()).width() << "," << (scaFak * img.size()).height()
		<< std::endl;

	/*
	resize(scaFak * img.size());
	QPixmap p = *pixmap();
	setPixmap(QPixmap::fromImage(img).scaled(scaFak * img.size()));
	*/

	// FIXME rescaling of the rectangles does not work
	/*
	// scale selection areas
	for (i = rectList.begin(); i != rectList.end(); ++i) {
		Rectangle& r = **i;
		r.move(r.pos() * nScaFak);
		r.resize(r.size() * nScaFak);
	}
	*/
}

void ImagePlane::wheelEvent(QWheelEvent* wev) {
	// mouse wheel event
	// CTRL key pressed?
	if (keyCTRL) {
		// do the zoom in / out
		if (wev->delta() > 0) {
			// zoom  in
			ScaleImage(1.25);
		} else {
			// zoom out
			ScaleImage(.8);
		}
	}
}

void ImagePlane::zoomIn() {
	ScaleImage(1.25);
}

void ImagePlane::zoomOut() {
	ScaleImage(0.8);
}

void ImagePlane::keyPressEvent(QKeyEvent* kev) {
	if (kev->key() == Qt::Key_Control) {
		// CTRL button pressed
		keyCTRL = true;
		setCursor(Qt::SizeAllCursor);
	}
}

void ImagePlane::keyReleaseEvent(QKeyEvent* kev) {
	if (kev->key() == Qt::Key_Control) {
		keyCTRL = false;
		unsetCursor();
	}
}

ImagePlane::~ImagePlane() {}
