/*
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "scaledview.h"

#include <stopwatch.h>
#include <QApplication>
#include <QPainter>
#include <QGraphicsSceneEvent>

#include <iostream>

/* TODO: do we really want sample buffers for these views? configurable?
 */
ScaledView::ScaledView()
    : width(50), height(50), // values don't matter much, but should be over 0
      zoom(1), inputMode(InputMode::Zoom)
{
	// by default small offsets; can be altered from outside
	offLeft = offTop = offRight = offBottom = 2;
}

void ScaledView::updateSizeHint()
{
	float src_aspect = 1.f;
	if (!pixmap.isNull())
		src_aspect = pixmap.width()/(float)pixmap.height();
	emit newSizeHint(QSize(300*src_aspect, 300));
}

void ScaledView::scalerUpdate() {
	// inverted transform to handle input, window damage etc.
	scalerI = scaler.inverted();

	// let the view know about the geometry we actually do occupy
	emit newContentRect(scaler.mapRect(pixmap.rect()));
}

void ScaledView::setPixmap(QPixmap p)
{
	bool cond = (p.width() != pixmap.width()
	                          || p.height() != pixmap.height());

	pixmap = p;

	if (cond) {
		resizeEvent();
		updateSizeHint();
	}
}

void ScaledView::drawBackground(QPainter *painter, const QRectF &rect)
{
	// update geometry
	int nwidth = painter->device()->width();
	int nheight = painter->device()->height();
	if (nwidth != width || nheight != height) {
		width = nwidth;
		height = nheight;
		resizeEvent();
	}

	// paint
	paintEvent(painter, rect);
}

void ScaledView::resizeEvent()
{
	if (pixmap.isNull())
		return;

	QRectF rect = scaler.mapRect(pixmap.rect());
	if (zoom > 1 && !sceneRect().contains(rect)) {
		adjustBoundaries();
		return;
	}

	// determine scale of correct aspect-ratio
	float src_aspect = pixmap.width()/(float)pixmap.height();
	float dest_aspect = width/(float)height;
	float w;	// new width
	if (src_aspect > dest_aspect)
		w = (width - offLeft - offRight);
	else
		w = (height - offTop - offBottom)*src_aspect;

	/* centering */
	scaler.reset();
	scaler.translate(offLeft + (width - offLeft - offRight - w)/2.f,
	                 offTop + (height - offTop - offBottom - w/src_aspect)/2.f);
	/* scaling */
	zoom = 1;
	float scale = w/pixmap.width();
	scaler.scale(scale, scale);
	scalerUpdate();
}

void ScaledView::paintEvent(QPainter *painter, const QRectF &rect)
{
	if (!pixmap) {
		painter->fillRect(rect, QBrush(Qt::gray, Qt::BDiagPattern));
		drawWaitMessage(painter);
		return;
	}

	fillBackground(painter, rect);

	painter->save();

	painter->setRenderHint(QPainter::SmoothPixmapTransform);
	painter->setWorldTransform(scaler);
	QRectF damaged = scalerI.mapRect(rect);
	painter->drawPixmap(damaged, pixmap, damaged);

	painter->restore();
}

void ScaledView::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	QGraphicsScene::mouseMoveEvent(event);
	cursorAction(event);

	if (inputMode != InputMode::Zoom)
		return;

	// avoid jitter
	if (!(event->buttons() & Qt::LeftButton) || zoom == 1)
		return;

	QRectF rect = scaler.mapRect(pixmap.rect());
	if (!rect.contains(event->scenePos()))
		return;

	// obtain current cursor and last cursor position in pixmap coordinates
	QPointF lastonscene = scalerI.map(event->lastScenePos());
	QPointF curronscene = scalerI.map(event->scenePos());

	qreal x = curronscene.x() - lastonscene.x();
	qreal y = curronscene.y() - lastonscene.y();

	scaler.translate(x,y);
	adjustBoundaries();
}

void ScaledView::adjustBoundaries()
{
	QRectF rect = scaler.mapRect(pixmap.rect());
	QRectF sceneRect = this->sceneRect();

	if (rect.width() > sceneRect.width()) {
		if (rect.x() > offLeft) {
			alignLeft();
		} else if (rect.x() + rect.width() < sceneRect.width() - offRight) {
			alignRight();
		}
	} else if (rect.x() < offLeft
	           || rect.x() + rect.width() > sceneRect.width() - offRight) {
		int left = rect.x() - offLeft;
		int right = sceneRect.width() - offRight - rect.x() - rect.width();

		if (left<right) {
			alignLeft();
		} else {
			alignRight();
		}
	}

	if (rect.height() > sceneRect.height()) {
		if (rect.y() > offTop) {
			alignTop();
		} else if (rect.y() + rect.height() < sceneRect.height() - offBottom) {
			alignBottom();
		}
	} else if (rect.y() < offTop
	           || rect.y() + rect.height() > sceneRect.height() - offBottom) {
		int top = rect.y() - offTop;
		int bottom = sceneRect.height() - offBottom - rect.y() - rect.height();

		if (top<bottom) {
			alignTop();
		} else {
			alignBottom();
		}
	}

	scalerUpdate();
	update();
}

void ScaledView::mousePressEvent(QGraphicsSceneMouseEvent *ev)
{
	// check for scene elements first (we are technically the background)
	QGraphicsScene::mousePressEvent(ev);
	if (ev->isAccepted())
		return;

	if (ev->button() == Qt::LeftButton && inputMode == InputMode::Zoom) {
		QApplication::setOverrideCursor(Qt::ClosedHandCursor);
	}

	cursorAction(ev, true);
}

void ScaledView::mouseReleaseEvent(QGraphicsSceneMouseEvent *ev)
{
	QGraphicsScene::mouseReleaseEvent(ev);
	if (ev->button() == Qt::LeftButton && inputMode == InputMode::Zoom) {
		QApplication::restoreOverrideCursor();
	}
}

void ScaledView::drawWaitMessage(QPainter *painter)
{
	painter->save();
	// darken
	painter->fillRect(sceneRect(), QColor(0, 0, 0, 127));

	// text in larger size with nice color
	painter->setPen(QColor(255, 230, 0));
	QFont tmp(font());
	tmp.setPointSize(tmp.pointSize() * 1.75);
	painter->setFont(tmp);
	painter->drawText(sceneRect(), Qt::AlignCenter,
	                  QString::fromUtf8("Calculating…"));
	painter->restore();
}

void ScaledView::cursorAction(QGraphicsSceneMouseEvent *ev, bool click)
{
	QPointF cursorF = scalerI.map(QPointF(ev->scenePos()));
	QPoint cursor(std::floor(cursorF.x() - 0.25),
	              std::floor(cursorF.y() - 0.25));

	if (ev->buttons() & Qt::RightButton && inputMode != InputMode::Seed
	    && inputMode != InputMode::Target) {
		showContextMenu(ev->screenPos());
	} else if (ev->buttons() == Qt::NoButton) {
		// overlay in spectral views but not during pixel labeling (reduce lag)
		emit pixelOverlay(cursor.y(), cursor.x());
	}

	if (ev->button() != Qt::NoButton && inputMode == InputMode::Target) {
		inputMode = InputMode::Zoom;
		actionTarget->setEnabled(true);
		if (ev->buttons() & Qt::LeftButton) {
			QRectF rect = scaler.mapRect(pixmap.rect());
			if (rect.contains(ev->scenePos())) {
				QPoint point = QPointF(scalerI.map(ev->scenePos())).toPoint();
				emit requestSpecSim(point.x(), point.y());
			}
		}
	}

	// note: we always have identity transform between scene and view
	if (!itemAt(ev->scenePos(), QTransform())) {
		updateCursor();
	}
}

void ScaledView::leaveEvent()
{
	QApplication::restoreOverrideCursor();
	// invalidate previous overlay
	emit pixelOverlay(-1,-1);
	update();
}

void ScaledView::updateInputMode()
{
	QAction* sender = (QAction*) QObject::sender();
	inputMode = sender->data().value<InputMode>();
	updateCursor();
}

void ScaledView::updateCursor()
{
	if (inputMode == InputMode::Target) {
		emit requestCursor(Qt::CrossCursor);
	} else if (inputMode == InputMode::Zoom) {
		emit requestCursor(Qt::OpenHandCursor);
	} else {
		emit requestCursor(Qt::ArrowCursor);
	}
}

void ScaledView::wheelEvent(QGraphicsSceneWheelEvent *event)
{
	QGraphicsScene::wheelEvent(event);

	if (inputMode != InputMode::Zoom)
		return;

	QRectF rect = scaler.mapRect(pixmap.rect());
	qreal newzoom;

	if (event->delta() > 0) {
		newzoom = 1.25;
	} else {
		newzoom = 0.8;
	}

	if ((zoom < 1 && newzoom > 1) ||
	    (zoom > 1 && zoom*newzoom <= 1)) {
		zoom = 1;
		resizeEvent();
	} else if (zoom == 1 && newzoom < 1
	           && pixmap.width()/rect.width() < 1) {
		scaleOriginal();
	} else if (zoom > 1 || newzoom > 1) {
		//obtain cursor position in scene coordinates
		QPointF scene = event->scenePos();
		//obtain cursor position in pixmap coordinates
		QPointF local = scalerI.map(scene);

		zoom *= newzoom;
		//scaling
		scaler.scale(newzoom, newzoom);

		//after scaling there's different point under cursor
		//so we have to obtain cursor position in pixmap coordinates
		//once again
		QPointF newlocal = scaler.inverted().map(scene);

		//translate the by the difference
		QPointF diff = newlocal - local;
		scaler.translate(diff.x(), diff.y());
		adjustBoundaries();
	}
}

void ScaledView::scaleOriginal()
{
	qreal currzoom = zoom;
	zoom = 1;
	resizeEvent();

	QRectF rect = scaler.mapRect(pixmap.rect());
	qreal ratio = pixmap.width() / rect.width();

	scaler.scale(ratio, ratio);
	zoom = currzoom * ratio;

	qreal x = 0.f;
	qreal y = 0.f;

	rect = scaler.mapRect(pixmap.rect());
	QRectF sceneRect = this->sceneRect();

	if (rect.width() < sceneRect.width() - offLeft - offRight) {
		QPointF space((sceneRect.width() - offLeft - offRight - rect.width())/2.f, 0.f);
		space = scaler.inverted().map(space);
		QPointF pos(rect.x(), 0.f);
		pos = scaler.inverted().map(pos);

		x = space.x() - pos.x() + offLeft;
	}

	if (rect.height() < sceneRect.height() - offTop - offBottom) {
		QPointF space(0.f,
		              (sceneRect.height() - offTop - offBottom - rect.height())/2.f);
		space = scaler.inverted().map(space);
		QPointF pos(0.f, rect.y());
		pos = scaler.inverted().map(pos);

		y = space.y() - pos.y() + offTop;
	}

	scaler.translate(x,y);
	scalerUpdate();
	update();

}

QMenu* ScaledView::createContextMenu()
{
	QMenu* contextMenu = new QMenu();

	QAction* tmp;
	tmp = contextMenu->addAction("Scale best fit");
	tmp->setIcon(QIcon::fromTheme("zoom-best-fit"));
	tmp->setIconVisibleInMenu(true);
	connect(tmp, SIGNAL(triggered()), this, SLOT(fitScene()));

	tmp = contextMenu->addAction("Scale 100%");
	tmp->setIcon(QIcon::fromTheme("zoom-original"));
	tmp->setIconVisibleInMenu(true);
	connect(tmp, SIGNAL(triggered()), this, SLOT(scaleOriginal()));

	connect(contextMenu, SIGNAL(aboutToHide()),
	        this, SIGNAL(updateScrolling()));

	return contextMenu;
}

void ScaledView::showContextMenu(QPoint screenpoint)
{
	if (!contextMenu) contextMenu = createContextMenu();
	contextMenu->exec(screenpoint);
}

void ScaledView::alignLeft()
{
	qreal x = 0.f;
	QRectF rect = scaler.mapRect(pixmap.rect());

	QPointF left(offLeft, 0.f);
	left = scaler.inverted().map(left);
	QPointF pos(rect.x(), 0.f);
	pos = scaler.inverted().map(pos);

	x = left.x() - pos.x();
	scaler.translate(x, 0.f);
}

void ScaledView::alignRight()
{
	qreal x = 0.f;
	QRectF rect = scaler.mapRect(pixmap.rect());
	QRectF sceneRect = this->sceneRect();

	QPointF right(sceneRect.width() - offRight, 0.f);
	right = scaler.inverted().map(right);
	QPointF pos(rect.x()+rect.width(), 0.f);
	pos = scaler.inverted().map(pos);

	x = right.x() - pos.x();
	scaler.translate(x, 0.f);
}

void ScaledView::alignBottom()
{
	qreal y = 0.f;
	QRectF rect = scaler.mapRect(pixmap.rect());
	QRectF sceneRect = this->sceneRect();

	QPointF bottom(0.f, sceneRect.height() - offBottom);
	bottom = scaler.inverted().map(bottom);
	QPointF pos(0.f, rect.y() + rect.height());
	pos = scaler.inverted().map(pos);

	y = bottom.y() - pos.y();
	scaler.translate(0.f, y);
}

void ScaledView::alignTop()
{
	qreal y = 0.f;
	QRectF rect = scaler.mapRect(pixmap.rect());

	QPointF top(0.f, offTop);
	top = scaler.inverted().map(top);
	QPointF pos(0.f, rect.y());
	pos = scaler.inverted().map(pos);

	y = top.y() - pos.y();
	scaler.translate(0.f, y);
}
