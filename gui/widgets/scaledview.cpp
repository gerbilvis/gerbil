/*
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "scaledview.h"

#include <stopwatch.h>
#include <QGLWidget>
#include <QPainter>
#include <QGraphicsSceneEvent>
#include <QDebug>

#include <iostream>

/* TODO: do we really want sample buffers for these views? configurable?
 */
ScaledView::ScaledView()
    : width(50), height(50), // values don't matter much, but should be over 0
      zoom(1), sm(Zoom)
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

void ScaledView::setPixmap(QPixmap p)
{
    bool cond = (p.width() != pixmap.width()
            || p.height() != pixmap.height());

    pixmap = p;
    if(cond)
    {
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
	float scale = w/pixmap.width();
    scaler.scale(scale, scale);

	// inverted transform to handle input
    scalerI = scaler.inverted();

	// let the view know about the geometry we actually do occupy
    emit newContentRect(scaler.mapRect(pixmap.rect()));
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

    if(sm != Zoom) return;

    if(event->buttons() == Qt::LeftButton)
    {
    //Obtain current cursor and last cursor position
    //in pixmap coordinates
    QPointF lastonscene = scalerI.map(event->lastScenePos());
    QPointF curronscene = scalerI.map(event->scenePos());

    //qDebug() << "CURRONSCENE" << curronscene;

    qreal x = curronscene.x() - lastonscene.x();
    qreal y = curronscene.y() - lastonscene.y();

    //qDebug() << "xp" << x << "yp" << y;

    scaler.translate(x,y);
    scalerI = scaler.inverted();

   // adjustBoundaries();
    }

}

void ScaledView::mousePressEvent(QGraphicsSceneMouseEvent *ev)
{
	// check for scene elements first (we are technically the background)
	QGraphicsScene::mousePressEvent(ev);
	if (ev->isAccepted())
		return;

	cursorAction(ev, true);
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
}


void ScaledView::wheelEvent(QGraphicsSceneWheelEvent *event)
{
      QGraphicsScene::wheelEvent(event);

      if(sm != Zoom) return;

      qreal newzoom;

      if (event->delta() > 0)
      {
          newzoom = 1.25;
      }
      else
      {
          newzoom = 0.8;
      }


      if(zoom*newzoom < 1)
      {

          resizeEvent();

      }
      else
      {
          //obtain cursor position in scene coordinates
          QPointF scene = event->scenePos();
          //obtain cursor position in pixmap coordinates
          QPointF local = scalerI.map(scene);

          zoom *= newzoom;
          //scaling
          scaler.scale(newzoom, newzoom);
          scalerI = scaler.inverted();

          //after scaling there's different point under cursor
          //so we have to obtain cursor position in pixmap coordinates
          //once again
          QPointF newlocal = scalerI.map(scene);

          //translate the by the difference
          QPointF diff = newlocal - local;
          scaler.translate(diff.x(), diff.y());
          scalerI = scaler.inverted();


          //adjustBoundaries();
      }

}

void ScaledView::adjustBoundaries()
{
    QPointF empty(0.f, 0.f);
    empty = scaler.map(empty);

    QPointF leftbound = empty;
    leftbound.setX(0.f);
    QPointF lb = scaler.map(leftbound);

    QPointF rightbound = empty;
    rightbound.setX(width);
    QPointF rb = scalerI.map(rightbound);

//    QPointF endpoint = empty;
//    endpoint.setX(width);
//    endpoint = scalerI.map(endpoint);
//    qDebug() <<"ENDPOINT" << endpoint;

    QPointF bottombound = empty;
    bottombound.setY(height);
    QPointF bb = scalerI.map(bottombound);

    QPointF topbound = empty;
    topbound.setY(0);
    QPointF tb = scalerI.map(topbound);

    qreal lbpos = 20;
    qreal rbpos = pixmap.width() + 10/zoom + 1;
    qreal bbpos = pixmap.height() + 10/zoom +1;
    qreal tbpos = -10/zoom -1;

    QPointF end(width, 0);
    QPointF test = scalerI.map(end);
    qDebug() << "END" << test;
    qDebug() << "LEFT BOUND " << lb
             << "RIGHT BOUND " << rb
             << "BOTTOM BOUND " << bb
             << "TOP BOUND " << tb;


    qreal xp = 0;
    qreal yp = 0;


    if(lb.x() > lbpos && rb.x() > rbpos)
    {
        QPointF pixcenter = empty;
        pixcenter.setX(pixmap.width()/2.f);
       // pixcenter = scalerI.map(pixcenter);

        qDebug() << "WIDTH" << width;
        qDebug() << "PIX WIDTH" << pixmap.width();
        qDebug() << "PIXCENTER" << pixcenter;

        QPointF center(width/2.f+ lbpos/2.f, 0);
        center = scalerI.map(center);

        qDebug() << "CENTER " << center;

        xp = center.x() - pixcenter.x();


        qDebug() << "ALIGNING TO CENTER!!!!!!";
//        //QPointF rb = modelview.map(rightbound);



    }
    else if(lb.x() > lbpos)
    {
        qDebug() << "LEFT BOUND IS VISIBLE!";
        QPointF topleft(lbpos, 0.f);
        topleft = scalerI.map(topleft);

        xp = topleft.x();

    }
    else if(rb.x() > rbpos)
    {
        qDebug() << "RIGHT BOUND IS VISIBLE!";

        QPointF right(rbpos, 0.f);
        right = scaler.map(right);
        rb = scaler.map(rb);

        xp = rb.x()-right.x();
    }



    if(bb.y() > bbpos)
    {
        qDebug() << "BOTTOM BOUND IS VISIBLE!";

        QPointF bottom(0.f, bbpos);
        bottom = scaler.map(bottom);
        bb = scaler.map(bb);

        yp = bb.y()-bottom.y();

    }
    else if(tb.y() < tbpos)
    {
        qDebug() << "TOP BOUND IS VISIBLE!";

        QPointF top(0, tbpos);
        top = scaler.map(top);
        tb = scaler.map(tb);

        yp = tb.y()-top.y();


    }

    qDebug() << "xcor" << xp << "ycorr" << yp;
    scaler.translate(xp, yp);
    scalerI = scaler.inverted();
}
