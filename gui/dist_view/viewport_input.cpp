#include "viewport.h"

#include <QGraphicsSceneEvent>
#include <QGraphicsProxyWidget>
#include <QDebug>
#include <cmath>

bool Viewport::updateXY(int sel, int bin)
{
	SharedDataLock ctxlock(ctx->mutex);

	if (sel < 0 || sel >= (int)(*ctx)->dimensionality)
		return false;

	bool highlightChanged = false;

	/* first: handle sel -> band selection */

	if (selection != sel && !holdSelection) {
		selection = sel;
		// selection results in new band to be shown
		emit bandSelected(selection);
		// it also results in a new overlay
		emit requestOverlay(selection, hover);
		// and we changed our own highlight
		highlightChanged = true;
	}

	// do this after the first chance to change selection (above)
	if (limiterMode)
		// no accidential jumping to limiters of other bands
		holdSelection = true;

	/* second: handle bin -> intensity highlight */

	/* correct y for illuminant */
	if (!illuminantAppl.empty())
		bin = std::floor(bin / illuminantAppl.at(sel) + 0.5f);

	if (bin >= 0 && bin < (*ctx)->nbins) {
		if (!limiterMode && (hover != bin)) {
			hover = bin;
			emit requestOverlay(selection, hover);
			highlightChanged = true;
		}
		if (limiterMode && updateLimiter(selection, bin)) {
			emit requestOverlay(limiters, selection);
			highlightChanged = true;
		}
	}

	return highlightChanged;
}

void Viewport::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	// check for scene elements first (we are technically the background)
	QGraphicsScene::mouseMoveEvent(event);
	if (event->isAccepted())
		return;

	bool needUpdate = false, needTextureUpdate = false;

	/* we could have entered the window right now, need to refresh
	   in case there was a pixel highlight (just a guess!!) */
	//needUpdate = clearView;
	//clearView = false;  TODO

	if (event->buttons() & Qt::LeftButton) {
		/* cursor control */

		QPointF pos = modelviewI.map(event->scenePos());
		// add .5 to x for rounding
		needTextureUpdate = updateXY(pos.x() + 0.5f, pos.y());

	} else if (event->buttons() & Qt::RightButton && zoom  > 1) {
		/* panning movement */

		QPointF lastonscene = modelviewI.map(event->lastScenePos());
		QPointF curronscene = modelviewI.map(event->scenePos());

		//qDebug() << "curronscene" << curronscene;

		qreal xp = curronscene.x() - lastonscene.x();
		qreal yp = curronscene.y() - lastonscene.y();

		modelview.translate(xp, yp);
		modelviewI = modelview.inverted();

		adjustBoundaries();
		updateBuffers();
		updateYAxis(true);
	}

	if (needTextureUpdate) {
		// calls update()
		updateBuffers(RM_SKIP, limiterMode ? RM_STEP : RM_FULL);
	} else if (needUpdate) {
		update();
	}

	event->accept();
}

void Viewport::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	activate(); // give ourselves active role if we don't have it yet

	// check for scene elements first (we are technically the background)
	QGraphicsScene::mousePressEvent(event);
	if (event->isAccepted())
		return;

	startNoHQ();

	if (event->button() == Qt::RightButton)
		target->setCursor(Qt::ClosedHandCursor);

	mouseMoveEvent(event);
}

void Viewport::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
	// check for scene elements first (we are technically the background)
	QGraphicsScene::mouseReleaseEvent(event);
	if (event->isAccepted())
		return;

	/* in limiterMode, holdSelect+act.Limiter is set on first mouse action,
	 * so we reset them now as mouse actions are finished */
	holdSelection = false;
	activeLimiter = 0;

	if (event->button() == Qt::RightButton)
		target->setCursor(Qt::ArrowCursor);

	endNoHQ((event->button() == Qt::RightButton) ? RM_STEP : RM_SKIP, RM_STEP);
}

void Viewport::wheelEvent(QGraphicsSceneWheelEvent *event)
{
	// check for scene elements first (we are technically the background)
	QGraphicsScene::wheelEvent(event);
	if (event->isAccepted())
		return;

	qreal newzoom;
	if (event->delta() > 0) {
		newzoom = 1.25;
	} else {
		newzoom = 0.8;
	}

	if (zoom*newzoom <= 1)
	{
		if (zoom == 1) // nothing to do here
			return;
		zoom = 1;
		updateYAxis();

		updateModelview();
	} else {
		QPointF scene = event->scenePos();
		QPointF local = modelviewI.map(scene);

		zoom *= newzoom;

		modelview.scale(newzoom,newzoom);
		modelviewI = modelview.inverted();

		QPointF newlocal = modelviewI.map(scene);
		QPointF diff = newlocal - local;
		modelview.translate(diff.x(), diff.y());
		modelviewI = modelview.inverted();

		adjustBoundaries();

		updateYAxis(true);
	}

	updateBuffers();
	/* TODO: make sure that we use full space */
}

void Viewport::keyPressEvent(QKeyEvent *event)
{
	// check for scene elements first (we are technically the background)
	QGraphicsScene::keyPressEvent(event);
	if (event->isAccepted())
		return;

	bool highlightAltered = false;

	switch (event->key()) {
	case Qt::Key_S:
		screenshot();
		break;

	case Qt::Key_Plus:
		emit addSelectionRequested();
		break;
	case Qt::Key_Minus:
		emit remSelectionRequested();
		break;

	case Qt::Key_Up:
		{
			SharedDataLock ctxlock(ctx->mutex);
			if (!limiterMode && hover < (*ctx)->nbins-2) {
				hover++;
				requestOverlay(selection, hover);
				highlightAltered = true;
			}
		}
		break;
	case Qt::Key_Down:
		if (!limiterMode && hover > 0) {
			hover--;
			requestOverlay(selection, hover);
			highlightAltered = true;
		}
		break;
	case Qt::Key_Left:
		{
			SharedDataLock ctxlock(ctx->mutex);
			if (selection > 0) {
				selection--;
				emit bandSelected(selection);
				if (!limiterMode) // we do not touch the limiters
					emit requestOverlay(selection, hover);
				highlightAltered = true;
			}
		}
		break;
	case Qt::Key_Right:
		{
			SharedDataLock ctxlock(ctx->mutex);
			if (selection < (int)(*ctx)->dimensionality-1) {
				selection++;
				emit bandSelected(selection);
				if (!limiterMode) // we do not touch the limiters
					emit requestOverlay(selection, hover);
				highlightAltered = true;
			}
		}
		break;

	case Qt::Key_Space:
		toggleHQ();
		break;
	case Qt::Key_L:
		toggleDrawLog();
		break;
	case Qt::Key_F:
		switch (bufferFormat) {
		case RGBA8: bufferFormat = RGBA16F; break;
		case RGBA16F: bufferFormat = RGBA32F; break;
		case RGBA32F: bufferFormat = RGBA8; break;
		}
		// initialize buffers with new format
		initBuffers();
		updateBuffers();
		break;
	case Qt::Key_M:
		drawMeans = !drawMeans;
		rebuild();
		break;
	}

	if (highlightAltered) {
		updateBuffers(RM_SKIP, RM_FULL);
	}
}
