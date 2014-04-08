#include "viewport.h"

#include <QGraphicsSceneEvent>
#include <QGraphicsProxyWidget>
#include <cmath>

bool Viewport::updateXY(int sel, int bin)
{
	SharedDataLock ctxlock(ctx->mutex);

	if (sel < 0 || sel >= (*ctx)->dimensionality)
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

	} else if (event->buttons() & Qt::RightButton) {
		/* panning movement */

		if (lasty < 0)
			return;

		shift += (event->scenePos().y() - lasty)/(qreal)height;
		lasty = event->scenePos().y(); // TODO: will be done by qgraphicsscene!

		/* TODO: make sure that we use full visible space */

		updateModelview();
		buffers[0].renderTimer.stop();
		buffers[1].renderTimer.stop();
		scrollTimer.start(10);
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

	if (event->button() == Qt::RightButton) {
		target->setCursor(Qt::ClosedHandCursor);
		lasty = event->scenePos().y(); // TODO: qgraphicsscene
	}

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

	if (event->button() == Qt::RightButton) {
		target->setCursor(Qt::ArrowCursor);
		lasty = -1;
	}

	endNoHQ((event->button() == Qt::RightButton) ? RM_STEP : RM_SKIP, RM_STEP);
}

void Viewport::wheelEvent(QGraphicsSceneWheelEvent *event)
{
	// check for scene elements first (we are technically the background)
	QGraphicsScene::wheelEvent(event);
	if (event->isAccepted())
		 return;

	// zoom in or out
	qreal oldzoom = zoom;
	if (event->delta() > 0)
		zoom *= 1.25;
	else
		zoom = std::max(zoom * 0.80, 1.);

	// adjust shift to new zoom
	shift += ((oldzoom - zoom) * 0.5);

	/* TODO: make sure that we use full space */

	updateModelview();
	updateBuffers();
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
			if (selection < (*ctx)->dimensionality-1) {
				selection++;
				emit bandSelected(selection);
				if (!limiterMode) // we do not touch the limiters
					emit requestOverlay(selection, hover);
				highlightAltered = true;
			}
		}
		break;

	case Qt::Key_Space:
		drawHQ = !drawHQ;
		if (drawHQ) {
			// triggers drawing update
			endNoHQ();
		} else {
			startNoHQ();
			// deliberately make display worse for user to see effect
			updateBuffers();
		}
		break;
	case Qt::Key_L:
		drawLog = !drawLog;
		updateBuffers();
	case Qt::Key_F:
		switch (bufferFormat) {
		case RGBA8: bufferFormat = RGBA16F; break;
		case RGBA16F: bufferFormat = RGBA32F; break;
		case RGBA32F: bufferFormat = RGBA8; break;
		}
		resizeScene(); // will init and update buffers
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
