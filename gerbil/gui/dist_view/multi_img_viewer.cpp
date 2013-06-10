/*
	Copyright(c) 2012 Johannes Jordan <johannes.jordan@cs.fau.de>.
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "multi_img_viewer.h"
#include "../tasks/distviewbinstbb.h"
#include "viewer_tasks.h"

#include "../mainwindow.h"
#include "../gerbil_gui_debug.h"

#include <background_task_queue.h>
#include <stopwatch.h>

#include <opencv2/core/core.hpp>
#include <iostream>
#include <QThread>

using namespace std;

multi_img_viewer::multi_img_viewer(QWidget *parent)
	: QWidget(parent), queue(NULL),
	  ignoreLabels(false)
{
	setupUi(this);

	// create target widget that is rendered into (handled by viewportGV)
	QGLWidget *target = new QGLWidget(QGLFormat(QGL::SampleBuffers), this);

	// create controller widget that will reside inside viewport
	control = new ViewportControl(this);

	// create viewport. The viewport is a GraphicsScene
	viewport = new Viewport(control, target);
	// initialize control with viewport	  limiterMenu(this)

	control->init(viewport);

	// finally attach everything to our member widget == QGraphicsView
	viewportGV->setViewport(target);
	viewportGV->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
	viewportGV->setScene(viewport);

	connect(viewport, SIGNAL(newOverlay(int)),
			this, SLOT(updateMask(int)));

	connect(topBar, SIGNAL(toggleFold()),
			this, SLOT(toggleFold()));
}

multi_img_viewer::~multi_img_viewer()
{
	delete viewport;
}

void multi_img_viewer::setType(representation::t type)
{
	this->type = type;
	control->setType(type);
	setTitle(type, 0.0, 0.0);
}

void multi_img_viewer::toggleFold()
{
	if (!viewportGV->isHidden()) {
		/** HIDE **/
		GGDBGM(format("viewer %1% folding")%getType() << endl);
		emit folding();
		viewportGV->setHidden(true);
		topBar->fold();
		setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
		setTitle(type, 0.0, 0.0);
		//GGDBGM(boost::format("emitting toggleViewer(false, %1%)\n") % getType());
		emit toggleViewer(false, getType());
		//GGDBGM(boost::format("past signal toggleViewer(false, %1%)\n") % getType());
		// TODO: let viewport clean itself up!
		viewport->sets.reset(new SharedData<std::vector<BinSet> >(new std::vector<BinSet>()));
		viewport->shuffleIdx.clear();
		viewport->vb.destroy();
	} else {
		/** SHOW **/
		GGDBGM(format("viewer %1% unfolding")%getType() << endl);
		emit folding();
		viewportGV->setShown(true);
		topBar->unfold();
		QSizePolicy pol(QSizePolicy::Preferred, QSizePolicy::Expanding);
		pol.setVerticalStretch(1);
		setSizePolicy(pol);
		emit toggleViewer(true, getType());
	}
}

void multi_img_viewer::subPixels(const std::map<std::pair<int, int>, short> &points)
{
	std::vector<cv::Rect> sub(points.size());
	std::map<std::pair<int, int>, short>::const_iterator it;
	for (it = points.begin(); it != points.end(); ++it) {
		sub.push_back(cv::Rect(it->first.first, it->first.second, 1, 1));
	}

	SharedDataLock ctxlock(viewport->ctx->mutex);
	ViewportCtx args = **viewport->ctx;
	ctxlock.unlock();

	BackgroundTaskPtr taskSub(new DistviewBinsTbb(
		image, labels, control->labelColors, illuminant, args, viewport->ctx, viewport->sets,
		sets_ptr(new SharedData<std::vector<BinSet> >(NULL)), sub, std::vector<cv::Rect>(), cv::Mat1b(), true, false));
	queue->push(taskSub);
	taskSub->wait();
	// TODO: why do we only update view after adding, but not after removing?
}

void multi_img_viewer::addPixels(const std::map<std::pair<int, int>, short> &points)
{
	std::vector<cv::Rect> add(points.size());
	std::map<std::pair<int, int>, short>::const_iterator it;
	for (it = points.begin(); it != points.end(); ++it) {
		add.push_back(cv::Rect(it->first.first, it->first.second, 1, 1));
	}

	SharedDataLock ctxlock(viewport->ctx->mutex);
	ViewportCtx args = **viewport->ctx;
	ctxlock.unlock();

	BackgroundTaskPtr taskAdd(new DistviewBinsTbb(
		image, labels, control->labelColors, illuminant, args, viewport->ctx, viewport->sets,
		sets_ptr(new SharedData<std::vector<BinSet> >(NULL)), std::vector<cv::Rect>(), add, cv::Mat1b(), true, false));
	queue->push(taskAdd);
	binningUpdate(taskAdd->wait());
}

void multi_img_viewer::subImage(sets_ptr temp,
								const std::vector<cv::Rect> &regions,
								cv::Rect roi)
{
	SharedDataLock ctxlock(viewport->ctx->mutex);
	ViewportCtx args = **viewport->ctx;
	ctxlock.unlock();

	BackgroundTaskPtr taskBins(new DistviewBinsTbb(
		image, labels, control->labelColors, illuminant, args, viewport->ctx,
		viewport->sets,	temp, regions,
		std::vector<cv::Rect>(), cv::Mat1b(), false, false, roi));
	queue->push(taskBins);
}

void multi_img_viewer::addImage(sets_ptr temp,
								const std::vector<cv::Rect> &regions,
								cv::Rect roi)
{
	SharedDataLock ctxlock(viewport->ctx->mutex);
	ViewportCtx args = **viewport->ctx;
	ctxlock.unlock();

	args.labelsValid = false;
	args.metaValid = false;
	args.minvalValid = false;
	args.maxvalValid = false;
	args.binsizeValid = false;

	args.reset.fetch_and_store(1);
	args.wait.fetch_and_store(1);

	BackgroundTaskPtr taskBins(new DistviewBinsTbb(
		image, labels, control->labelColors, illuminant, args, viewport->ctx,
		viewport->sets, temp, std::vector<cv::Rect>(), regions,
		cv::Mat1b(), false, true, roi));
	// connect to binningRangeUpdate as this operation can change binning range
	QObject::connect(taskBins.get(), SIGNAL(finished(bool)),
					 this, SLOT(binningRangeUpdate(bool)));
	queue->push(taskBins);
}

void multi_img_viewer::setTitle(representation::t type, multi_img::Value min, multi_img::Value max)
{
	QString title;
	if (type == representation::IMG)
		title = QString("<b>Image Spectrum</b> [%1..%2]");
	if (type == representation::GRAD)
		title = QString("<b>Spectral Gradient Spectrum</b> [%1..%2]");
	if (type == representation::IMGPCA)
		title = QString("<b>Image PCA</b> [%1..%2]");
	if (type == representation::GRADPCA)
		title = QString("<b>Spectral Gradient PCA</b> [%1..%2]");

	topBar->setTitle(title.arg(min).arg(max));
}

void multi_img_viewer::setImage(SharedMultiImgPtr img, cv::Rect roi)
{
	image = img;
	//GGDBGM(format("image.get()=%1%\n") %image.get());

	SharedDataLock ctxlock(viewport->ctx->mutex);
	ViewportCtx args = **viewport->ctx;
	ctxlock.unlock();

	args.type = type;
	args.ignoreLabels = ignoreLabels;

	args.nbins = control->getBinCount();

	args.dimensionalityValid = false;
	args.labelsValid = false;
	args.metaValid = false;
	args.minvalValid = false;
	args.maxvalValid = false;
	args.binsizeValid = false;

	args.reset.fetch_and_store(1);
	args.wait.fetch_and_store(1);

	assert(viewport->ctx);
	BackgroundTaskPtr taskBins(new DistviewBinsTbb(
		image, labels, control->labelColors, illuminant, args, viewport->ctx, viewport->sets,
		sets_ptr(new SharedData<std::vector<BinSet> >(NULL)), std::vector<cv::Rect>(), 
		std::vector<cv::Rect>(), cv::Mat1b(), false, true, roi));
	// connect to binningRangeUpdate, new image may have new range
	QObject::connect(taskBins.get(), SIGNAL(finished(bool)),
					 this, SLOT(binningRangeUpdate(bool)));
	queue->push(taskBins);
}

void multi_img_viewer::setIlluminant(cv::Mat1f illum)
{
	{
		SharedDataLock ctxlock(viewport->ctx->mutex);

	if ((*viewport->ctx)->type != representation::IMG)
		return;
	}

	// TODO: the whole code below this line should be done by the viewport itself!
	viewport->illuminant = illum;

	// TODO: update after updateTextures() or even only if that one is not called?
	viewport->update();

	// vertices need to change, but TODO: should this not call prepareLines() also?
	if(!viewport->illuminant_correction) {
		viewport->updateTextures();
	}
}

void multi_img_viewer::setIlluminantIsApplied(bool applied)
{
	// only set it to true, never to false again
	if(applied) {
		viewport->illuminant_correction = true;
	}
}


void multi_img_viewer::showIlluminationCurve(bool show)
{
	if (show) {
		// HACK
		// nothing, illuminant is always drawn if coefficients viewport->illuminant
		// non-empty. Should really be a bool in viewport.
	} else {
		cv::Mat1f empty;
		setIlluminant(empty);
	}
}

void multi_img_viewer::changeBinCount(int bins)
{
	queue->cancelTasks();

	setGUIEnabled(false, TT_BIN_COUNT);
	viewportGV->setEnabled(false);

	updateBinning(bins);

	BackgroundTaskPtr taskEpilog(new BackgroundTask());
	QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)), 
		this, SLOT(finishBinCountChange(bool)), Qt::QueuedConnection);
	queue->push(taskEpilog);
}

void multi_img_viewer::updateBinning(int bins)
{
	SharedDataLock ctxlock(viewport->ctx->mutex);
	ViewportCtx args = **viewport->ctx;
	ctxlock.unlock();

	if (bins > 0) {
		args.nbins = bins;
		control->setBinCount(bins);
	}

	args.minvalValid = false;
	args.maxvalValid = false;
	args.binsizeValid = false;

	args.reset.fetch_and_store(1);
	args.wait.fetch_and_store(1);

	if (!image.get())
		return;

	BackgroundTaskPtr taskBins(new DistviewBinsTbb(
		image, labels, control->labelColors, illuminant, args, viewport->ctx, viewport->sets));
	QObject::connect(taskBins.get(), SIGNAL(finished(bool)),
					 this, SLOT(binningUpdate(bool)), Qt::QueuedConnection);
	queue->push(taskBins);
}

void multi_img_viewer::finishBinCountChange(bool success)
{
	if (success) {
		viewportGV->setEnabled(true);
		emit finishTask(success);
	}
}

void multi_img_viewer::updateLabels(const cv::Mat1s &newLabels,
									const QVector<QColor> &colors,
									bool colorsChanged)
{
	if (!colors.empty())
		control->updateLabelColors(colors);

	// binset update is needed if labels or colors changed
	if (newLabels.empty() && (!colorsChanged))
		return;

	if (!newLabels.empty())
		labels = newLabels.clone();

	// check if we are ready to compute anything
	// TODO: also exit when in roi change. how to propagate this properly?
	if (!image.get() || labels.empty())
		return;

	SharedDataLock ctxlock(viewport->ctx->mutex);
	ViewportCtx args = **viewport->ctx;
	ctxlock.unlock();

	args.wait.fetch_and_store(1);

	BackgroundTaskPtr taskBins(new DistviewBinsTbb(
		image, labels, control->labelColors, illuminant, args, viewport->ctx,
								   viewport->sets));
	QObject::connect(taskBins.get(), SIGNAL(finished(bool)),
					 this, SLOT(binningUpdate(bool)), Qt::QueuedConnection);
	queue->push(taskBins);
}

void multi_img_viewer::updateLabelsPartially(const cv::Mat1s &newLabels,
											 const cv::Mat1b &mask)
{
	if (!image.get())
		return;

	// save old configuration for partial updates
	cv::Mat1s oldLabels = labels.clone();
	// just override the whole thing
	labels = newLabels.clone();

	SharedDataLock ctxlock(viewport->ctx->mutex);
	ViewportCtx args = **viewport->ctx;
	ctxlock.unlock();

	args.wait.fetch_and_store(1);

	// we calculate into temp, then from temp in the second round
	sets_ptr temp(new SharedData<std::vector<BinSet> >(NULL));

	{	// first round: delete all pixels from their *previous* labels
		std::vector<cv::Rect> sub;
		sub.push_back(cv::Rect(0, 0, mask.cols, mask.rows));
		BackgroundTaskPtr taskBins(new DistviewBinsTbb(
			image, oldLabels, control->labelColors, illuminant, args,
			viewport->ctx, viewport->sets, temp, sub, std::vector<cv::Rect>(),
			mask, false, false));
		queue->push(taskBins);
	}

	{	// second round: now add them back according to their current labels
		/* we do not clone labels, as labels is never changed, only replaced
		 * TODO: write that down at the appropriate place so people will
		 * adhere to that */
		std::vector<cv::Rect> add;
		add.push_back(cv::Rect(0, 0, mask.cols, mask.rows));
		BackgroundTaskPtr taskBins(new DistviewBinsTbb(
			image, labels, control->labelColors, illuminant, args,
			viewport->ctx, viewport->sets, temp, std::vector<cv::Rect>(), add,
			mask, false, true));

		// final signal
		QObject::connect(taskBins.get(), SIGNAL(finished(bool)),
						 this, SLOT(binningUpdate(bool)), Qt::QueuedConnection);
		queue->push(taskBins);
	}
}

void multi_img_viewer::binningUpdate(bool updated)
{
	if (!updated || !image.get())
		return;

	viewport->rebuild();
}

void multi_img_viewer::binningRangeUpdate(bool updated)
{
	if (!updated || !image.get())
		return;

	{
		SharedDataLock imagelock(image->mutex);
		highlightMask = cv::Mat1b((*image)->height, (*image)->width, (uchar)0);

		SharedDataLock ctxlock(viewport->ctx->mutex);
		setTitle((*viewport->ctx)->type, (*viewport->ctx)->minval,
				 (*viewport->ctx)->maxval);
	}
	viewport->rebuild();

	/* we have done our binning range update, so we expect that we can now
	 * reconnect again. */
	emit finishedCalculation(type);
}

/* create mask from single-band user selection */
void multi_img_viewer::fillMaskSingle(int dim, int sel)
{
	SharedDataLock imagelock(image->mutex);
	SharedDataLock ctxlock(viewport->ctx->mutex);
	fillMaskSingleBody body(highlightMask, (**image)[dim], dim, sel,
		(*viewport->ctx)->minval, (*viewport->ctx)->binsize, illuminant);
	tbb::parallel_for(tbb::blocked_range2d<size_t>(
		0, highlightMask.rows, 0, highlightMask.cols), body);
}

/* create mask from multi-band range selection */
void multi_img_viewer::fillMaskLimiters(const std::vector<std::pair<int, int> >& l)
{
	SharedDataLock imagelock(image->mutex);
	SharedDataLock ctxlock(viewport->ctx->mutex);
	fillMaskLimitersBody body(highlightMask, **image, (*viewport->ctx)->minval,
		(*viewport->ctx)->binsize, illuminant, l);
	tbb::parallel_for(tbb::blocked_range2d<size_t>(
		0,(*image)->height, 0, (*image)->width), body);
}

/* update mask according to user change in one band */
void multi_img_viewer::updateMaskLimiters(
		const std::vector<std::pair<int, int> >& l, int dim)
{
	SharedDataLock imagelock(image->mutex);
	SharedDataLock ctxlock(viewport->ctx->mutex);
	updateMaskLimitersBody body(highlightMask, **image, dim, (*viewport->ctx)->minval,
		(*viewport->ctx)->binsize, illuminant, l);
	tbb::parallel_for(tbb::blocked_range2d<size_t>(
		0,(*image)->height, 0, (*image)->width), body);
}

void multi_img_viewer::updateMask(int dim)
{
	if (viewport->limiterMode) {
		if (dim > -1)	// incremental update in one dimension
			updateMaskLimiters(viewport->limiters, dim);
		else	// we need to build from a label/etc.
			fillMaskLimiters(viewport->limiters);
	} else {
		fillMaskSingle(viewport->selection, viewport->hover);
	}
}

void multi_img_viewer::overlay(int x, int y)
{
	if (x < 0) {
		viewport->overlayMode = false;
		if (viewportGV->isVisible())
			viewport->update();
		return;
	}

	//GGDBGM(format("multi_img_viewer::overlay(int x, int y): image.get()=%1% type=%2%\n")
	//	   % image.get() %getType());
	if(viewportGV->isHidden()) {
	//	GGDBGM(format("WARNING: slot activated for type %1% while payload hidden!") % getType() << endl);
		return;
	}

	SharedDataLock imagelock(image->mutex);
	SharedDataLock ctxlock(viewport->ctx->mutex);
	assert(image);
	const multi_img::Pixel &pixel = (**image)(y, x);
	QPolygonF &points = viewport->overlayPoints;
	points.resize((*image)->size());

	for (unsigned int d = 0; d < (*image)->size(); ++d) {
		points[d] = QPointF(d, Compute::curpos(pixel[d], d,
			(*viewport->ctx)->minval, (*viewport->ctx)->binsize, illuminant));
	}

	viewport->overlayMode = true;
	viewport->update();
}

void multi_img_viewer::toggleLabeled(bool toggle)
{
	viewport->showLabeled = toggle;
	viewport->updateTextures();
}

void multi_img_viewer::toggleUnlabeled(bool toggle)
{
	viewport->showUnlabeled = toggle;
	viewport->updateTextures();
}

void multi_img_viewer::toggleLabels(bool toggle)
{
	SharedDataLock ctxlock(viewport->ctx->mutex);
	ViewportCtx args = **viewport->ctx;
	ctxlock.unlock();

	ignoreLabels = toggle;
	args.ignoreLabels = toggle;
	args.wait.fetch_and_store(1);

	if (!image.get())
		return;

	BackgroundTaskPtr taskBins(new DistviewBinsTbb(
		image, labels, control->labelColors, illuminant, args, viewport->ctx, viewport->sets));
	QObject::connect(taskBins.get(), SIGNAL(finished(bool)),
					 this, SLOT(binningUpdate(bool)), Qt::QueuedConnection);
	queue->push(taskBins);
}

void multi_img_viewer::toggleLimiters(bool toggle)
{
	viewport->limiterMode = toggle;
	viewport->updateTextures(Viewport::RM_SKIP, Viewport::RM_STEP);
	viewportGV->repaint();
	viewport->activate();
	updateMask(-1);
	emit newOverlay();
}

void multi_img_viewer::changeEvent(QEvent *e)
{
	QWidget::changeEvent(e);
	switch (e->type()) {
	case QEvent::LanguageChange:
		retranslateUi(this);
		break;
	default:
		break;
	}
}