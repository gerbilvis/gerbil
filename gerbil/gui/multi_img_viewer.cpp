/*
	Copyright(c) 2012 Johannes Jordan <johannes.jordan@cs.fau.de>.
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "multi_img_viewer.h"
#include "viewerwindow.h"
#include <stopwatch.h>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <QThread>
#include <background_task_queue.h>

#define REUSE_THRESHOLD 0.1

using namespace std;

multi_img_viewer::multi_img_viewer(QWidget *parent)
	: QWidget(parent), queue(NULL),
	  ignoreLabels(false),
	  limiterMenu(this)
{
	setupUi(this);

	connect(binSlider, SIGNAL(valueChanged(int)),
			this, SLOT(changeBinCount(int)));
	connect(alphaSlider, SIGNAL(valueChanged(int)),
			this, SLOT(setAlpha(int)));
	connect(alphaSlider, SIGNAL(sliderPressed()),
			viewport, SLOT(startNoHQ()));
	connect(alphaSlider, SIGNAL(sliderReleased()),
			viewport, SLOT(endNoHQ()));
	connect(limiterButton, SIGNAL(toggled(bool)),
			this, SLOT(toggleLimiters(bool)));
	connect(limiterMenuButton, SIGNAL(clicked()),
			this, SLOT(showLimiterMenu()));

	connect(rgbButton, SIGNAL(toggled(bool)),
			viewport, SLOT(toggleRGB(bool)));

	connect(viewport, SIGNAL(newOverlay(int)),
			this, SLOT(updateMask(int)));

	setAlpha(alphaSlider->value());

	connect(topBar, SIGNAL(toggleFold()),
			this, SLOT(toggleFold()));
}

void multi_img_viewer::setType(representation type)
{
	this->type = type;
	if (type != IMG)
		rgbButton->setVisible(false);
	setTitle(type, 0.0, 0.0);
}

void multi_img_viewer::toggleFold()
{
	if (!payload->isHidden()) {
		emit folding();
		payload->setHidden(true);
		topBar->fold();
		setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
		setTitle(type, 0.0, 0.0);
		emit toggleViewer(false, getType());
		viewport->sets.reset(new SharedData<std::vector<BinSet> >(new std::vector<BinSet>()));
		viewport->shuffleIdx.clear();
		viewport->vb.destroy();
	} else {
		emit folding();
		payload->setShown(true);
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

	SharedDataLock ctxlock(viewport->ctx->lock);
	ViewportCtx args = **viewport->ctx;
	ctxlock.unlock();

	BackgroundTaskPtr taskSub(new BinsTbb(
		image, labels, labelColors, illuminant, args, viewport->ctx, viewport->sets,
		sets_ptr(new SharedData<std::vector<BinSet> >(NULL)), sub, std::vector<cv::Rect>(), cv::Mat1b(), true, false));
	queue->push(taskSub);
	taskSub->wait();
}

void multi_img_viewer::addPixels(const std::map<std::pair<int, int>, short> &points)
{
	std::vector<cv::Rect> add(points.size());
	std::map<std::pair<int, int>, short>::const_iterator it;
	for (it = points.begin(); it != points.end(); ++it) {
		add.push_back(cv::Rect(it->first.first, it->first.second, 1, 1));
	}

	SharedDataLock ctxlock(viewport->ctx->lock);
	ViewportCtx args = **viewport->ctx;
	ctxlock.unlock();

	BackgroundTaskPtr taskAdd(new BinsTbb(
		image, labels, labelColors, illuminant, args, viewport->ctx, viewport->sets,
		sets_ptr(new SharedData<std::vector<BinSet> >(NULL)), std::vector<cv::Rect>(), add, cv::Mat1b(), true, false));
	queue->push(taskAdd);
	render(taskAdd->wait());
}

void multi_img_viewer::subImage(sets_ptr temp, const std::vector<cv::Rect> &regions, cv::Rect roi)
{
	SharedDataLock ctxlock(viewport->ctx->lock);
	ViewportCtx args = **viewport->ctx;
	ctxlock.unlock();

	BackgroundTaskPtr taskBins(new BinsTbb(
		image, labels, labelColors, illuminant, args, viewport->ctx, viewport->sets,
		temp, regions, std::vector<cv::Rect>(), cv::Mat1b(), false, false, roi));
	queue->push(taskBins);
}

void multi_img_viewer::setTitle(representation type, multi_img::Value min, multi_img::Value max)
{
	QString title;
	if (type == IMG)
		title = QString("<b>Image Spectrum</b> [%1..%2]");
	if (type == GRAD)
		title = QString("<b>Spectral Gradient Spectrum</b> [%1..%2]");
	if (type == IMGPCA)
		title = QString("<b>Image PCA</b> [%1..%2]");
	if (type == GRADPCA)
		title = QString("<b>Spectral Gradient PCA</b> [%1..%2]");

	topBar->setTitle(title.arg(min).arg(max));
}

void multi_img_viewer::addImage(sets_ptr temp, const std::vector<cv::Rect> &regions, cv::Rect roi)
{
	SharedDataLock ctxlock(viewport->ctx->lock);
	ViewportCtx args = **viewport->ctx;
	ctxlock.unlock();

	maskReset = true;
	titleReset = true;

	args.labelsValid = false;
	args.metaValid = false;
	args.minvalValid = false;
	args.maxvalValid = false;
	args.binsizeValid = false;

	args.reset.fetch_and_store(1);
	args.wait.fetch_and_store(1);

	BackgroundTaskPtr taskBins(new BinsTbb(
		image, labels, labelColors, illuminant, args, viewport->ctx, viewport->sets,
		temp, std::vector<cv::Rect>(), regions, cv::Mat1b(), false, true, roi));
	QObject::connect(taskBins.get(), SIGNAL(finished(bool)), this, SLOT(render(bool)));
	queue->push(taskBins);
}

void multi_img_viewer::setImage(multi_img_ptr img, cv::Rect roi)
{
	SharedDataLock ctxlock(viewport->ctx->lock);
	ViewportCtx args = **viewport->ctx;
	ctxlock.unlock();

	image = img;

	args.type = type;
	args.ignoreLabels = ignoreLabels;

	int bins = binSlider->value();
	if (bins > 0) {
		args.nbins = bins;
		binLabel->setText(QString("%1 bins").arg(bins));
	}

	maskReset = true;
	titleReset = true;

	args.dimensionalityValid = false;
	args.labelsValid = false;
	args.metaValid = false;
	args.minvalValid = false;
	args.maxvalValid = false;
	args.binsizeValid = false;

	args.reset.fetch_and_store(1);
	args.wait.fetch_and_store(1);

	BackgroundTaskPtr taskBins(new BinsTbb(
		image, labels, labelColors, illuminant, args, viewport->ctx, viewport->sets,
		sets_ptr(new SharedData<std::vector<BinSet> >(NULL)), std::vector<cv::Rect>(), 
		std::vector<cv::Rect>(), cv::Mat1b(), false, true, roi));
	QObject::connect(taskBins.get(), SIGNAL(finished(bool)), this, SLOT(render(bool)));
	queue->push(taskBins);
}

void multi_img_viewer::setIlluminant(
		const std::vector<multi_img::Value> &coeffs, bool for_real)
{
	SharedDataLock ctxlock(viewport->ctx->lock);

	if ((*viewport->ctx)->type != IMG)
		return;

	ctxlock.unlock();

	if (for_real) {
		// only set it to true, never to false again
		viewport->illuminant_correction = true;
		illuminant = coeffs;
	} else {
		viewport->illuminant = coeffs;
		viewport->updateTextures();
	}
}

void multi_img_viewer::changeBinCount(int bins)
{
	queue->cancelTasks();

	setGUIEnabled(false, TT_BIN_COUNT);
	binSlider->setEnabled(true);
	alphaSlider->setEnabled(false);
	limiterButton->setEnabled(false);
	limiterMenuButton->setEnabled(false);
	rgbButton->setEnabled(false);
	viewport->setEnabled(false);

	updateBinning(bins);

	BackgroundTaskPtr taskEpilog(new BackgroundTask());
	QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)), 
		this, SLOT(finishBinCountChange(bool)), Qt::QueuedConnection);
	queue->push(taskEpilog);
}

void multi_img_viewer::updateBinning(int bins)
{
	SharedDataLock ctxlock(viewport->ctx->lock);
	ViewportCtx args = **viewport->ctx;
	ctxlock.unlock();

	if (bins > 0) {
		args.nbins = bins;
		binLabel->setText(QString("%1 bins").arg(bins));
	}

	args.minvalValid = false;
	args.maxvalValid = false;
	args.binsizeValid = false;

	args.reset.fetch_and_store(1);
	args.wait.fetch_and_store(1);

	if (!image.get())
		return;

	BackgroundTaskPtr taskBins(new BinsTbb(
		image, labels, labelColors, illuminant, args, viewport->ctx, viewport->sets));
	QObject::connect(taskBins.get(), SIGNAL(finished(bool)), this, SLOT(render(bool)), Qt::QueuedConnection);
	queue->push(taskBins);
}

void multi_img_viewer::finishBinCountChange(bool success)
{
	if (success) {
		viewport->setEnabled(true);
		rgbButton->setEnabled(true);
		limiterMenuButton->setEnabled(true);
		limiterButton->setEnabled(true);
		alphaSlider->setEnabled(true);
		emit finishTask(success);
	}
}

void multi_img_viewer::subLabelMask(sets_ptr temp, const cv::Mat1b &mask)
{
	SharedDataLock ctxlock(viewport->ctx->lock);
	ViewportCtx args = **viewport->ctx;
	ctxlock.unlock();

	args.wait.fetch_and_store(1);

	if (!image.get())
		return;

	std::vector<cv::Rect> sub;
	sub.push_back(cv::Rect(0, 0, mask.cols, mask.rows));
	BackgroundTaskPtr taskBins(new BinsTbb(
		image, labels.clone(), labelColors, illuminant, args, viewport->ctx, viewport->sets,
		temp, sub, std::vector<cv::Rect>(), mask, false, false));
	queue->push(taskBins);
}

void multi_img_viewer::addLabelMask(sets_ptr temp, const cv::Mat1b &mask)
{
	SharedDataLock ctxlock(viewport->ctx->lock);
	ViewportCtx args = **viewport->ctx;
	ctxlock.unlock();

	args.wait.fetch_and_store(1);

	if (!image.get())
		return;

	std::vector<cv::Rect> add;
	add.push_back(cv::Rect(0, 0, mask.cols, mask.rows));
	BackgroundTaskPtr taskBins(new BinsTbb(
		image, labels.clone(), labelColors, illuminant, args, viewport->ctx, viewport->sets,
		temp, std::vector<cv::Rect>(), add, mask, false, true));
	QObject::connect(taskBins.get(), SIGNAL(finished(bool)), this, SLOT(render(bool)), Qt::QueuedConnection);
	queue->push(taskBins);
}

void multi_img_viewer::updateLabels()
{
	SharedDataLock ctxlock(viewport->ctx->lock);
	ViewportCtx args = **viewport->ctx;
	ctxlock.unlock();

	args.wait.fetch_and_store(1);

	if (!image.get())
		return;

	BackgroundTaskPtr taskBins(new BinsTbb(
		image, labels, labelColors, illuminant, args, viewport->ctx, viewport->sets));
	QObject::connect(taskBins.get(), SIGNAL(finished(bool)), this, SLOT(render(bool)), Qt::QueuedConnection);
	queue->push(taskBins);
}

void multi_img_viewer::render(bool necessary)
{
	if (necessary && image.get()) {
		if (maskReset) {
			SharedDataLock imagelock(image->lock);
			maskholder = multi_img::Mask((*image)->height, (*image)->width, (uchar)0);
			maskReset = false;
		}
		if (titleReset) {
			SharedDataLock ctxlock(viewport->ctx->lock);
			setTitle((*viewport->ctx)->type, (*viewport->ctx)->minval, (*viewport->ctx)->maxval);
			titleReset = false;
		}
		viewport->rebuild();
	}
}

bool multi_img_viewer::BinsTbb::run() 
{
	bool reuse = ((!add.empty() || !sub.empty()) && !inplace);
	bool keepOldContext = false;
	if (reuse) {
		keepOldContext = ((fabs(args.minval) * REUSE_THRESHOLD) >= (fabs(args.minval - (*multi)->minval))) &&
			((fabs(args.maxval) * REUSE_THRESHOLD) >= (fabs(args.maxval - (*multi)->maxval)));
		if (!keepOldContext) {
			reuse = false;
			add.clear();
			sub.clear();
		}
	}

	std::vector<BinSet> *result;
	SharedDataSwapLock current_lock(current->lock, boost::defer_lock_t());
	if (reuse) {
		SharedDataSwapLock temp_wlock(temp->lock);
		result = temp->swap(NULL);
		if (!result) {
			result = new std::vector<BinSet>(**current);
		} else {
			for (int i = result->size(); i < colors.size(); ++i) {
				result->push_back(BinSet(colors[i], (*multi)->size()));
			}
		}
	} else if (inplace) {
		current_lock.lock();
		result = &(**current);
		for (int i = result->size(); i < colors.size(); ++i) {
			result->push_back(BinSet(colors[i], (*multi)->size()));
		}
	} else {
		result = new std::vector<BinSet>();
		for (int i = 0; i < colors.size(); ++i) {
			result->push_back(BinSet(colors[i], (*multi)->size()));
		}
		add.push_back(cv::Rect(0, 0, (*multi)->width, (*multi)->height));
	}

	if (!args.dimensionalityValid) {
		if (!keepOldContext)
			args.dimensionality = (*multi)->size();
		args.dimensionalityValid = true;
	}

	if (!args.metaValid) {
		if (!keepOldContext)
			args.meta = (*multi)->meta;
		args.metaValid = true;
	}

	if (!args.labelsValid) {
		if (!keepOldContext) {
			args.labels.resize((*multi)->size());
			for (unsigned int i = 0; i < (*multi)->size(); ++i) {
				if (!(*multi)->meta[i].empty)
					args.labels[i].setNum((*multi)->meta[i].center);
			}
		}
		args.labelsValid = true;
	}

	if (!args.binsizeValid) {
		if (!keepOldContext)
			args.binsize = ((*multi)->maxval - (*multi)->minval) / (multi_img::Value)(args.nbins - 1);
		args.binsizeValid = true;
	}

	if (!args.minvalValid) {
		if (!keepOldContext)
			args.minval = (*multi)->minval;
		args.minvalValid = true;
	}

	if (!args.maxvalValid) {
		if (!keepOldContext)
			args.maxval = (*multi)->maxval;
		args.maxvalValid = true;
	}

	std::vector<cv::Rect>::iterator it;
	for (it = sub.begin(); it != sub.end(); ++it) {
		Accumulate substract(
			true, **multi, labels, mask, args.nbins, args.binsize, args.minval, args.ignoreLabels, illuminant, *result);
		tbb::parallel_for(
			tbb::blocked_range2d<int>(it->y, it->y + it->height, it->x, it->x + it->width), 
				substract, tbb::auto_partitioner(), stopper);
	}
	for (it = add.begin(); it != add.end(); ++it) {
		Accumulate add(
			false, **multi, labels, mask, args.nbins, args.binsize, args.minval, args.ignoreLabels, illuminant, *result);
		tbb::parallel_for(
			tbb::blocked_range2d<int>(it->y, it->y + it->height, it->x, it->x + it->width), 
				add, tbb::auto_partitioner(), stopper);
	}

	if (stopper.is_group_execution_cancelled()) {
		delete result;
		return false;
	} else {
		if (reuse && !apply) {
			SharedDataSwapLock temp_wlock(temp->lock);
			temp->swap(result);
		} else if (inplace) {
			current_lock.unlock();
		} else {
			SharedDataSwapLock context_wlock(context->lock);
			SharedDataSwapLock current_wlock(current->lock);
			**context = args;
			delete current->swap(result);
		}
		return true;
	}
}

void multi_img_viewer::BinsTbb::Accumulate::operator()(const tbb::blocked_range2d<int> &r) const
{
	for (int y = r.rows().begin(); y != r.rows().end(); ++y) {
		short *lr = labels[y];
		for (int x = r.cols().begin(); x != r.cols().end(); ++x) {
			if (!mask.empty() && !mask(y, x))
				continue;

			int label = (ignoreLabels ? 0 : lr[x]);
			label = (label >= sets.size()) ? 0 : label;
			const multi_img::Pixel& pixel = multi(y, x);
			BinSet &s = sets[label];

			BinSet::HashKey hashkey(boost::extents[multi.size()]);
			for (int d = 0; d < multi.size(); ++d) {
				int pos = floor(curpos(pixel[d], d, minval, binsize, illuminant));
				pos = max(pos, 0); pos = min(pos, nbins-1);
				hashkey[d] = (unsigned char)pos;
			}

			if (substract) {
				BinSet::HashMap::accessor ac;
				if (s.bins.find(ac, hashkey)) {
					ac->second.sub(pixel);
					if (ac->second.weight == 0.f)
						s.bins.erase(ac);
				}
				ac.release();
				s.totalweight--; // atomic
			} else {
				BinSet::HashMap::accessor ac;
				s.bins.insert(ac, hashkey);
				ac->second.add(pixel);
				ac.release();
				s.totalweight++; // atomic
			}
		}
	}
}

struct fillMaskSingleBody {
	multi_img::Mask &mask;
	const multi_img::Band &band;
	int dim; 
	int sel;
	multi_img::Value minval;
	multi_img::Value binsize;
	const std::vector<multi_img::Value> &illuminant;

	fillMaskSingleBody(multi_img::Mask &mask, const multi_img::Band &band, int dim, int sel,
		multi_img::Value minval, multi_img::Value binsize, 
		const std::vector<multi_img::Value> &illuminant)
		: mask(mask), band(band), dim(dim), sel(sel), minval(minval), 
		binsize(binsize), illuminant(illuminant) {}

	void operator()(const tbb::blocked_range2d<size_t> &r) const {
		for (int y = r.rows().begin(); y != r.rows().end(); ++y) {
			unsigned char *mrow = mask[y];
			const multi_img::Value *brow = band[y];
			for (int x = r.cols().begin(); x != r.cols().end(); ++x) {
				int pos = floor(multi_img_viewer::curpos(brow[x], dim, minval, binsize, illuminant));
					mrow[x] = (pos == sel) ? 1 : 0;
			}
		}
	}
};

/* create mask of single-band user selection */
void multi_img_viewer::fillMaskSingle(int dim, int sel)
{
	SharedDataLock imagelock(image->lock);
	SharedDataLock ctxlock(viewport->ctx->lock);
	fillMaskSingleBody body(maskholder, (**image)[dim], dim, sel, 
		(*viewport->ctx)->minval, (*viewport->ctx)->binsize, illuminant);
	tbb::parallel_for(tbb::blocked_range2d<size_t>(
		0, maskholder.rows, 0, maskholder.cols), body);
}

struct fillMaskLimitersBody {
	multi_img::Mask &mask;
	const multi_img &image;
	multi_img::Value minval;
	multi_img::Value binsize;
	const std::vector<multi_img::Value> &illuminant;
	const std::vector<std::pair<int, int> > &l;

	fillMaskLimitersBody(multi_img::Mask &mask, const multi_img &image,
		multi_img::Value minval, multi_img::Value binsize, 
		const std::vector<multi_img::Value> &illuminant,
		const std::vector<std::pair<int, int> > &l)
		: mask(mask), image(image), minval(minval), 
		binsize(binsize), illuminant(illuminant), l(l) {}

	void operator()(const tbb::blocked_range2d<size_t> &r) const {
		for (int y = r.rows().begin(); y != r.rows().end(); ++y) {
			unsigned char *row = mask[y];
			for (int x = r.cols().begin(); x != r.cols().end(); ++x) {
				row[x] = 1;
				const multi_img::Pixel &p = image(y, x);
				for (unsigned int d = 0; d < image.size(); ++d) {
					int pos = floor(multi_img_viewer::curpos(p[d], d, minval, binsize, illuminant));
					if (pos < l[d].first || pos > l[d].second) {
						row[x] = 0;
						break;
					}
				}
			}
		}
	}
};

void multi_img_viewer::fillMaskLimiters(const std::vector<std::pair<int, int> >& l)
{
	SharedDataLock imagelock(image->lock);
	SharedDataLock ctxlock(viewport->ctx->lock);
	fillMaskLimitersBody body(maskholder, **image, (*viewport->ctx)->minval, 
		(*viewport->ctx)->binsize, illuminant, l);
	tbb::parallel_for(tbb::blocked_range2d<size_t>(
		0,(*image)->height, 0, (*image)->width), body);
}

struct updateMaskLimitersBody {
	multi_img::Mask &mask;
	const multi_img &image;
	int dim;
	multi_img::Value minval;
	multi_img::Value binsize;
	const std::vector<multi_img::Value> &illuminant;
	const std::vector<std::pair<int, int> > &l;

	updateMaskLimitersBody(multi_img::Mask &mask, const multi_img &image, int dim,
		multi_img::Value minval, multi_img::Value binsize, 
		const std::vector<multi_img::Value> &illuminant,
		const std::vector<std::pair<int, int> > &l)
		: mask(mask), image(image), dim(dim), minval(minval), 
		binsize(binsize), illuminant(illuminant), l(l) {}

	void operator()(const tbb::blocked_range2d<size_t> &r) const {
		for (int y = r.rows().begin(); y != r.rows().end(); ++y) {
			unsigned char *mrow = mask[y];
			const multi_img::Value *brow = image[dim][y];
			for (int x = r.cols().begin(); x != r.cols().end(); ++x) {
				int pos = floor(multi_img_viewer::curpos(brow[x], dim, minval, binsize, illuminant));
				if (pos < l[dim].first || pos > l[dim].second) {
					mrow[x] = 0;
				} else if (mrow[x] == 0) { // we need to do exhaustive test
					mrow[x] = 1;
					const multi_img::Pixel& p = image(y, x);
					for (unsigned int d = 0; d < image.size(); ++d) {
						int pos = floor(multi_img_viewer::curpos(p[d], d, minval, binsize, illuminant));
						if (pos < l[d].first || pos > l[d].second) {
							mrow[x] = 0;
							break;
						}
					}
				}
			}
		}
	}
};

void multi_img_viewer::updateMaskLimiters(
		const std::vector<std::pair<int, int> >& l, int dim)
{
	SharedDataLock imagelock(image->lock);
	SharedDataLock ctxlock(viewport->ctx->lock);
	updateMaskLimitersBody body(maskholder, **image, dim, (*viewport->ctx)->minval, 
		(*viewport->ctx)->binsize, illuminant, l);
	tbb::parallel_for(tbb::blocked_range2d<size_t>(
		0,(*image)->height, 0, (*image)->width), body);
}

void multi_img_viewer::updateMask(int dim)
{
	if (viewport->limiterMode) {		
		if (maskValid && dim > -1)
			updateMaskLimiters(viewport->limiters, dim);
		else
			fillMaskLimiters(viewport->limiters);
		maskValid = true;
	} else {
		fillMaskSingle(viewport->selection, viewport->hover);
	}
}

void multi_img_viewer::overlay(int x, int y)
{
	SharedDataLock imagelock(image->lock);
	SharedDataLock ctxlock(viewport->ctx->lock);
	const multi_img::Pixel &pixel = (**image)(y, x);
	QPolygonF &points = viewport->overlayPoints;
	points.resize((*image)->size());

	for (unsigned int d = 0; d < (*image)->size(); ++d) {
		points[d] = QPointF(d, curpos(pixel[d], d, 
			(*viewport->ctx)->minval, (*viewport->ctx)->binsize, illuminant));
	}

	viewport->overlayMode = true;
	viewport->repaint();
	viewport->overlayMode = false;
}

void multi_img_viewer::setAlpha(int alpha)
{
	viewport->useralpha = (float)alpha/100.f;
	alphaLabel->setText(QString::fromUtf8("α: %1").arg(viewport->useralpha, 0, 'f', 2));
	viewport->updateTextures(Viewport::RM_STEP, Viewport::RM_SKIP);
}

void multi_img_viewer::createLimiterMenu()
{
	limiterMenu.clear();
	QAction *tmp;
	tmp = limiterMenu.addAction("No limits");
	tmp->setData(0);
	tmp = limiterMenu.addAction("Limit from current highlight");
	tmp->setData(-1);
	limiterMenu.addSeparator();
	for (int i = 1; i < labelColors.size(); ++i) {
		tmp = limiterMenu.addAction(ViewerWindow::colorIcon(labelColors[i]),
													  "Limit by label");
		tmp->setData(i);
	}
}

void multi_img_viewer::showLimiterMenu()
{
	QAction *a = limiterMenu.exec(limiterMenuButton->mapToGlobal(QPoint(0, 0)));
	if (!a)
		return;

	int choice = a->data().toInt(); assert(choice < labelColors.size());
	viewport->setLimiters(choice);
	if (!limiterButton->isChecked()) {
		limiterButton->toggle();	// change button state AND toggleLimiters()
	} else {
		toggleLimiters(true);
	}
}

void multi_img_viewer::updateLabelColors(const QVector<QColor> &colors, bool changed)
{
	SharedDataLock ctxlock(viewport->ctx->lock);
	ViewportCtx args = **viewport->ctx;
	ctxlock.unlock();

	args.wait.fetch_and_store(1);

	labelColors = colors;
	createLimiterMenu();
	if (changed) {
		if (!image.get())
			return;

		BackgroundTaskPtr taskBins(new BinsTbb(
			image, labels, labelColors, illuminant, args, viewport->ctx, viewport->sets));
		QObject::connect(taskBins.get(), SIGNAL(finished(bool)), this, SLOT(render(bool)), Qt::QueuedConnection);
		queue->push(taskBins);
	}
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
	SharedDataLock ctxlock(viewport->ctx->lock);
	ViewportCtx args = **viewport->ctx;
	ctxlock.unlock();

	ignoreLabels = toggle;
	args.ignoreLabels = toggle;
	args.wait.fetch_and_store(1);

	if (!image.get())
		return;

	BackgroundTaskPtr taskBins(new BinsTbb(
		image, labels, labelColors, illuminant, args, viewport->ctx, viewport->sets));
	QObject::connect(taskBins.get(), SIGNAL(finished(bool)), this, SLOT(render(bool)), Qt::QueuedConnection);
	queue->push(taskBins);
}

void multi_img_viewer::toggleLimiters(bool toggle)
{
	viewport->limiterMode = toggle;
	viewport->updateTextures(Viewport::RM_SKIP, Viewport::RM_STEP);
	viewport->repaint();
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
