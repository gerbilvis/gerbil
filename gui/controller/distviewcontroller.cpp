#include "distviewcontroller.h"
#include "dist_view/viewport.h"
#include "widgets/mainwindow.h"

#include <QSignalMapper>

// for debug msgs
#include <boost/format.hpp>
#include <QDebug>

#define GGDBG_REPR(type) GGDBGM(format("%1%")%type << endl)

DistViewController::DistViewController(Controller *chief,
									   BackgroundTaskQueue *taskQueue)
 : chief(chief)
{
	/* Create all viewers. Only one viewer per representation type is supported.
	 */
	foreach (representation::t i, representation::all()) {
		map.insert(i, new payload(i));
		map[i]->model.setTaskQueue(taskQueue);
		// TODO: the other way round. ctx+sets come from model
		map[i]->model.setContext(map[i]->gui.getContext());
		map[i]->model.setBinSets(map[i]->gui.getBinSets());
	}
}

void DistViewController::init()
{
	// start with IMG, hide IMGPCA, GRADPCA at the beginning
	/* NOTE that we did not connect signals yet! This is actually good as we
	 * will not get stray signals before the content is ready.
	 */
	activeView = representation::IMG;
	map[activeView]->gui.setActive();
#ifdef WITH_IMGPCA
	map[representation::IMGPCA]->gui.toggleFold();
#endif /* WITH_IMGPCA */
#ifdef WITH_GRADPCA
	map[representation::GRADPCA]->gui.toggleFold();
#endif /* WITH_GRADPCA */


	foreach (payload *p, map) {
		// fill GUI with distribution view
		chief->mainWindow()->addDistView(p->gui.getFrame());
	}

	foreach (payload *p, map) {
		DistViewModel *m = &p->model;

		connect(m, SIGNAL(newBinning(representation::t)),
				this, SLOT(processNewBinning(representation::t)));
		connect(m, SIGNAL(newBinningRange(representation::t)),
				this, SLOT(processNewBinningRange(representation::t)));

		DistViewGUI *g = &p->gui;
		// let GUI connect its member's signals to us
		g->initSignals(this);

		// connect signals from outside to GUI
		connect(chief, SIGNAL(showIlluminationCurve(bool)),
				g, SIGNAL(toggleIlluminationShown(bool)));

		connect(chief, SIGNAL(toggleSingleLabel(bool)),
				g, SLOT(toggleSingleLabel(bool)));
	}

	/* connect pass-throughs / signals we process */
	connect(chief, SIGNAL(currentLabelChanged(int)),
			this, SLOT(setCurrentLabel(int)));
	connect(chief, SIGNAL(requestPixelOverlay(int,int)),
			this, SLOT(pixelOverlay(int,int)));
	connect(chief, SIGNAL(singleLabelSelected(int)),
			this, SIGNAL(singleLabelSelected(int)));
	connect(chief, SIGNAL(toggleIgnoreLabels(bool)),
			this, SLOT(toggleIgnoreLabels(bool)));

	/* connect illuminant correction stuff only to IMG distview */
	DistViewGUI *g = &map[representation::IMG]->gui;
	connect(this, SIGNAL(newIlluminantCurve(QVector<multi_img::Value>)),
			g, SIGNAL(newIlluminantCurve(QVector<multi_img::Value>)));
	connect(this, SIGNAL(toggleIlluminationShown(bool)),
			g, SIGNAL(toggleIlluminationShown(bool)));
	connect(this, SIGNAL(newIlluminantApplied(QVector<multi_img::Value>)),
			g, SIGNAL(newIlluminantApplied(QVector<multi_img::Value>)));

	/* model needs to know applied illuminant */
	connect(this, SIGNAL(newIlluminantApplied(QVector<multi_img::Value>)),
			&map[representation::IMG]->model,
			SLOT(setIlluminant(QVector<multi_img::Value>)));
}

void DistViewController::setGUIEnabled(bool enable, TaskType tt)
{
	foreach (payload *p, map) {
		p->gui.setEnabled(enable
						  || tt == TT_BIN_COUNT || tt == TT_TOGGLE_VIEWER);
	}
}

sets_ptr DistViewController::subImage(representation::t type,
							   const std::vector<cv::Rect> &regions,
							   cv::Rect roi)
{
	return map[type]->model.subImage(regions, roi);
}

void DistViewController::addImage(representation::t type, sets_ptr temp,
							   const std::vector<cv::Rect> &regions,
							   cv::Rect roi)
{
	map[type]->model.addImage(temp, regions, roi);
}

void DistViewController::setImage(representation::t type, SharedMultiImgPtr image,
							   cv::Rect roi)
{
	int bins = map[type]->gui.getBinCount();
	map[type]->model.setImage(image, roi, bins);
}

void DistViewController::setActiveViewer(representation::t type)
{
	activeView = type;
	QMap<representation::t, payload*>::const_iterator it;
	for (it = map.begin(); it != map.end(); ++it) {
		 if (it.key() != activeView)
			it.value()->gui.setInactive();
	}
}

void DistViewController::pixelOverlay(int y, int x)
{
	if (y < 0 || x < 0) { // overlay became invalid
		emit pixelOverlayInvalid();
		return;
	}

	foreach (payload *p, map) {
		QPolygonF overlay = p->model.getPixelOverlay(y, x);
		if (!overlay.empty())
			p->gui.insertPixelOverlay(overlay);
	}
}

void DistViewController::changeBinCount(representation::t type, int bins)
{
	// TODO: might be called on initialization, which might be nasty
	// TODO: too harsh
	//queue->cancelTasks();
	chief->disableGUI(TT_BIN_COUNT);

	map[type]->model.updateBinning(bins);

	chief->enableGUILater();
}

void DistViewController::updateLabels(const cv::Mat1s& labels,
								   const QVector<QColor> &colors,
								   bool colorsChanged)
{
	if (!colors.empty()) {
		foreach (payload *p, map) {
			p->model.setLabelColors(colors);
			p->gui.updateLabelColors(colors);
		}
	}

	// binset update is needed if labels or colors changed
	if (labels.empty() && (!colorsChanged))
		return;

	chief->disableGUI();

	foreach (payload *p, map) {
		p->model.updateLabels(labels, colors);
	}

	chief->enableGUILater();
}

void DistViewController::updateLabelsPartially(const cv::Mat1s &labels,
											const cv::Mat1b &mask)
{
	/* test: is it worth it to do it incrementally
	 * (2 updates for each positive entry)
	 */
	bool profitable = ((2 * cv::countNonZero(mask)) < mask.total());
	if (profitable) {

		chief->disableGUI();

		foreach (payload *p, map) {
			p->model.updateLabelsPartially(labels, mask);
		}

		chief->enableGUILater();
	} else {
		// just update the whole thing
		updateLabels(labels);
	}
}

void DistViewController::processNewBinning(representation::t type)
{
	map[type]->gui.rebuild();
}

void DistViewController::processNewBinningRange(representation::t type)
{
	DistViewModel &m = map[type]->model;
	m.clearMask();
	std::pair<multi_img::Value, multi_img::Value> range = m.getRange();
	map[type]->gui.setTitle(type, range.first, range.second);

	processNewBinning(type);
}

void DistViewController::propagateBandSelection(int band)
{
	emit bandSelected(activeView, band);
}

void DistViewController::drawOverlay(int band, int bin)
{
	DistViewModel &m = map[activeView]->model;
	m.fillMaskSingle(band, bin);
	emit requestOverlay(m.getHighlightMask());
}

void DistViewController::drawOverlay(const std::vector<std::pair<int, int> >& l,
									 int dim)
{
	DistViewModel &m = map[activeView]->model;
	if (dim > -1)
		m.updateMaskLimiters(l, dim);
	else
		m.fillMaskLimiters(l);

	emit requestOverlay(m.getHighlightMask());
}

void DistViewController::finishNormRangeImgChange(bool success)
{
	/* TODO: this is *so* wrong. what this does is after changing the range
	 * ensure that band cache is deleted and new band is shown
	 * however bandSelected will select this representation, even if it was not
	 * selected before. also this should be done by the image model
	 * and the image model has other means of doing this (empty the map)
	 */
/*	if (success) {
		SharedDataLock hlock((*image)->mutex);
		(*bands)[representation::IMG].assign((**image)->size(), NULL);
		hlock.unlock();
		emit bandSelected(representation::IMG, map.value(representation::IMG)->getSelection());
	}*/
}

void DistViewController::finishNormRangeGradChange(bool success)
{
	// ************** see above
/*	if (success) {
		SharedDataLock hlock((*gradient)->mutex);
		(*bands)[GRAD].assign((**gradient)->size(), NULL);
		hlock.unlock();
		emit bandSelected(GRAD,	map.value(GRAD)->getSelection());
	}*/
}

void DistViewController::toggleIgnoreLabels(bool toggle)
{
	// TODO: cancel previous toggleignorelabel tasks here!
	chief->disableGUI(TT_TOGGLE_LABELS);

	foreach (payload *p, map) {
		p->model.toggleLabels(toggle);
	}

	chief->enableGUILater(false);
}

void DistViewController::addHighlightToLabel()
{
	cv::Mat1b mask = map[activeView]->model.getHighlightMask();
	emit alterLabelRequested(currentLabel, mask, false);
}

void DistViewController::remHighlightFromLabel()
{
	cv::Mat1b mask = map[activeView]->model.getHighlightMask();
	emit alterLabelRequested(currentLabel, mask, true);
}
