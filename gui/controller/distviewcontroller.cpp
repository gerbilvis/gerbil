#include "distviewcontroller.h"

#include <dist_view/viewport.h>
#include <widgets/mainwindow.h>

#include <model/imagemodel.h>

#include <QSignalMapper>

// for debug msgs
#include <boost/format.hpp>
#include <QDebug>

#define GGDBG_MODULE
#include <QSettings>
#include <gerbil_gui_debug.h>

#ifdef GGDBG_MODULE
#define GGDBG_REPR(type) GGDBGM(format("%1%")%type << endl)
#endif

DistViewController::DistViewController(
        Controller *ctrl,
        BackgroundTaskQueue *taskQueue,
        ImageModel *im)
    : QObject(ctrl), ctrl(ctrl), im(im)
{
	/* Create all viewers. Only one viewer per representation type is supported.
	 */
	for (auto r : representation::all()) {
		payloadMap.insert(r, new Payload(r));
		auto p = payloadMap[r];
		p->model.setTaskQueue(taskQueue);
		// TODO: the other way round. ctx+sets come from model
		p->model.setContext(payloadMap[r]->gui.getContext());
		p->model.setBinSets(payloadMap[r]->gui.getBinSets());
		p->viewFolded = false;
		p->binningNeeded = false;
		p->isSubscribed = false;
	}
}

void DistViewController::init()
{
	// First, set viewport folding and activeView.
	// NOTE that we did not connect signals yet! This is actually good as we
	// will not get stray signals before the content is ready.
	QSettings settings;
	if (false && settings.contains("viewports/IMGfolded")) { /* TODO: disabled to avoid crashes */
		// settings are available
		QString activeStr = settings.value("viewports/active").value<QString>();
		activeView = representation::fromStr(activeStr);
		// make sure activeView is a valid enum value and we actually have a
		// viewer for this representation.
		if (!representation::valid(activeView) ||
		    !payloadMap.contains(activeView)) {
			activeView = representation::IMG;
		}
		payloadMap[activeView]->gui.setActive();
		for (auto r : payloadMap.keys()) {
			auto &p = payloadMap[r];
			QString key = QString("viewports/") +
			              representation::str(r) + "folded";
			// read folded setting, default unfolded
			bool folded = settings.value(key, false).value<bool>();
			p->viewFolded = folded;
			p->gui.fold(folded);
		}
	} else {
		// no settings available
		// Show IMG and GRAD. Hide NORM, IMGPCA and GRADPCA at the
		// beginning.
		activeView = representation::IMG;
		payloadMap[activeView]->gui.setActive();
		payloadMap[representation::NORM]->gui.fold(true);
		payloadMap[representation::IMGPCA]->gui.fold(true);
//		payloadMap[representation::GRADPCA]->gui.fold(true);
	}

	connect(QApplication::instance(), SIGNAL(lastWindowClosed()),
	        this, SLOT(processLastWindowClosed()));

	connect(ctrl->imageModel(), SIGNAL(roiRectChanged(cv::Rect)),
	        this, SLOT(processROIChage(cv::Rect)));
	connect(ctrl->imageModel(),
	        SIGNAL(imageUpdate(representation::t,SharedMultiImgPtr,bool)),
	        this,
	        SLOT(processImageUpdate(representation::t,SharedMultiImgPtr,bool)));


	for (auto p : payloadMap) {
		// fill GUI with distribution view
		ctrl->mainWindow()->addDistView(p->gui.getFrame());
	}

	for (auto p : payloadMap) {
		DistViewModel *m = &p->model;

		connect(m, SIGNAL(newBinning(representation::t)),
		        this, SLOT(processNewBinning(representation::t)));
		connect(m, SIGNAL(newBinningRange(representation::t)),
		        this, SLOT(processNewBinningRange(representation::t)));

		DistViewGUI *g = &p->gui;
		// let GUI connect its member's signals to us
		g->initSignals(this);

		// connect signals from outside to GUI
		connect(ctrl, SIGNAL(showIlluminationCurve(bool)),
		        g, SIGNAL(toggleIlluminationShown(bool)));


		connect(g, SIGNAL(needBinning(representation::t)),
		        this, SLOT(processDistviewNeedsBinning(representation::t)));


		// forward representation subscriptions to controller
		connect(g, SIGNAL(subscribeRepresentation(QObject*,representation::t)),
		        ctrl, SLOT(subscribeRepresentation(QObject*,representation::t)));
		connect(g, SIGNAL(unsubscribeRepresentation(QObject*,representation::t)),
		        ctrl, SLOT(unsubscribeRepresentation(QObject*,representation::t)));
		// fill-in our own state-holder (TODO)
		connect(g, SIGNAL(subscribeRepresentation(QObject*,representation::t)),
		        this, SLOT(subscribeRepresentation(QObject*,representation::t)));
		connect(g, SIGNAL(unsubscribeRepresentation(QObject*,representation::t)),
		        this, SLOT(unsubscribeRepresentation(QObject*,representation::t)));

		connect(g, SIGNAL(foldingStateChanged(representation::t,bool)),
		        this, SLOT(processFoldingStateChanged(representation::t,bool)));
	}

	/* connect pass-throughs / signals we process */
	connect(ctrl, SIGNAL(currentLabelChanged(int)),
	        this, SLOT(setCurrentLabel(int)));
	connect(ctrl, SIGNAL(requestPixelOverlay(int,int)),
	        this, SLOT(pixelOverlay(int,int)));
	connect(ctrl, SIGNAL(labelSelected(int)),
	        this, SIGNAL(labelSelected(int)));
	connect(ctrl, SIGNAL(toggleIgnoreLabels(bool)),
	        this, SLOT(toggleIgnoreLabels(bool)));

	/* connect illuminant correction stuff only to IMG distview */
	DistViewGUI *g = &payloadMap[representation::IMG]->gui;
	connect(this, SIGNAL(newIlluminantCurve(QVector<multi_img::Value>)),
	        g, SIGNAL(newIlluminantCurve(QVector<multi_img::Value>)));
	connect(this, SIGNAL(toggleIlluminationShown(bool)),
	        g, SIGNAL(toggleIlluminationShown(bool)));
	connect(this, SIGNAL(newIlluminantApplied(QVector<multi_img::Value>)),
	        g, SIGNAL(newIlluminantApplied(QVector<multi_img::Value>)));

	/* model needs to know applied illuminant */
	connect(this, SIGNAL(newIlluminantApplied(QVector<multi_img::Value>)),
	        &payloadMap[representation::IMG]->model,
	        SLOT(setIlluminant(QVector<multi_img::Value>)));
}

void DistViewController::initSubscriptions()
{
	for (auto p : payloadMap) {
		p->gui.initSubscriptions();
	}
}

sets_ptr DistViewController::subImage(representation::t type,
                                      const std::vector<cv::Rect> &regions,
                                      cv::Rect roi)
{
	GGDBGM(type << endl);
	return payloadMap[type]->model.subImage(regions, roi);
}

void DistViewController::addImage(representation::t type, sets_ptr temp,
                                  const std::vector<cv::Rect> &regions,
                                  cv::Rect roi)
{
	GGDBGM(type << endl);
	payloadMap[type]->model.addImage(temp, regions, roi);
}

void DistViewController::setActiveViewer(representation::t type)
{
	activeView = type;
	for (auto r : payloadMap.keys()) {
		if (r != activeView)
			payloadMap[r]->gui.setInactive();
	}
}

void DistViewController::pixelOverlay(int y, int x)
{

	if (y < 0 || x < 0) { // overlay became invalid
		emit pixelOverlayInvalid();
		return;
	}

	for (auto p : payloadMap) {
		if (!p->isSubscribed)
			continue;
		QPolygonF overlay = p->model.getPixelOverlay(y, x);
		if (!overlay.empty())
			p->gui.insertPixelOverlay(overlay);
	}
}

void DistViewController::processROIChage(cv::Rect roi)
{
	curROI = roi;
}

void DistViewController::processImageUpdate(representation::t repr,
                                            SharedMultiImgPtr image,
                                            bool duplicate)
{

	if (cv::Rect() == curROI) {
		std::cerr << "DistViewController::processImageUpdate() error: "
		          << "DVC internal ROI is empty" << std::endl;
	}
	if (duplicate) {
		GGDBGM(repr  << " is re-spawn" << endl);
	}
	if (payloadMap[repr]->binningNeeded) {
		GGDBGM("following distview " << repr <<
		       " request for new binning" << endl);
		updateBinning(repr, image);
	} else {
		GGDBGM("no binning requests, ignoring update" << endl);
	}
}

void DistViewController::processDistviewNeedsBinning(representation::t repr)
{
	GGDBGM(repr << " needs binning" << endl);
	payloadMap[repr]->binningNeeded = true;
}

void DistViewController::processPreROISpawn(const cv::Rect &oldroi,
                                            const cv::Rect &newroi,
                                            const std::vector<cv::Rect> &sub,
                                            const std::vector<cv::Rect> &add,
                                            bool profitable)
{
	if (profitable) {
		GGDBGM("INCREMENTAL distview update" << endl);
		for (auto r : payloadMap.keys()) {
			auto &p = payloadMap[r];
			if (!p->isSubscribed)
				continue;
			GGDBGM("   BEGIN " << r <<" distview update" << endl);
			p->tmp_binset = subImage(r, sub, newroi);
			GGDBGM("   END " << r <<" distview update" << endl);
		}
	} else {
		GGDBGM("FULL distview update (ignored)" << endl);
		// nothing to do, except let them know that ROI change is underway
		for (auto p : payloadMap) {
			if (!p->isSubscribed)
				continue;
			p->model.initiateROIChange();
		}
	}
}

void DistViewController::processPostROISpawn(const cv::Rect &oldroi,
                                             const cv::Rect &newroi,
                                             const std::vector<cv::Rect> &sub,
                                             const std::vector<cv::Rect> &add,
                                             bool profitable)
{
	GGDBGM((profitable ? "INCREMENTAL" : "FULL") << " distview update" << endl);
	for (auto r : payloadMap.keys()) {
		auto &p = payloadMap[r];
		if (!p->isSubscribed)
			continue;
		if (profitable && p->tmp_binset) {
			addImage(r, p->tmp_binset, add, newroi);
			p->tmp_binset.reset();
		} else {
			updateBinning(r, im->getImage(r));
		}
	}
}

void DistViewController::updateBinning(representation::t repr,
                                       SharedMultiImgPtr image)
{
	payloadMap[repr]->binningNeeded = false;
	int bins = payloadMap[repr]->gui.getBinCount();
	payloadMap[repr]->model.setImage(image, curROI, bins);
}

void DistViewController::changeBinCount(representation::t type, int bins)
{
	// FIXME: we cannot queue->cancelTasks(): might be called on
	// initialization -> nasty!
	// We need a better way to replace active binning tasks.
	//queue->cancelTasks();

	payloadMap[type]->model.updateBinning(bins);
}

void DistViewController::updateLabels(const cv::Mat1s& labels,
                                      const QVector<QColor> &colors,
                                      bool colorsChanged)
{
	if (!colors.empty()) {
		for (auto p : payloadMap) {
			p->model.setLabelColors(colors);
			p->gui.updateLabelColors(colors);
		}
	}

	// binset update is needed if labels or colors changed
	if (labels.empty() && (!colorsChanged))
		return;

	for (auto p : payloadMap) {
		p->model.updateLabels(labels, p->isSubscribed);
	}
}

void DistViewController::updateLabelsPartially(const cv::Mat1s &labels,
                                               const cv::Mat1b &mask)
{
	/* test: is it worth it to do it incrementally
	 * (2 updates for each positive entry)
	 */
	bool profitable = (size_t(2 * cv::countNonZero(mask)) < mask.total());
	if (profitable) {
		for (auto p : payloadMap) {
			p->model.updateLabelsPartially(labels, mask, p->isSubscribed);
		}
	} else {
		// just update the whole thing
		updateLabels(labels);
	}
}

void DistViewController::processNewBinning(representation::t type)
{
	payloadMap[type]->gui.rebuild();
}

void DistViewController::processNewBinningRange(representation::t type)
{
	DistViewModel &m = payloadMap[type]->model;
	m.clearMask();
	std::pair<multi_img::Value, multi_img::Value> range = m.getRange();
	payloadMap[type]->gui.setTitle(type, range.first, range.second);

	processNewBinning(type);
}

void DistViewController::propagateBandSelection(int band)
{
	emit bandSelected(activeView, band);
}

void DistViewController::drawOverlay(int band, int bin)
{
	DistViewModel &m = payloadMap[activeView]->model;
	m.fillMaskSingle(band, bin);
	emit requestOverlay(m.getHighlightMask());
}

void DistViewController::drawOverlay(const std::vector<std::pair<int, int> >& l,
                                     int dim)
{
	DistViewModel &m = payloadMap[activeView]->model;
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

	for (auto p : payloadMap) {
		p->model.toggleLabels(toggle, p->isSubscribed);
	}

}

void DistViewController::addHighlightToLabel()
{
	cv::Mat1b mask = payloadMap[activeView]->model.getHighlightMask();
	emit alterLabelRequested(currentLabel, mask, false);
}

void DistViewController::remHighlightFromLabel()
{
	cv::Mat1b mask = payloadMap[activeView]->model.getHighlightMask();
	emit alterLabelRequested(currentLabel, mask, true);
}

void DistViewController::processFoldingStateChanged(representation::t repr, bool folded)
{
	payloadMap[repr]->viewFolded = folded;
	// TODO: if all folded, disable add/remove from label buttons.
}

void DistViewController::processLastWindowClosed()
{
	// save folding state of views
	QSettings settings;
	for (auto r : payloadMap.keys()) {
		bool folded = payloadMap[r]->viewFolded;
		QString key = QString("viewports/") +
		              representation::str(r) + "folded";
		GGDBGM(key.toStdString() << " " << folded << endl);
		settings.setValue(key, QVariant(folded));
	}
	settings.setValue("viewports/active", representation::str(activeView));
}
