/*
	Copyright(c) 2012 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "viewerwindow.h"
#include "iogui.h"
#include "commandrunner.h"

#include <background_task_queue.h>

#include <labeling.h>
#include <qtopencv.h>

#include <highgui.h>

#include <QPainter>
#include <QIcon>
#include <QSignalMapper>
#include <iostream>
#include <QShortcut>

ViewerWindow::ViewerWindow(multi_img *image, QWidget *parent)
	: QMainWindow(parent),
	  full_image(new SharedData<multi_img>(image)),
	  image(new SharedData<multi_img>(new multi_img(0, 0, 0))),
	  gradient(new SharedData<multi_img>(new multi_img(0, 0, 0))),
	  imagepca(new SharedData<multi_img>(new multi_img(0, 0, 0))),
	  gradientpca(new SharedData<multi_img>(new multi_img(0, 0, 0))),
	  normIMG(NORM_OBSERVED), normGRAD(NORM_OBSERVED),
	  normIMGRange(new SharedData<std::pair<multi_img::Value, multi_img::Value> >(
	  		new std::pair<multi_img::Value, multi_img::Value>(0, 0))),
	  normGRADRange(new SharedData<std::pair<multi_img::Value, multi_img::Value> >(
	  		new std::pair<multi_img::Value, multi_img::Value>(0, 0))),
	  full_rgb_temp(new SharedData<QImage>(new QImage())),
	  usRunner(NULL), contextMenu(NULL),
	  viewers(REPSIZE), activeViewer(0)
{
	// create all objects
	setupUi(this);
	viewers[IMG] = viewIMG;
	viewers[GRAD] = viewGRAD;
	viewers[IMGPCA] = viewIMGPCA;
	viewers[GRADPCA] = viewGRADPCA;

	// do all the bling-bling
	initUI();

	roiView->roi = QRect(0, 0, (*full_image)->width, (*full_image)->height);

	// default to full image if small enough
	if ((*full_image)->width*(*full_image)->height < 513*513) {
		roi = cv::Rect(0, 0, (*full_image)->width, (*full_image)->height);
		applyROI(false);
	} else {
		ROITrigger();
	}
}

void ViewerWindow::applyROI(bool reuse)
{
	/// first: set up new working images

	if (!reuse) {
		cv::Rect empty(0, 0, 0, 0);
		(*image)->roi = empty;
		(*gradient)->roi = empty;
		(*imagepca)->roi = empty;
		(*gradientpca)->roi = empty;
	}

	size_t numbands = bandsSlider->value();
	if (numbands <= 0)
		numbands = 1;
	if (numbands > (*full_image)->size())
		numbands = (*full_image)->size();

	multi_img_ptr scoped_image(
		new SharedData<multi_img>(new multi_img(**full_image, roi)));
	BackgroundTaskPtr taskRescale(new MultiImg::RescaleTbb(
		scoped_image, image, numbands, roi));
	//QObject::connect(taskRescale.get(), SIGNAL(finished(bool)), , SLOT());
	BackgroundTaskQueue::instance().push(taskRescale);
	taskRescale->wait();

	BackgroundTaskPtr taskGradient(new MultiImg::GradientTbb(
		image, gradient, roi));
	//QObject::connect(taskGradient.get(), SIGNAL(finished(bool)), , SLOT());
	BackgroundTaskQueue::instance().push(taskGradient);
	taskGradient->wait();

	{
		BackgroundTaskPtr taskPca(new MultiImg::PcaTbb(
			image, imagepca, 0, roi));
		//QObject::connect(taskPca.get(), SIGNAL(finished(bool)), , SLOT());
		BackgroundTaskQueue::instance().push(taskPca);
		taskPca->wait();
	}
	{
		BackgroundTaskPtr taskPca(new MultiImg::PcaTbb(
			gradient, gradientpca, 0, roi));
		//QObject::connect(taskPca.get(), SIGNAL(finished(bool)), , SLOT());
		BackgroundTaskQueue::instance().push(taskPca);
		taskPca->wait();
	}

	// calculate new norm ranges inside ROI

	BackgroundTaskPtr taskImgNormRange(new NormRangeTbb(
		image, normIMGRange, normIMG, 0, true, roi));
	//QObject::connect(taskImgNormRange.get(), SIGNAL(finished(bool)), , SLOT());
	BackgroundTaskQueue::instance().push(taskImgNormRange);
	taskImgNormRange->wait();

	BackgroundTaskPtr taskGradNormRange(new NormRangeTbb(
		gradient, normGRADRange, normGRAD, 1, true, roi));
	//QObject::connect(taskGradNormRange.get(), SIGNAL(finished(bool)), , SLOT());
	BackgroundTaskQueue::instance().push(taskGradNormRange);
	taskGradNormRange->wait();

	// gui update (if norm ranges changed)
	normTargetChanged(true);

	/* set labeling, and label colors (depend on ROI size) */
	cv::Mat1s &labels = bandView->labels;
	if (labels.empty() ||
		labels.rows != (*image)->height || labels.cols != (*image)->width) {
		labels = cv::Mat1s((*image)->height, (*image)->width, (short)0);
		for (size_t i = 0; i < viewers.size(); ++i)
			viewers[i]->labels = labels;
	}
	if (labelColors.empty())
		setLabelColors(vole::Labeling::colors(2, true));

	/// second: re-initialize gui
	/* empty/init caches */
	// note: each vector's size is atmost #bands (e.g., gradient has one less)
	bands.assign(viewers.size(), std::vector<QPixmap*>((*image)->size(), NULL));

	/* do setImage() last */
	viewIMG->setImage(&(**image), IMG);
	viewIMGPCA->setImage(&(**imagepca), IMGPCA);
	viewGRAD->setImage(&(**gradient), GRAD);
	viewGRADPCA->setImage(&(**gradientpca), GRADPCA);

	// updateBand only works after image is set (it gets image from viewer)
	updateBand();
	updateRGB(true);

	bandDock->widget()->setEnabled(true);
	rgbDock->widget()->setEnabled(true);
	mainStack->setCurrentIndex(0);
}

void ViewerWindow::initUI()
{
	/* GUI elements */
	initGraphsegUI();
	initIlluminantUI();
#ifdef WITH_SEG_MEANSHIFT
	initUnsupervisedSegUI();
#endif
	initNormalizationUI();

	/* more manual work to get GUI in proper shape */
	graphsegWidget->hide();

	bandDock->widget()->setDisabled(true);
	rgbDock->widget()->setDisabled(true);

	// start with IMG, hide IMGPCA, GRADPCA at the beginning
	activeViewer = viewIMG;
	viewIMG->setActive();
	viewIMGPCA->toggleFold();
	viewGRADPCA->toggleFold();

	// dock arrangement
	tabifyDockWidget(labelDock, illumDock);
	tabifyDockWidget(labelDock, normDock);
#ifdef WITH_SEG_MEANSHIFT
	usDock->hide();
#else
	usDock->deleteLater();
#endif

	/* slots & signals */
	connect(docksButton, SIGNAL(clicked()),
			this, SLOT(openContextMenu()));

	connect(bandDock, SIGNAL(topLevelChanged(bool)),
			this, SLOT(reshapeDock(bool)));

	connect(markerSelector, SIGNAL(currentIndexChanged(int)),
			bandView, SLOT(changeLabel(int)));
	connect(clearButton, SIGNAL(clicked()),
			bandView, SLOT(clearLabelPixels()));

	connect(bandView, SIGNAL(newLabel()),
			this, SLOT(createLabel()));

	// label buttons
	connect(lLoadButton, SIGNAL(clicked()),
			this, SLOT(loadLabeling()));
	connect(lSaveButton, SIGNAL(clicked()),
			this, SLOT(saveLabeling()));
	connect(lLoadSeedButton, SIGNAL(clicked()),
			this, SLOT(loadSeeds()));

	// signals for ROI
	connect(roiButtons, SIGNAL(clicked(QAbstractButton*)),
			this, SLOT(ROIDecision(QAbstractButton*)));
	connect(roiButton, SIGNAL(clicked()), this, SLOT(ROITrigger()));
	connect(roiView, SIGNAL(newSelection(QRect)),
			this, SLOT(ROISelection(QRect)));

	connect(ignoreButton, SIGNAL(toggled(bool)),
			markButton, SLOT(setDisabled(bool)));
	connect(ignoreButton, SIGNAL(toggled(bool)),
			nonmarkButton, SLOT(setDisabled(bool)));
	connect(ignoreButton, SIGNAL(toggled(bool)),
			bandView, SLOT(toggleShowLabels(bool)));

	connect(addButton, SIGNAL(clicked()),
			this, SLOT(addToLabel()));
	connect(remButton, SIGNAL(clicked()),
			this, SLOT(remFromLabel()));
	connect(this, SIGNAL(alterLabel(const multi_img::Mask&,bool)),
			bandView, SLOT(alterLabel(const multi_img::Mask&,bool)));
	connect(this, SIGNAL(drawOverlay(const multi_img::Mask&)),
			bandView, SLOT(drawOverlay(const multi_img::Mask&)));

	connect(this, SIGNAL(newLabelColors(const QVector<QColor>&, bool)),
			bandView, SLOT(setLabelColors(const QVector<QColor>&, bool)));
	connect(alphaSlider, SIGNAL(valueChanged(int)),
			bandView, SLOT(applyLabelAlpha(int)));

	// for self-activation of viewports
	QSignalMapper *vpmap = new QSignalMapper(this);
	for (size_t i = 0; i < viewers.size(); ++i)
		vpmap->setMapping(viewers[i]->getViewport(), (int)i);
	connect(vpmap, SIGNAL(mapped(int)),
			this, SLOT(setActive(int)));

	for (size_t i = 0; i < viewers.size(); ++i)
	{
		multi_img_viewer *v = viewers[i];
		const Viewport *vp = v->getViewport();

		connect(applyButton, SIGNAL(clicked()),
				v, SLOT(rebuild()));
		connect(markButton, SIGNAL(toggled(bool)),
				v, SLOT(toggleLabeled(bool)));
		connect(nonmarkButton, SIGNAL(toggled(bool)),
				v, SLOT(toggleUnlabeled(bool)));
		connect(ignoreButton, SIGNAL(toggled(bool)),
				v, SLOT(toggleLabels(bool)));

		connect(this, SIGNAL(newLabelColors(const QVector<QColor>&, bool)),
				v, SLOT(updateLabelColors(const QVector<QColor>&, bool)));

		connect(bandView, SIGNAL(pixelOverlay(int, int)),
				v, SLOT(overlay(int, int)));

		connect(vp, SIGNAL(activated()),
				vpmap, SLOT(map()));

		for (size_t j = 0; j < viewers.size(); ++j) {
			multi_img_viewer *v2 = viewers[j];
			const Viewport *vp2 = v2->getViewport();
			// connect folding signal to all viewports
			connect(v, SIGNAL(folding()),
					vp2, SLOT(folding()));

			if (i == j)
				continue;
			// connect activation signal to all *other* viewers
			connect(vp, SIGNAL(activated()),
			        v2, SLOT(setInactive()));
		}

		connect(vp, SIGNAL(bandSelected(representation, int)),
				this, SLOT(selectBand(representation, int)));

		connect(v, SIGNAL(newOverlay()),
				this, SLOT(newOverlay()));
		connect(vp, SIGNAL(newOverlay(int)),
				this, SLOT(newOverlay()));

		connect(vp, SIGNAL(addSelection()),
				this, SLOT(addToLabel()));
		connect(vp, SIGNAL(remSelection()),
				this, SLOT(remFromLabel()));

		connect(bandView, SIGNAL(killHover()),
				vp, SLOT(killHover()));
	}

	/// init bandsSlider
	bandsLabel->setText(QString("%1 bands").arg((*full_image)->size()));
	bandsSlider->setMaximum((*full_image)->size());
	bandsSlider->setValue((*full_image)->size());
	connect(bandsSlider, SIGNAL(valueChanged(int)),
			this, SLOT(bandsSliderMoved(int)));
	connect(bandsSlider, SIGNAL(sliderMoved(int)),
			this, SLOT(bandsSliderMoved(int)));

	/// global shortcuts
	QShortcut *scr = new QShortcut(Qt::CTRL + Qt::Key_S, this);
	connect(scr, SIGNAL(activated()), this, SLOT(screenshot()));

	BackgroundTaskPtr taskRgb(new ViewerWindow::RgbTbb(
		full_image, mat_vec3f_ptr(new SharedData<cv::Mat_<cv::Vec3f> >(new cv::Mat_<cv::Vec3f>)), full_rgb_temp));
	//QObject::connect(taskRgb.get(), SIGNAL(finished(bool)), this, SLOT(updateRGB(bool)));
	BackgroundTaskQueue::instance().push(taskRgb);
	updateRGB(taskRgb->wait());
}

void ViewerWindow::bandsSliderMoved(int b)
{
	bandsLabel->setText(QString("%1 bands").arg(b));
	if (!bandsSlider->isSliderDown())
		applyROI(false);
}

#ifdef WITH_SEG_MEANSHIFT
void ViewerWindow::usMethodChanged(int idx)
{
	// idx: 0 Meanshift, 1 Medianshift, 2 Probabilistic Shift
	usSkipPropWidget->setEnabled(idx == 1);
	usSpectralWidget->setEnabled(idx == 2);
	usMSPPWidget->setEnabled(idx == 2);
}

void ViewerWindow::usInitMethodChanged(int idx)
{
	switch (usInitMethodBox->itemData(idx).toInt()) {
	case vole::JUMP:
		usInitPercentWidget->hide();
		usInitJumpWidget->show();
		break;
	case vole::PERCENT:
		usInitJumpWidget->hide();
		usInitPercentWidget->show();
		break;
	default:
		usInitJumpWidget->hide();
		usInitPercentWidget->hide();
	}
}
#endif

bool ViewerWindow::setLabelColors(const std::vector<cv::Vec3b> &colors)
{
	QVector<QColor> col = vole::Vec2QColor(colors);
	col[0] = Qt::white; // override black for 0 label

	// test if costy rebuilds necessary (existing colors changed)
	bool changed = false;
	for (int i = 1; i < labelColors.size() && i < col.size(); ++i) {
		if (col[i] != labelColors[i])
			changed = true;
	}

	labelColors = col;

	// use colors for our awesome label menu
	markerSelector->clear();
	for (int i = 1; i < labelColors.size(); ++i) // 0 is index for unlabeled
	{
		markerSelector->addItem(colorIcon(labelColors[i]), "");
	}
	markerSelector->addItem(QIcon(":/toolbar/add"), "");

	// tell others about colors
	emit newLabelColors(labelColors, changed);
	return changed;
}

void ViewerWindow::setLabels(const vole::Labeling &labeling)
{
	assert(labeling().rows == (*image)->height && labeling().cols == (*image)->width);
	/* note: always update labels before updating label colors, for the case
	   that there are less colors available than used in previous labeling */
	cv::Mat1s labels = labeling();
	// following assignments are probably redundant (OpenCV shallow copy)
	bandView->labels = labels;
	for (size_t i = 0; i < viewers.size(); ++i)
		viewers[i]->labels = labels;

	bool updated = setLabelColors(labeling.colors());
	if (!updated) {
		for (size_t i = 0; i < viewers.size(); ++i)
			viewers[i]->rebuild();
		bandView->refresh();
	}
}

void ViewerWindow::createLabel()
{
	int index = labelColors.count();
	// increment colors by 1
	setLabelColors(vole::Labeling::colors(index + 1, true));
	// select our new label for convenience
	markerSelector->setCurrentIndex(index - 1);
}

const QPixmap* ViewerWindow::getBand(representation type, int dim)
{
	std::vector<QPixmap*> &v = bands[type];

	if (!v[dim]) {
		multi_img_ptr multi; // = viewers[type]->getImage();
		if (type == IMG)
			multi = image;
		else if (type == GRAD)
			multi = gradient;
		else if (type == IMGPCA)
			multi = imagepca;
		else //if (type == GRADPCA)
			multi = gradientpca;

		qimage_ptr qimg(new SharedData<QImage>(new QImage()));

		BackgroundTaskPtr taskExport(new MultiImg::Band2QImageTbb(multi, qimg, dim));
		//QObject::connect(taskExport.get(), SIGNAL(finished(bool)), , SLOT());
		BackgroundTaskQueue::instance().push(taskExport);
		taskExport->wait();

		v[dim] = new QPixmap(QPixmap::fromImage(**qimg));
	}
	return v[dim];
}

void ViewerWindow::updateBand()
{
	selectBand(activeViewer->getViewport()->type,
			   activeViewer->getViewport()->selection);
	bandView->update();
}

void ViewerWindow::selectBand(representation type, int dim)
{
	bandView->setPixmap(*getBand(type, dim));
	const multi_img *m = viewers[type]->getImage();
	std::string banddesc = m->meta[dim].str();
	QString title;
	if (banddesc.empty())
		title = QString("%1 Band #%2")
			.arg(type == GRAD ? "Gradient" : "Image") // TODO
			.arg(dim+1);
	else
		title = QString("%1 Band %2")
			.arg(type == GRAD ? "Gradient" : "Image") // TODO
			.arg(banddesc.c_str());

	bandDock->setWindowTitle(title);
}

void ViewerWindow::applyIlluminant() {
	int i1 = i1Box->itemData(i1Box->currentIndex()).value<int>();
	int i2 = i2Box->itemData(i2Box->currentIndex()).value<int>();
	if (i1 == i2)
		return;

	i1Box->setDisabled(true);
	i1Check->setVisible(true);

	/* remove old illuminant */
	if (i1 != 0) {
		const Illuminant &il = getIlluminant(i1);
		(*full_image)->apply_illuminant(il, true);
	}

	/* add new illuminant */
	if (i2 != 0) {
		const Illuminant &il = getIlluminant(i2);
		(*full_image)->apply_illuminant(il);
	}

	viewIMG->setIlluminant((i2 ? &getIlluminantC(i2) : NULL), true);
	/* rebuild  */
	if (roi.width > 0)
		applyROI(false);

	BackgroundTaskPtr taskRgb(new ViewerWindow::RgbTbb(
		full_image, mat_vec3f_ptr(new SharedData<cv::Mat_<cv::Vec3f> >(new cv::Mat_<cv::Vec3f>)), full_rgb_temp));
	//QObject::connect(taskRgb.get(), SIGNAL(finished(bool)), this, SLOT(updateRGB(bool)));
	BackgroundTaskQueue::instance().push(taskRgb);
	updateRGB(taskRgb->wait());

	/* reflect change in our own gui (will propagate to viewIMG) */
	i1Box->setCurrentIndex(i2Box->currentIndex());
}

bool ViewerWindow::RgbSerial::run()
{
	if (!MultiImg::BgrSerial::run())
		return false;
	SharedDataRead rlock(bgr->lock);
	cv::Mat_<cv::Vec3f> &bgrmat = *(*bgr);
	QImage *newRgb = new QImage(bgrmat.cols, bgrmat.rows, QImage::Format_ARGB32);
	for (int y = 0; y < bgrmat.rows; ++y) {
		cv::Vec3f *row = bgrmat[y];
		QRgb *destrow = (QRgb*)newRgb->scanLine(y);
		for (int x = 0; x < bgrmat.cols; ++x) {
			cv::Vec3f &c = row[x];
			destrow[x] = qRgb(c[2]*255., c[1]*255., c[0]*255.);
		}
	}
	SharedDataWrite wlock(rgb->lock);
	delete rgb->swap(newRgb);
	return true;
}

bool ViewerWindow::RgbTbb::run()
{
	if (!MultiImg::BgrTbb::run())
		return false;
	if (stopper.is_group_execution_cancelled())
		return false;

	SharedDataRead rlock(bgr->lock);
	cv::Mat_<cv::Vec3f> &bgrmat = *(*bgr);
	QImage *newRgb = new QImage(bgrmat.cols, bgrmat.rows, QImage::Format_ARGB32);

	Rgb computeRgb(bgrmat, *newRgb);
	tbb::parallel_for(tbb::blocked_range2d<int>(0, bgrmat.rows, 0, bgrmat.cols), 
		computeRgb, tbb::auto_partitioner(), stopper);

	if (stopper.is_group_execution_cancelled()) {
		delete newRgb;
		return false;
	} else {
		SharedDataWrite wlock(rgb->lock);
		delete rgb->swap(newRgb);
		return true;
	}
}

void ViewerWindow::RgbTbb::Rgb::operator()(const tbb::blocked_range2d<int> &r) const
{
	for (int y = r.rows().begin(); y != r.rows().end(); ++y) {
		cv::Vec3f *row = bgr[y];
		QRgb *destrow = (QRgb*)rgb.scanLine(y);
		for (int x = r.cols().begin(); x != r.cols().end(); ++x) {
			cv::Vec3f &c = row[x];
			destrow[x] = qRgb(c[2]*255., c[1]*255., c[0]*255.);
		}
	}
}

void ViewerWindow::updateRGB(bool success)
{
	if (!success)
		return;

	SharedDataWrite lock(full_rgb_temp->lock);
	if (!(*full_rgb_temp)->isNull()) {
		full_rgb = QPixmap::fromImage(**full_rgb_temp);
		delete full_rgb_temp->swap(new QImage());
	}
	lock.unlock();
	roiView->setPixmap(full_rgb);
	roiView->update();
	if (roi.width > 0) {
		rgb = full_rgb.copy(roi.x, roi.y, roi.width, roi.height);
		rgbView->setPixmap(rgb);
		rgbView->update();
	}
}

void ViewerWindow::initIlluminantUI()
{
	for (int i = 0; i < 2; ++i) {
		QComboBox *b = (i ? i2Box : i1Box);
		b->addItem("Neutral", 0);
		b->addItem("2,856 K (Illuminant A, light bulb)",	2856);
		b->addItem("3,100 K (Tungsten halogen lamp)",		3100);
		b->addItem("5,000 K (Horizon light)",				5000);
		b->addItem("5,500 K (Mid-morning daylight)",		5500);
		b->addItem("6,500 K (Noon daylight)",				6500);
		b->addItem("7,500 K (North sky daylight)",			7500);
	}
	connect(i2Button, SIGNAL(clicked()),
			this, SLOT(applyIlluminant()));
	connect(i1Box, SIGNAL(currentIndexChanged(int)),
			this, SLOT(setI1(int)));
	connect(i1Check, SIGNAL(toggled(bool)),
			this, SLOT(setI1Visible(bool)));
	i1Check->setVisible(false);
}

void ViewerWindow::initNormalizationUI()
{
	normModeBox->addItem("Observed");
	normModeBox->addItem("Theoretical");
	normModeBox->addItem("Fixed");
	connect(normIButton, SIGNAL(toggled(bool)),
			this, SLOT(normTargetChanged()));
	connect(normGButton, SIGNAL(toggled(bool)),
			this, SLOT(normTargetChanged()));
	connect(normModeBox, SIGNAL(currentIndexChanged(int)),
			this, SLOT(normModeSelected(int)));
	connect(normMinBox, SIGNAL(valueChanged(double)),
			this, SLOT(normModeFixed()));
	connect(normMaxBox, SIGNAL(valueChanged(double)),
			this, SLOT(normModeFixed()));
	connect(normApplyButton, SIGNAL(clicked()),
			this, SLOT(applyNormUserRange()));
	connect(normClampButton, SIGNAL(clicked()),
			this, SLOT(clampNormUserRange()));
}

bool ViewerWindow::NormRangeTbb::run()
{
	switch (mode) {
	case NORM_OBSERVED:
		if (!MultiImg::DataRangeTbb::run())
			return false;
		break;
	case NORM_THEORETICAL:
		if (!stopper.is_group_execution_cancelled()) {
			SharedDataWrite wlock(range->lock);
			// hack!
			if (target == 0) {
				(*range)->first = (multi_img::Value)MULTI_IMG_MIN_DEFAULT;
				(*range)->second = (multi_img::Value)MULTI_IMG_MAX_DEFAULT;
			} else {
				(*range)->first = (multi_img::Value)-log(MULTI_IMG_MAX_DEFAULT);
				(*range)->second = (multi_img::Value)log(MULTI_IMG_MAX_DEFAULT);
			}
		}
		break;
	default:
		break; // keep previous setting
	}

	if (!stopper.is_group_execution_cancelled() && update) {
		SharedDataRead rlock(range->lock);
		SharedDataWrite wlock(multi->lock);
		(*multi)->minval = (*range)->first;
		(*multi)->maxval = (*range)->second;
		return true;
	} else {
		return false;
	}
}

void ViewerWindow::normTargetChanged(bool usecurrent)
{
	/* reset gui to current settings */
	int target = (normIButton->isChecked() ? 0 : 1);
	normMode m = (target == 0 ? normIMG : normGRAD);

	// update normModeBox
	normModeBox->setCurrentIndex(m);

	// update norm range spin boxes
	normModeSelected(m, true, usecurrent);
}

void ViewerWindow::normModeSelected(int mode, bool targetchange, bool usecurrent)
{
	normMode nm = static_cast<normMode>(mode);
	if (nm == NORM_FIXED && !targetchange) // user edits from currenty viewed values
		return;

	int target = (normIButton->isChecked() ? 0 : 1);

	if (!usecurrent) {
		BackgroundTaskPtr taskNormRange(new NormRangeTbb(
			(target == 0 ? image : gradient), 
			(target == 0 ? normIMGRange : normGRADRange), 
			nm, target, false));
		//QObject::connect(taskNormRange.get(), SIGNAL(finished(bool)), , SLOT());
		BackgroundTaskQueue::instance().push(taskNormRange);
		taskNormRange->wait();
	}

	double min;
	double max;
	if (target == 0) {
		SharedDataRead rlock(normIMGRange->lock);
		min = (*normIMGRange)->first;
		max = (*normIMGRange)->second;
	} else {
		SharedDataRead rlock(normGRADRange->lock);
		min = (*normGRADRange)->first;
		max = (*normGRADRange)->second;
	}

	// prevent signal loop
	normMinBox->blockSignals(true);
	normMaxBox->blockSignals(true);
	normMinBox->setValue(min);
	normMaxBox->setValue(max);
	normMinBox->blockSignals(false);
	normMaxBox->blockSignals(false);
}

void ViewerWindow::normModeFixed()
{
	if (normModeBox->currentIndex() != 2)
		normModeBox->setCurrentIndex(2);
}

void ViewerWindow::applyNormUserRange(bool update)
{
	int target = (normIButton->isChecked() ? 0 : 1);

	// set internal norm mode
	normMode &nm = (target == 0 ? normIMG : normGRAD);
	nm = static_cast<normMode>(normModeBox->currentIndex());

	// set internal range
	if (target == 0) {
		SharedDataWrite wlock(normIMGRange->lock);
		(*normIMGRange)->first = normMinBox->value();
		(*normIMGRange)->second = normMaxBox->value();
	} else {
		SharedDataWrite wlock(normGRADRange->lock);
		(*normGRADRange)->first = normMinBox->value();
		(*normGRADRange)->second = normMaxBox->value();
	}

	// if available, overwrite with more precise values than in the spin boxes.
	BackgroundTaskPtr taskNormRange(new NormRangeTbb(
		(target == 0 ? image : gradient), 
		(target == 0 ? normIMGRange : normGRADRange), 
		(target == 0 ? normIMG : normGRAD), target, true));
	//QObject::connect(taskNormRange.get(), SIGNAL(finished(bool)), , SLOT());
	BackgroundTaskQueue::instance().push(taskNormRange);
	taskNormRange->wait();

	if (update) {
		// re-initialize gui (duplication from applyROI())
		if (target == 0) {
			viewIMG->rebuild(-1);
			/* empty cache */
			bands[IMG].assign((*image)->size(), NULL);
		} else {
			viewGRAD->rebuild(-1);
			/* empty cache */
			bands[GRAD].assign((*gradient)->size(), NULL);
		}
		updateBand();
	}
}

void ViewerWindow::clampNormUserRange()
{
	// make sure internal settings are consistent
	applyNormUserRange(false);

	int target = (normIButton->isChecked() ? 0 : 1);

	/* if image is changed, change full image. for gradient, we cannot preserve
		the gradient over ROI or illuminant changes, so it remains a local change */
	if (target == 0) {
		(*full_image)->minval = (*image)->minval;
		(*full_image)->maxval = (*image)->maxval;
		(*full_image)->clamp();
		if (roi.width > 0)
			applyROI(false);

		BackgroundTaskPtr taskRgb(new ViewerWindow::RgbTbb(
			full_image, mat_vec3f_ptr(new SharedData<cv::Mat_<cv::Vec3f> >(new cv::Mat_<cv::Vec3f>)), full_rgb_temp));
		//QObject::connect(taskRgb.get(), SIGNAL(finished(bool)), this, SLOT(updateRGB(bool)));
		BackgroundTaskQueue::instance().push(taskRgb);
		updateRGB(taskRgb->wait());
	} else {
		(*gradient)->clamp();
		viewGRAD->rebuild(-1);
		/* empty cache */
		bands[GRAD].assign((*gradient)->size(), NULL);
		updateBand();
	}
}

void ViewerWindow::initGraphsegUI()
{
	graphsegSourceBox->addItem("Image", 0);
	graphsegSourceBox->addItem("Gradient", 1); // TODO PCA
	graphsegSourceBox->addItem("Shown Band", 2);
	graphsegSourceBox->setCurrentIndex(0);

	graphsegSimilarityBox->addItem("Manhattan distance (L1)", vole::MANHATTAN);
	graphsegSimilarityBox->addItem("Euclidean distance (L2)", vole::EUCLIDEAN);
	graphsegSimilarityBox->addItem(QString::fromUtf8("Chebyshev distance (L∞)"), vole::CHEBYSHEV);
	graphsegSimilarityBox->addItem("Spectral Angle", vole::MOD_SPEC_ANGLE);
	graphsegSimilarityBox->addItem("Spectral Information Divergence", vole::SPEC_INF_DIV);
	graphsegSimilarityBox->addItem("SID+SAM I", vole::SIDSAM1);
	graphsegSimilarityBox->addItem("SID+SAM II", vole::SIDSAM2);
	graphsegSimilarityBox->addItem("Normalized L2", vole::NORM_L2);
	graphsegSimilarityBox->setCurrentIndex(3);

	graphsegAlgoBox->addItem("Kruskal", vole::KRUSKAL);
	graphsegAlgoBox->addItem("Prim", vole::PRIM);
	graphsegAlgoBox->addItem("Power Watershed q=2", vole::WATERSHED2);
	graphsegAlgoBox->setCurrentIndex(2);

	connect(graphsegButton, SIGNAL(toggled(bool)),
			graphsegWidget, SLOT(setVisible(bool)));
	connect(graphsegButton, SIGNAL(toggled(bool)),
			bandView, SLOT(toggleSeedMode(bool)));
	connect(graphsegGoButton, SIGNAL(clicked()),
			this, SLOT(startGraphseg()));
	connect(this, SIGNAL(seedingDone(bool)),
			graphsegButton, SLOT(setChecked(bool)));
}

void ViewerWindow::runGraphseg(const multi_img& input,
							   const vole::GraphSegConfig &config)
{
	vole::GraphSeg seg(config);
	multi_img::Mask result;
	result = seg.execute(input, bandView->seedMap);

	/* add segmentation to current labeling */
	bandView->alterLabel(result, false);

	emit seedingDone();
}

void ViewerWindow::startGraphseg()
{
	vole::GraphSegConfig conf("graphseg");
	conf.algo = (vole::graphsegalg)
				graphsegAlgoBox->itemData(graphsegAlgoBox->currentIndex())
				.value<int>();
	conf.similarity.measure = (vole::similarity_fun)
	      graphsegSimilarityBox->itemData(graphsegSimilarityBox->currentIndex())
	      .value<int>();
#ifdef WITH_EDGE_DETECT
	conf.som_similarity = false;
#endif
	conf.geodesic = graphsegGeodCheck->isChecked();
	conf.multi_seed = false;
	int src = graphsegSourceBox->itemData(graphsegSourceBox->currentIndex())
								 .value<int>();
	if (src == 0) {
		runGraphseg(**image, conf);
	} else if (src == 1) {
		runGraphseg(**gradient, conf);
	} else {	// currently shown band, construct from selection in viewport
		int band = activeViewer->getViewport()->selection;
		const multi_img *img = activeViewer->getImage();
		multi_img i((*img)[band], img->minval, img->maxval);
		runGraphseg(i, conf);
	}
}

#ifdef WITH_SEG_MEANSHIFT
void ViewerWindow::initUnsupervisedSegUI()
{
	usMethodBox->addItem("Meanshift", 0);
#ifdef WITH_SEG_MEDIANSHIFT
	usMethodBox->addItem("Medianshift", 1);
#endif
#ifdef WITH_SEG_PROBSHIFT
	usMethodBox->addItem("Probabilistic Shift", 2);
#endif
	usMethodChanged(0); // set default state

	usInitMethodBox->addItem("all", vole::ALL);
	usInitMethodBox->addItem("jump", vole::JUMP);
	usInitMethodBox->addItem("percent", vole::PERCENT);
	usInitMethodChanged(0);

	usBandwidthBox->addItem("adaptive");
	usBandwidthBox->addItem("fixed");
	usBandwidthMethodChanged("adaptive");

	usBandsSpinBox->setValue((*full_image)->size());
	usBandsSpinBox->setMaximum((*full_image)->size());

	// we do not expose the density estimation functionality
	usInitWidget->hide();
	// we also do not expose options exclusive to unavailable methods
#ifndef WITH_SEG_MEDIANSHIFT
	usSkipPropWidget->hide();
#endif
#ifndef WITH_SEG_PROBSHIFT
	usSpectralWidget->hide();
	usMSPPWidget->hide();
#endif

	usInitJumpWidget->hide();
	usInitPercentWidget->hide();
	usFoundKLWidget->hide();
	usProgressWidget->hide();

	connect(usGoButton, SIGNAL(clicked()),
			this, SLOT(startUnsupervisedSeg()));
	connect(usFindKLGoButton, SIGNAL(clicked()),
			this, SLOT(startFindKL()));
	connect(usCancelButton, SIGNAL(clicked()),
			this, SLOT(unsupervisedSegCancelled()));

	connect(usMethodBox, SIGNAL(currentIndexChanged(int)),
			this, SLOT(usMethodChanged(int)));

	connect(usLshCheckBox, SIGNAL(toggled(bool)),
			usLshWidget, SLOT(setEnabled(bool)));

	connect(usBandwidthBox, SIGNAL(currentIndexChanged(const QString&)),
			this, SLOT(usBandwidthMethodChanged(const QString&)));

	connect(usInitMethodBox, SIGNAL(currentIndexChanged(int)),
			this, SLOT(usInitMethodChanged(int)));

	connect(usSpectralCheckBox, SIGNAL(toggled(bool)),
			usSpectralConvCheckBox, SLOT(setEnabled(bool)));
	connect(usSpectralCheckBox, SIGNAL(toggled(bool)),
			usSpectralMinMaxWidget, SLOT(setEnabled(bool)));

	/// pull default values from temporary instance of config class
	vole::MeanShiftConfig def;
	usKSpinBox->setValue(def.K);
	usLSpinBox->setValue(def.L);
	/// TODO: random seed box
	usPilotKSpinBox->setValue(def.k);
	usInitMethodBox->setCurrentIndex(
			usInitMethodBox->findData(def.starting));
	usInitJumpBox->setValue(def.jump);
	usFixedBWSpinBox->setValue(def.bandwidth);
	usFindKLKMinBox->setValue(def.Kmin);
	usFindKLKStepBox->setValue(def.Kjump);
	usFindKLEpsilonBox->setValue(def.epsilon);

#ifdef WITH_SEG_PROBSHIFT
	vole::ProbShiftConfig def_ps;
	usProbShiftMSPPAlphaSpinBox->setValue(def_ps.msBwFactor);
#endif
}

void ViewerWindow::usBandwidthMethodChanged(const QString &current) {
	if (current == "fixed") {
		usAdaptiveBWWidget->hide();
		usFixedBWWidget->show();
	} else if (current == "adaptive") {
		usFixedBWWidget->hide();
		usAdaptiveBWWidget->show();
	} else {
		assert(0);
	}
}

void ViewerWindow::unsupervisedSegCancelled() {
	usCancelButton->setDisabled(true);
	usCancelButton->setText("Please wait...");
	/// runner->terminate() will be called by the Cancel button
}

void ViewerWindow::startFindKL()
{
	startUnsupervisedSeg(true);
}

void ViewerWindow::startUnsupervisedSeg(bool findKL)
{
	// allow only one runner at a time (UI enforces that)
	assert(usRunner == NULL);
	usRunner = new CommandRunner();

	int method = usMethodBox->itemData(usMethodBox->currentIndex()).value<int>();

	if (findKL) { // run MeanShift::findKL()
		usRunner->cmd = new vole::MeanShiftShell();
		vole::MeanShiftConfig &config = ((vole::MeanShiftShell *) usRunner->cmd)->config;

		config.batch = true;
		config.findKL = true;
		config.k = usPilotKSpinBox->value();
		config.K = usFindKLKmaxBox->value();
		config.L = usFindKLLmaxBox->value();
		config.Kmin = usFindKLKMinBox->value();
		config.Kjump = usFindKLKStepBox->value();
		config.epsilon = usFindKLEpsilonBox->value();
	} else if (method == 0) { // Meanshift
		usRunner->cmd = new vole::MeanShiftShell();
		vole::MeanShiftConfig &config = ((vole::MeanShiftShell *) usRunner->cmd)->config;

		// fixed settings
		config.batch = true;

		config.use_LSH = usLshCheckBox->isChecked();
		config.K = usKSpinBox->value();
		config.L = usLSpinBox->value();

		config.starting = (vole::ms_sampling) usInitMethodBox->itemData(usInitMethodBox->currentIndex()).value<int>();
		config.percent = usInitPercentBox->value();
		config.jump = usInitJumpBox->value();
		config.k = usPilotKSpinBox->value();

		if (usBandwidthBox->currentText() == "fixed") {
			config.bandwidth = usFixedBWSpinBox->value();
		} else {
			config.bandwidth = 0;
		}
#ifdef WITH_SEG_MEDIANSHIFT
	} else if (method == 1) { // Medianshift
		usRunner->cmd = new vole::MedianShiftShell();
		vole::MedianShiftConfig &config = ((vole::MedianShiftShell *) usRunner->cmd)->config;

		config.K = usKSpinBox->value();
		config.L = usLSpinBox->value();
		config.k = usPilotKSpinBox->value();
		config.skipprop = usSkipPropCheckBox->isChecked();
#endif
#ifdef WITH_SEG_PROBSHIFT
	} else { // Probabilistic Shift
		usRunner->cmd = new vole::ProbShiftShell();
		vole::ProbShiftConfig &config = ((vole::ProbShiftShell *) usRunner->cmd)->config;

		config.useLSH = usLshCheckBox->isChecked();
		config.lshK = usKSpinBox->value();
		config.lshL = usLSpinBox->value();

		config.useSpectral = usSpectralCheckBox->isChecked();
		config.useConverged = usSpectralConvCheckBox->isChecked();
		config.minClusts = usSpectralMinBox->value();
		config.maxClusts = usSpectralMaxBox->value();
		config.useMeanShift = usProbShiftMSPPCheckBox->isChecked();
		config.msBwFactor = usProbShiftMSPPAlphaSpinBox->value();
#endif
	}

	// connect runner with progress bar, cancel button and finish-slot
	connect(usRunner, SIGNAL(progressChanged(int)), usProgressBar, SLOT(setValue(int)));
	connect(usCancelButton, SIGNAL(clicked()), usRunner, SLOT(terminate()));

	qRegisterMetaType< std::map<std::string, boost::any> >("std::map<std::string, boost::any>");
	connect(usRunner, SIGNAL(success(std::map<std::string,boost::any>)), this, SLOT(segmentationApply(std::map<std::string,boost::any>)));
	connect(usRunner, SIGNAL(finished()), this, SLOT(segmentationFinished()));

	usProgressWidget->show();
	usSettingsWidget->setDisabled(true);

	// prepare input image
	boost::shared_ptr<multi_img> input(new multi_img(**full_image, roi)); // image data is not copied
	int numbands = usBandsSpinBox->value();
	bool gradient = usGradientCheckBox->isChecked();

	if (numbands > 0 && numbands < (int) input->size()) {
		boost::shared_ptr<multi_img> input_tmp(new multi_img(input->spec_rescale(numbands)));
		input = input_tmp;
	}

	if (gradient) {
		// copy needed here
		multi_img loginput(*input);
		loginput.apply_logarithm();
		input = boost::shared_ptr<multi_img>(new multi_img(loginput.spec_gradient()));
	}

	usRunner->input["multi_img"] = input;

	usRunner->start();
}

void ViewerWindow::segmentationFinished() {
	if (usRunner->abort) {
		// restore Cancel button
		usCancelButton->setEnabled(true);
		usCancelButton->setText("Cancel");
	}

	// hide progress, re-enable settings
	usProgressWidget->hide();
	usSettingsWidget->setEnabled(true);

	/// clean up runner
	delete usRunner;
	usRunner = NULL;
}

void ViewerWindow::segmentationApply(std::map<std::string, boost::any> output) {
	if (output.count("labels")) {
		boost::shared_ptr<cv::Mat1s> labelMask = boost::any_cast< boost::shared_ptr<cv::Mat1s> >(output["labels"]);
		setLabels(*labelMask);
	}

	if (output.count("findKL.K") && output.count("findKL.L")) {
		int foundK = boost::any_cast<int>(output["findKL.K"]);
		int foundL = boost::any_cast<int>(output["findKL.L"]);
		usFoundKLLabel->setText(QString("Found values: K=%1 L=%2").arg(foundK).arg(foundL));
		usFoundKLWidget->show();
	}
}
#else // method stubs as using define in header does not work (moc problem?)
void ViewerWindow::startUnsupervisedSeg(bool findKL) {}
void ViewerWindow::startFindKL() {}
void ViewerWindow::segmentationFinished() {}
void ViewerWindow::segmentationApply(std::map<std::string, boost::any>) {}
void ViewerWindow::usMethodChanged(int idx) {}
void ViewerWindow::usInitMethodChanged(int idx) {}
void ViewerWindow::usBandwidthMethodChanged(const QString &current) {}
void ViewerWindow::unsupervisedSegCancelled() {}
#endif // WITH_SEG_MEANSHIFT

void ViewerWindow::setActive(int id)
{
	activeViewer = viewers[id];
}

void ViewerWindow::labelmask(bool negative)
{
	emit alterLabel(activeViewer->getMask(), negative);
	for (size_t i = 0; i < viewers.size(); ++i)
		viewers[i]->rebuild();
}

void ViewerWindow::newOverlay()
{
	emit drawOverlay(activeViewer->getMask());
}

void ViewerWindow::reshapeDock(bool floating)
{
	if (!floating || (*image)->height == 0)
		return;

	float src_aspect = (*image)->width/(float)(*image)->height;
	float dest_aspect = bandView->width()/(float)bandView->height();
	// we force the dock aspect ratio to fit band image aspect ratio.
	// this is not 100% correct
	if (src_aspect > dest_aspect) {
		bandDock->resize(bandDock->width(), bandDock->width()/src_aspect);
	} else
		bandDock->resize(bandDock->height()*src_aspect, bandDock->height());
}

QIcon ViewerWindow::colorIcon(const QColor &color)
{
	QPixmap pm(32, 32);
	pm.fill(color);
	return QIcon(pm);
}

void ViewerWindow::buildIlluminant(int temp)
{
	assert(temp > 0);
	Illuminant il(temp);
	std::vector<multi_img::Value> cf;
	il.calcWeight((*full_image)->meta[0].center,
				  (*full_image)->meta[(*full_image)->size()-1].center);
	cf = (*full_image)->getIllumCoeff(il);
	illuminants[temp] = make_pair(il, cf);
}

const Illuminant & ViewerWindow::getIlluminant(int temp)
{
	assert(temp > 0);
	Illum_map::iterator i = illuminants.find(temp);
	if (i != illuminants.end())
		return i->second.first;

	buildIlluminant(temp);
	return illuminants[temp].first;
}

const std::vector<multi_img::Value> & ViewerWindow::getIlluminantC(int temp)
{
	assert(temp > 0);
	Illum_map::iterator i = illuminants.find(temp);
	if (i != illuminants.end())
		return i->second.second;

	buildIlluminant(temp);
	return illuminants[temp].second;
}

void ViewerWindow::setI1(int index) {
	int i1 = i1Box->itemData(index).value<int>();
	if (i1 > 0) {
		i1Check->setEnabled(true);
		if (i1Check->isChecked())
			viewIMG->setIlluminant(&getIlluminantC(i1), false);
	} else {
		i1Check->setEnabled(false);
		viewIMG->setIlluminant(NULL, false);
	}
}

void ViewerWindow::setI1Visible(bool visible)
{
	if (visible) {
		int i1 = i1Box->itemData(i1Box->currentIndex()).value<int>();
		viewIMG->setIlluminant(&getIlluminantC(i1), false);
	} else {
		viewIMG->setIlluminant(NULL, false);
	}
}

void ViewerWindow::loadLabeling(QString filename)
{
	IOGui io("Labeling Image File", "labeling image", this);
	cv::Mat input = io.readFile(filename,
								-1, (*full_image)->height, (*full_image)->width);
	if (input.empty())
		return;

	vole::Labeling labeling(input, false);

	// if the user is operating within ROI, apply it to labeling as well */
	if (roi != cv::Rect(0, 0, (*full_image)->width, (*full_image)->height))
		labeling.setLabels(labeling.getLabels()(roi));

	setLabels(labeling);
}

void ViewerWindow::loadSeeds()
{
	IOGui io("Seed Image File", "seed image", this);
	cv::Mat1s seeding = io.readFile(QString(),
									0, (*full_image)->height, (*full_image)->width);
	if (seeding.empty())
		return;

	bandView->seedMap = seeding;

	// now make sure we are in seed mode
	if (graphsegButton->isChecked()) {
		bandView->refresh();
	} else {
		graphsegButton->toggle();
	}
}

void ViewerWindow::saveLabeling()
{
	vole::Labeling labeling(bandView->labels);
	cv::Mat3b output = labeling.bgr();

	IOGui io("Labeling As Image File", "labeling image", this);
	io.writeFile(QString(), output);
}

void ViewerWindow::screenshot()
{
	// grabWindow reads from the display server, so GL parts are not missing
	QPixmap shot = QPixmap::grabWindow(this->winId());

	// we use OpenCV so the user can expect the same data type support
	cv::Mat output = vole::QImage2Mat(shot.toImage());

	IOGui io("Screenshot File", "screenshot", this);
	io.writeFile(QString(), output);
}

void ViewerWindow::ROITrigger()
{
	mainStack->setCurrentIndex(1);
}

void ViewerWindow::ROIDecision(QAbstractButton *sender)
{
	QDialogButtonBox::ButtonRole role = roiButtons->buttonRole(sender);
	bool apply = (role == QDialogButtonBox::ApplyRole);

	if (role == QDialogButtonBox::ResetRole) {
		if (roi.width > 1)
			roiView->roi = QRect(roi.x, roi.y, roi.width, roi.height);
		else
			roiView->roi = QRect(0, 0, (*full_image)->width, (*full_image)->height);
		roiView->update();
	} else if (role == QDialogButtonBox::RejectRole) {
		if (roi.width < 2) {
			// implicit accept full image if no interest in ROI
			roiView->roi = QRect(0, 0, (*full_image)->width, (*full_image)->height);
			apply = true;
		} else {
			// just get out of the view
			mainStack->setCurrentIndex(0);
		}
	}

	if (apply) {
		const QRect &r = roiView->roi;
		roi = cv::Rect(r.x(), r.y(), r.width(), r.height());
		applyROI(true);
	}
}

void ViewerWindow::ROISelection(const QRect &roi)
{
	QString title("<b>Select Region of Interest:</b> %1.%2 - %3.%4 (%5x%6)");
	title = title.arg(roi.x()).arg(roi.y()).arg(roi.right()).arg(roi.bottom())
			.arg(roi.width()).arg(roi.height());
	roiTitle->setText(title);
}

void ViewerWindow::openContextMenu()
{
	delete contextMenu;
	contextMenu = createPopupMenu();
	contextMenu->exec(QCursor::pos());
}

void ViewerWindow::changeEvent(QEvent *e)
{
	QMainWindow::changeEvent(e);
	switch (e->type()) {
	case QEvent::LanguageChange:
		retranslateUi(this);
		break;
	default:
		break;
	}
}
