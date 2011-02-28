/*	
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "viewerwindow.h"

#include <labeling.h>
#include <qtopencv.h>

#include <highgui.h>

#include <QPainter>
#include <QIcon>
#include <QSignalMapper>
#include <QMessageBox>
#include <QFileDialog>
#include <iostream>

ViewerWindow::ViewerWindow(multi_img *image, QWidget *parent)
	: QMainWindow(parent),
	  full_image(image), image(NULL), gradient(NULL),
	  activeViewer(0), normIMG(NORM_OBSERVED), normGRAD(NORM_OBSERVED),
	  contextMenu(NULL)
{
	initUI();
	roiView->roi = QRect(0, 0, full_image->width, full_image->height);

	// default to full image if small enough
	if (full_image->width*full_image->height < 513*513) {
		roi = cv::Rect(0, 0, full_image->width, full_image->height);
		applyROI();
	}
}

void ViewerWindow::applyROI()
{
	/// first: set up new working images
	delete image;
	delete gradient;
	image = new multi_img(*full_image, roi);

	multi_img log(*image);
	log.apply_logarithm();
	gradient = new multi_img(log.spec_gradient());

	// calculate new norm ranges inside ROI
	for (int i = 0; i < 2; ++i) {
		setNormRange(i);
		updateImageRange(i);
	}
	// gui update (if norm ranges changed)
	normTargetChanged();

	/* set labeling, and label colors (depend on ROI size) */
	cv::Mat1s labels(image->height, image->width, (short)0);
	viewIMG->labels = viewGRAD->labels = bandView->labels = labels;
	if (labelColors.empty())
		setLabelColors(vole::Labeling::colors(2, true));

	/// second: re-initialize gui
	/* empty caches */
	ibands.assign(image->size(), NULL);
	gbands.assign(gradient->size(), NULL);

	updateBand();
	updateRGB(false);

	/* do setImage() last */
	viewIMG->setImage(image);
	viewGRAD->setImage(gradient, true);

	bandDock->setEnabled(true);
	rgbDock->setEnabled(true);
	mainStack->setCurrentIndex(0);
}

void ViewerWindow::initUI()
{
	/* GUI elements */
	setupUi(this);
	initGraphsegUI();
	initIlluminantUI();
	initNormalizationUI();

	/* more manual work to get GUI in proper shape */
	graphsegWidget->hide();

	bandDock->setDisabled(true);
	rgbDock->setDisabled(true);

	activeViewer = viewIMG;
	viewIMG->setActive();

	// dock arrangement
	tabifyDockWidget(labelDock, illumDock);
	tabifyDockWidget(labelDock, normDock);

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

	// for self-activation of viewports
	QSignalMapper *vpmap = new QSignalMapper(this);
	vpmap->setMapping(viewIMG->getViewport(), 0);
	vpmap->setMapping(viewGRAD->getViewport(), 1);
	connect(vpmap, SIGNAL(mapped(int)),
			this, SLOT(setActive(int)));

	for (int i = 0; i < 2; ++i)
	{
		multi_img_viewer *v = (i == 1 ? viewGRAD : viewIMG);
		multi_img_viewer *v2 = (i == 0 ? viewGRAD : viewIMG);
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

		connect(bandView, SIGNAL(pixelOverlay(int,int)),
				v, SLOT(overlay(int,int)));

		connect(vp, SIGNAL(activated()),
				vpmap, SLOT(map()));
		// connect same signal also to the _other_ viewport
		connect(vp, SIGNAL(activated()),
				v2, SLOT(setInactive()));

		connect(vp, SIGNAL(bandSelected(int, bool)),
				this, SLOT(selectBand(int, bool)));

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

	updateRGB(true);
}

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
	assert(labeling().rows == image->height && labeling().cols == image->width);
	/* note: always update labels before updating label colors, for the case
	   that there are less colors available than used in previous labeling */
	cv::Mat1s labels = labeling()(roi);
	viewIMG->labels = viewGRAD->labels = bandView->labels = labels;

	bool updated = setLabelColors(labeling.colors());
	if (!updated) {
		viewIMG->rebuild();
		viewGRAD->rebuild();
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

const QPixmap* ViewerWindow::getBand(int dim, bool grad)
{
	// select variables according to which set is asked for
	std::vector<QPixmap*> &v = (grad ? gbands : ibands);

	if (!v[dim]) {
		const multi_img *m = (grad ? gradient : image);
		// create here
		QImage img = m->export_qt(dim);
		v[dim] = new QPixmap(QPixmap::fromImage(img));
	}
	return v[dim];
}

void ViewerWindow::updateBand()
{
	int band = activeViewer->getViewport()->selection;
	selectBand(band, activeViewer == viewGRAD);
	bandView->update();
}

void ViewerWindow::selectBand(int dim, bool grad)
{
	bandView->setPixmap(*getBand(dim, grad));
	const multi_img *m = (grad ? gradient : image);
	std::string banddesc = m->meta[dim].str();
	QString title;
	if (banddesc.empty())
		title = QString("%1 Band #%2")
			.arg(grad ? "Gradient" : "Image")
			.arg(dim+1);
	else
		title = QString("%1 Band %2")
			.arg(grad ? "Gradient" : "Image")
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
		full_image->apply_illuminant(il, true);
	}

	/* add new illuminant */
	if (i2 != 0) {
		const Illuminant &il = getIlluminant(i2);
		full_image->apply_illuminant(il);
	}

	viewIMG->setIlluminant((i2 ? &getIlluminantC(i2) : NULL), true);
	/* rebuild  */
	if (roi.width > 0)
		applyROI();
	updateRGB(true);

	/* reflect change in our own gui (will propagate to viewIMG) */
	i1Box->setCurrentIndex(i2Box->currentIndex());
}

void ViewerWindow::updateRGB(bool full)
{
	if (full || full_rgb.isNull()) {
		cv::Mat_<cv::Vec3f> bgrmat = full_image->bgr();
		QImage img(bgrmat.cols, bgrmat.rows, QImage::Format_ARGB32);
		for (int y = 0; y < bgrmat.rows; ++y) {
			cv::Vec3f *row = bgrmat[y];
			QRgb *destrow = (QRgb*)img.scanLine(y);
			for (int x = 0; x < bgrmat.cols; ++x) {
				cv::Vec3f &c = row[x];
				destrow[x] = qRgb(c[2]*255., c[1]*255., c[0]*255.);
			}
		}
		full_rgb = QPixmap::fromImage(img);
	}
	if (full) {
		roiView->setPixmap(full_rgb);
		roiView->update();
	}
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
	connect(normApplyButton, SIGNAL(clicked()),
			this, SLOT(applyNormUserRange()));
	connect(normClampButton, SIGNAL(clicked()),
			this, SLOT(clampNormUserRange()));
}

std::pair<multi_img::Value, multi_img::Value>
ViewerWindow::getNormRange(normMode mode, int target,
						   std::pair<multi_img::Value, multi_img::Value> cur)
{
	const multi_img *img = (target == 0 ? image : gradient);
	std::pair<multi_img::Value, multi_img::Value> ret;
	switch (mode) {
	case NORM_OBSERVED:
		ret = img->data_range();
		break;
	case NORM_THEORETICAL:
		// hack!
		if (target == 0)
			ret = make_pair(MULTI_IMG_MIN_DEFAULT, MULTI_IMG_MAX_DEFAULT);
		else
			ret = make_pair(-log(MULTI_IMG_MAX_DEFAULT), log(MULTI_IMG_MAX_DEFAULT));
		break;
	default:
		ret = cur; // keep previous setting
	}
	return ret;
}

void ViewerWindow::setNormRange(int target)
{
	// select respective normalization mode, range variable and image
	normMode m = (target == 0 ? normIMG : normGRAD);
	std::pair<multi_img::Value, multi_img::Value> &r =
			(target == 0 ? normIMGRange : normGRADRange);

	// set range according to mode
	r = getNormRange(m, target, r);
}

void ViewerWindow::updateImageRange(int target)
{
	const std::pair<multi_img::Value, multi_img::Value> &r =
			(target == 0 ? normIMGRange : normGRADRange);
	multi_img *i = (target == 0 ? image : gradient);
	i->minval = r.first;
	i->maxval = r.second;
}

void ViewerWindow::normTargetChanged()
{
	/* reset gui to current settings */
	int target = (normIButton->isChecked() ? 0 : 1);
	normMode m = (target == 0 ? normIMG : normGRAD);

	// update normModeBox
	normModeBox->setCurrentIndex(m);

	// update norm range spin boxes
	normModeSelected(m);
}

void ViewerWindow::normModeSelected(int mode)
{
	int target = (normIButton->isChecked() ? 0 : 1);
	const std::pair<multi_img::Value, multi_img::Value> &cur =
			(target == 0 ? normIMGRange : normGRADRange);

	normMode nm = static_cast<normMode>(mode);
	std::pair<multi_img::Value, multi_img::Value> r
			= getNormRange(nm, target, cur);
	normMinBox->setValue(r.first);
	normMaxBox->setValue(r.second);

	normMinBox->setReadOnly(nm != NORM_FIXED);
	normMaxBox->setReadOnly(nm != NORM_FIXED);
}

void ViewerWindow::applyNormUserRange(bool update)
{
	int target = (normIButton->isChecked() ? 0 : 1);

	// set internal norm mode
	normMode &nm = (target == 0 ? normIMG : normGRAD);
	nm = static_cast<normMode>(normModeBox->currentIndex());

	// set internal range
	std::pair<multi_img::Value, multi_img::Value> &r =
				(target == 0 ? normIMGRange : normGRADRange);
	r.first = normMinBox->value();
	r.second = normMaxBox->value();

	// if available, overwrite with more precise values than in the spin boxes.
	setNormRange(target);

	// update image
	updateImageRange(target);

	if (update) {
		// re-initialize gui (duplication from applyROI())
		if (target == 0) {
			viewIMG->rebuild(-1);
			/* empty cache */
			ibands.assign(image->size(), NULL);
		} else {
			viewGRAD->rebuild(-1);
			/* empty cache */
			gbands.assign(gradient->size(), NULL);
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
		full_image->minval = image->minval;
		full_image->maxval = image->maxval;
		full_image->clamp();
		if (roi.width > 0)
			applyROI();
		updateRGB(true);
	} else {
		gradient->clamp();
		viewGRAD->rebuild(-1);
		/* empty cache */
		gbands.assign(gradient->size(), NULL);
		updateBand();
	}
}

void ViewerWindow::initGraphsegUI()
{
	graphsegSourceBox->addItem("Image", 0);
	graphsegSourceBox->addItem("Gradient", 1);
	graphsegSourceBox->addItem("Shown Band", 2);
	graphsegAlgoBox->addItem("Kruskal", vole::KRUSKAL);
	graphsegAlgoBox->addItem("Prim", vole::PRIM);
	graphsegAlgoBox->addItem("Power Watershed q=2", vole::WATERSHED2);
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
	conf.distance = vole::CHEBYSHEV; // as in Couprie et al., PAMI 2010
	conf.geodesic = graphsegGeodCheck->isChecked();
	conf.multi_seed = false;
	int src = graphsegSourceBox->itemData(graphsegSourceBox->currentIndex())
								 .value<int>();
	if (src == 0) {
		runGraphseg(*image, conf);
	} else if (src == 1) {
		runGraphseg(*gradient, conf);
	} else {	// currently shown band, construct from selection in viewport
		int band = activeViewer->getViewport()->selection;
		const multi_img *img = activeViewer->image;
		multi_img i((*img)[band], img->minval, img->maxval);
		runGraphseg(i, conf);
	}
}

void ViewerWindow::setActive(int id)
{
	activeViewer = (id == 1 ? viewGRAD : viewIMG);
}

void ViewerWindow::labelmask(bool negative)
{
	emit alterLabel(activeViewer->getMask(), negative);
	viewIMG->rebuild();
	viewGRAD->rebuild();
}

void ViewerWindow::newOverlay()
{
	emit drawOverlay(activeViewer->getMask());
}

void ViewerWindow::reshapeDock(bool floating)
{
	if (!floating || !image)
		return;

	float src_aspect = image->width/(float)image->height;
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
	il.calcWeight(full_image->meta[0].center,
				  full_image->meta[full_image->size()-1].center);
	cf = full_image->getIllumCoeff(il);
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

void ViewerWindow::loadLabeling()
{
	std::string filename = QFileDialog::getOpenFileName
						   (this, "Open Labeling Image File").toStdString();
	if (filename.empty())
		return;
	vole::Labeling labeling(filename, false);
	if (labeling().empty()) {
		QMessageBox::critical(this, "Error loading labels",
						"The labeling image could not be read."
						"\nSupported are all image formats readable by OpenCV.");
		return;
	}
	if (labeling().rows != full_image->height || labeling().cols != full_image->width) {
		QMessageBox::critical(this, "Labeling image does not match",
				QString("The labeling image has wrong proportions."
						"\nIt has to be of size %1x%2 for this image.")
						.arg(full_image->width).arg(full_image->height));
		return;
	}

	setLabels(labeling);
}

void ViewerWindow::loadSeeds()
{
	std::string filename = QFileDialog::getOpenFileName
						   (this, "Open Seed Image File").toStdString();
	if (filename.empty())
		return;
	cv::Mat1s seeding = cv::imread(filename, 0);
	if (seeding.empty()) {
		QMessageBox::critical(this, "Error loading seeds",
						"The seed image could not be read."
						"\nSupported are all image formats readable by OpenCV.");
		return;
	}
	if (seeding.rows != full_image->height || seeding.cols != full_image->width) {
		QMessageBox::critical(this, "Seed image does not match",
				QString("The seed image has wrong proportions."
						"\nIt has to be of size %1x%2 for this image.")
						.arg(full_image->width).arg(full_image->height));
		return;
	}

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
	std::string filename = QFileDialog::getSaveFileName
						   (this, "Save Labeling as Image File").toStdString();
	if (filename.empty())
		return;
	vole::Labeling labeling(bandView->labels);
	cv::Mat3b output = labeling.bgr();
	bool success = cv::imwrite(filename, output);
	if (!success)
		QMessageBox::critical(this, QString("Could not write output file."),
							QString("See console for OpenCV error output."));
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
			roiView->roi = QRect(0, 0, full_image->width, full_image->height);
		roiView->update();
	} else if (role == QDialogButtonBox::RejectRole) {
		if (roi.width < 2) {
			// implicit accept full image if no interest in ROI
			roiView->roi = QRect(0, 0, full_image->width, full_image->height);
			apply = true;
		} else {
			// just get out of the view
			mainStack->setCurrentIndex(0);
		}
	}

	if (apply) {
		const QRect &r = roiView->roi;
		roi = cv::Rect(r.x(), r.y(), r.width(), r.height());
		applyROI();
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
