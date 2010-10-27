/*	
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "viewerwindow.h"

#include <graphseg_config.h>

#include <QPainter>
#include <QIcon>
#include <iostream>

ViewerWindow::ViewerWindow(multi_img *image, QWidget *parent)
	: QMainWindow(parent),
	  full_image(image), image(NULL), gradient(NULL),
	  activeViewer(viewIMG)
{
	init();
}

void ViewerWindow::applyROI()
{
	// first: set up new working images
	delete image;
	delete gradient;
	image = new multi_img(*full_image, roi);

	multi_img log(*image);
	log.apply_logarithm();
	gradient = new multi_img(log.spec_gradient());

	// second: re-initialize gui
	labels = cv::Mat_<uchar>(image->height, image->width, (uchar)0);

	bandView->labels = labels;
	viewIMG->labels = viewGRAD->labels = labels;

	/* update caches */
	ibands.assign(image->size(), NULL);
	gbands.assign(gradient->size(), NULL);

	selectBand(0, false);
	updateRGB(false);

	/* do setImage() last */
	viewIMG->setImage(image);
	viewGRAD->setImage(gradient, true);

	bandDock->setEnabled(true);
	rgbDock->setEnabled(true);
	mainStack->setCurrentIndex(0);
}

void ViewerWindow::init()
{
	setupUi(this);
	bandButton->hide();

	bandDock->setDisabled(true);
	rgbDock->setDisabled(true);

	/* setup labeling stuff first */
	labelColors << Qt::white // 0 is index for unlabeled
		<< Qt::green << Qt::red << Qt::cyan << Qt::magenta << Qt::blue;

	bandView->setLabelColors(labelColors);
	viewIMG->labelColors = viewGRAD->labelColors = &labelColors;
	createMarkers();

	graphsegWidget->hide();

	connect(bandDock, SIGNAL(visibilityChanged(bool)),
			bandButton, SLOT(setHidden(bool)));
	connect(bandButton, SIGNAL(clicked()),
			bandDock, SLOT(show()));

	connect(bandDock, SIGNAL(topLevelChanged(bool)),
			this, SLOT(reshapeDock(bool)));

	connect(markerSelector, SIGNAL(currentIndexChanged(int)),
			bandView, SLOT(changeLabel(int)));
	connect(clearButton, SIGNAL(clicked()),
			bandView, SLOT(clearLabelPixels()));

	initGraphsegUI();
	initIlluminantUI();

	// signals for ROI
	connect(roiButtons, SIGNAL(clicked(QAbstractButton*)),
			this, SLOT(roi_decision(QAbstractButton*)));
	connect(roiButton, SIGNAL(clicked()), this, SLOT(roi_trigger()));
	connect(roiView, SIGNAL(newSelection(QRect)),
			this, SLOT(roi_selection(QRect)));

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

	for (int i = 0; i < 2; ++i)
	{
		multi_img_viewer *v = (i == 1 ? viewGRAD : viewIMG);
		connect(applyButton, SIGNAL(clicked()),
				v, SLOT(rebuild()));
		connect(markButton, SIGNAL(toggled(bool)),
				v, SLOT(toggleLabeled(bool)));
		connect(nonmarkButton, SIGNAL(toggled(bool)),
				v, SLOT(toggleUnlabeled(bool)));
		connect(ignoreButton, SIGNAL(toggled(bool)),
				v, SLOT(toggleLabels(bool)));

		connect(bandView, SIGNAL(pixelOverlay(int,int)),
				v, SLOT(overlay(int,int)));

		const Viewport *vp = v->getViewport();

		connect(vp, SIGNAL(bandSelected(int, bool)),
				this, SLOT(selectBand(int, bool)));
		connect(vp, SIGNAL(activated(bool)),
				this, SLOT(setActive(bool)));
		connect(vp, SIGNAL(activated(bool)),
				(i ? viewIMG : viewGRAD), SLOT(setActive(bool)));

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
	viewIMG->setActive(false);

	updateRGB(true);

	roiView->roi = QRect(0, 0, full_image->width, full_image->height);
	// default to full image if small enough
	if (full_image->width*full_image->height < 513*513) {
		roi = cv::Rect(0, 0, full_image->width, full_image->height);
		applyROI();
	}
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
	/* rebuild spectral gradient and update other stuff */
	if (roi.width > 0)
	{
		applyROI();
		viewGRAD->rebuild();
		int band = activeViewer->getViewport()->selection;
		selectBand(band, activeViewer == viewGRAD);
		bandView->update();
	}
	updateRGB(true);

	/* reflect change in our own gui (will propagate to viewIMG) */
	i1Box->setCurrentIndex(i2Box->currentIndex());
}

void ViewerWindow::updateRGB(bool full)
{
	if (full || full_rgb.isNull()) {
		cv::Mat_<cv::Vec3f> rgbmat = full_image->rgb();
		QImage img(rgbmat.cols, rgbmat.rows, QImage::Format_ARGB32);
		for (int y = 0; y < rgbmat.rows; ++y) {
			cv::Vec3f *row = rgbmat[y];
			QRgb *destrow = (QRgb*)img.scanLine(y);
			for (int x = 0; x < rgbmat.cols; ++x) {
				cv::Vec3f &c = row[x];
				destrow[x] = qRgb(c[0]*255., c[1]*255., c[2]*255.);
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
	connect(bandView, SIGNAL(seedingDone(bool)),
			graphsegButton, SLOT(setChecked(bool)));
}

void ViewerWindow::startGraphseg()
{
	vole::GraphSegConfig conf("graphseg");
	conf.algo = (vole::graphsegalg)
				graphsegAlgoBox->itemData(graphsegAlgoBox->currentIndex())
				.value<int>();
	conf.distance = vole::MANHATTAN;
	conf.geodesic = graphsegGeodCheck->isEnabled();
	conf.multi_seed = false;
	int src = graphsegSourceBox->itemData(graphsegSourceBox->currentIndex())
								 .value<int>();
	if (src == 0) {
		bandView->startGraphseg(*image, conf);
	} else if (src == 1) {
		bandView->startGraphseg(*gradient, conf);
	} else {	// currently shown band, construct from selection in viewport
		int band = activeViewer->getViewport()->selection;
		const multi_img *img = activeViewer->image;
		multi_img i((*img)[band], img->minval, img->maxval);
		bandView->startGraphseg(i, conf);
	}
}

void ViewerWindow::setActive(bool gradient)
{
	activeViewer = (gradient ? viewGRAD : viewIMG);
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

void ViewerWindow::createMarkers()
{
	for (int i = 1; i < labelColors.size(); ++i) // 0 is index for unlabeled
	{
		markerSelector->addItem(colorIcon(labelColors[i]), "");
	}
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

void ViewerWindow::roi_trigger()
{
	mainStack->setCurrentIndex(1);
}

void ViewerWindow::roi_decision(QAbstractButton *sender)
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

void ViewerWindow::roi_selection(const QRect &roi)
{
	QString title("<b>Select Region of Interest:</b> %1.%2 - %3.%4 (%5x%6)");
	title = title.arg(roi.x()).arg(roi.y()).arg(roi.right()).arg(roi.bottom())
			.arg(roi.width()).arg(roi.height());
	roiTitle->setText(title);
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
