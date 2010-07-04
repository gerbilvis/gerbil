#include "viewerwindow.h"

#include <graphseg_config.h>

#include <QPainter>
#include <QIcon>
#include <iostream>

ViewerWindow::ViewerWindow(const multi_img &image, const multi_img &gradient, QWidget *parent)
	: QMainWindow(parent), activeViewer(0),
	  image(&image), image_illum(&image),
	  gradient(&gradient), gradient_illum(&gradient),
	  ibands(image.size(), NULL), gbands(gradient.size(), NULL),
	  labels(image.height, image.width, (uchar)0)
{
	init();
}

/* RGB HACK */
ViewerWindow::ViewerWindow(const multi_img &image, const multi_img &gradient, const char* rgbfile)
	: QMainWindow(NULL), activeViewer(0),
	  image(&image), image_illum(&image),
	  gradient(&gradient), gradient_illum(&gradient),
	  ibands(image.size(), NULL), gbands(gradient.size(), NULL),
	  labels(image.height, image.width, (uchar)0)
{
	init();
	updateRGB(true, rgbfile);
}

void ViewerWindow::init()
{
	setupUi(this);
	bandButton->hide();

	/* setup labeling stuff first */
	QVector<QColor> &labelcolors = bandView->markerColors;
	bandView->labels = labels;
	rgbView->labels = labels;

	createMarkers();
	selectBand(0, false);
	graphsegWidget->hide();

	updateRGB();

	/* setup viewers, do setImage() last */
	viewIMG->labels = viewGRAD->labels = labels;
	viewIMG->labelcolors = viewGRAD->labelcolors = &labelcolors;
	viewIMG->setImage(image);
	viewGRAD->setImage(gradient, true);
	viewIMG->setActive(false);

	/* signals & slots */
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

		connect(vp, SIGNAL(newOverlay()),
				this, SLOT(newOverlay()));

		connect(vp, SIGNAL(addSelection()),
				this, SLOT(addToLabel()));
		connect(vp, SIGNAL(remSelection()),
				this, SLOT(remFromLabel()));

		connect(bandView, SIGNAL(killHover()),
				vp, SLOT(killHover()));
	}
}

const QPixmap* ViewerWindow::getBand(int dim, bool grad)
{
	// select variables according to which set is asked for
	std::vector<QPixmap*> &v = (grad ? gbands : ibands);

	if (!v[dim]) {
		const multi_img *m = (grad ? gradient_illum : image_illum);
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
	/* start with a copy of original image */
	image_work = *image;

	/* apply illuminant here */
	int i1 = i1Box->itemData(i1Box->currentIndex()).value<int>();
	int i2 = i2Box->itemData(i2Box->currentIndex()).value<int>();
	if (i1 == i2)
		return;

	if (i1 != 0) {	// remove old illuminant
		const multi_img::Illuminant &il = getIlluminant(i1);
		image_work.apply_illuminant(il, true);
	}
	if (i2 != 0) {	// add new illuminant
		const multi_img::Illuminant &il = getIlluminant(i2);
		image_work.apply_illuminant(il);
	}

	/* rebuild spectral gradient */
	multi_img log(image_work);
	log.apply_logarithm();
	gradient_work = log.spec_gradient();

	/* set references accordingly */
	image_illum = &image_work;
	gradient_illum = &gradient_work;

	/* update caches */
	ibands.clear(); ibands.resize(image->size(), NULL);
	gbands.clear(); gbands.resize(image->size(), NULL);
	updateRGB();
	int band = (activeViewer == 0 ? viewIMG->getViewport()->selection
								  : viewGRAD->getViewport()->selection);
	selectBand(band, activeViewer == 1);

	/* only gradient view shows new image.
	   image view has its own way of handling illumination */
	viewGRAD->setImage(gradient_illum, true);

	/* reflect change in our own gui (will propagate to viewIMG) */
	i1Box->setCurrentIndex(i2Box->currentIndex());
}

void ViewerWindow::updateRGB(bool hack, const char *rgbfile)
{
	/* RGB HACK */
	if (hack) {
		QPixmap source(rgbfile);
		if (source.width() != image->width || source.height() != image->height)
			rgb = source.scaled(image->width, image->height);
		else
			rgb = source;
		rgbView->setPixmap(rgb);
		return;
	}
	
	cv::Mat3f rgbmat = image_illum->rgb();
	QImage img(rgbmat.cols, rgbmat.rows, QImage::Format_ARGB32);
	for (int y = 0; y < rgbmat.rows; ++y) {
		cv::Vec3f *row = rgbmat[y];
		for (int x = 0; x < rgbmat.cols; ++x) {
			cv::Vec3f &c = row[x];
			img.setPixel(x, y, qRgb(c[2]*255., c[1]*255., c[0]*255.));
		}
	}
	rgb = QPixmap::fromImage(img);
	rgbView->setPixmap(rgb);
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
	conf.geodesic = graphsegGeodCheck->isEnabled();
	conf.multi_seed = false;
	int src = graphsegSourceBox->itemData(graphsegSourceBox->currentIndex())
								 .value<int>();
	if (src == 0) {
		bandView->startGraphseg(*image_illum, conf);
	} else if (src == 1) {
		bandView->startGraphseg(*gradient_illum, conf);
	} else {	// currently shown band, yes I know ITS FUCKING COMPLICATED
		multi_img_viewer *viewer; const multi_img *img;
		if (activeViewer == 0) {	viewer = viewIMG;	img = image_illum;    }
						  else {	viewer = viewGRAD;	img = gradient_illum; }
		int band = viewer->getViewport()->selection;
		multi_img i((*img)[band], img->minval, img->maxval);
		bandView->startGraphseg(i, conf);
	}
}

void ViewerWindow::setActive(bool gradient)
{
	activeViewer = (gradient ? 1 : 0);
}

void ViewerWindow::labelmask(bool negative)
{
	multi_img_viewer *viewer = (activeViewer == 0 ? viewIMG : viewGRAD);
	emit alterLabel(viewer->createMask(), negative);
	viewIMG->rebuild();
	viewGRAD->rebuild();
}

void ViewerWindow::newOverlay()
{
	multi_img_viewer *viewer = (activeViewer == 0 ? viewIMG : viewGRAD);
	emit drawOverlay(viewer->createMask());
}

void ViewerWindow::reshapeDock(bool floating)
{
	if (!floating)
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

void ViewerWindow::createMarkers()
{
	QVector<QColor> &col = bandView->markerColors;
	for (int i = 1; i < col.size(); ++i) // 0 is index for unlabeled
	{
		markerSelector->addItem(colorIcon(col[i]), "");
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
	multi_img::Illuminant il(temp);
	std::vector<multi_img::Value> cf;
	if (temp > 0) {
		il.calcWeight(image->meta[0].center,
					  image->meta[image->size()-1].center);
		cf = image->getIllumCoeff(il);
	} // when 0, empty vector is returned, which is just fine!
	illuminants[temp] = make_pair(il, cf);
}

const multi_img::Illuminant & ViewerWindow::getIlluminant(int temp)
{
	Illum_map::iterator i = illuminants.find(temp);
	if (i != illuminants.end())
		return i->second.first;

	buildIlluminant(temp);
	return illuminants[temp].first;
}

const std::vector<multi_img::Value> & ViewerWindow::getIlluminantC(int temp)
{
	Illum_map::iterator i = illuminants.find(temp);
	if (i != illuminants.end())
		return i->second.second;

	buildIlluminant(temp);
	return illuminants[temp].second;
}

void ViewerWindow::setI1(int index) {
	int i1 = i1Box->itemData(index).value<int>();
	viewIMG->setIlluminant(getIlluminantC(i1));
}
