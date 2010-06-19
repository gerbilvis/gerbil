#include "viewerwindow.h"

#include <graphseg_config.h>

#include <QPainter>
#include <QIcon>
#include <iostream>

ViewerWindow::ViewerWindow(const multi_img &image, const multi_img &gradient, QWidget *parent)
	: QMainWindow(parent), image(image), gradient(gradient), activeViewer(0),
	  ibands(image.size(), NULL), gbands(gradient.size(), NULL),
	  labels(image.height, image.width, (uchar)0)
{
	setupUi(this);
	bandButton->hide();

	/* setup labeling stuff first */
	QVector<QColor> &labelcolors = bandLabel->markerColors;
	bandLabel->labels = labels;
	createMarkers();
	selectBand(0, false);
	graphsegWidget->hide();

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
			bandLabel, SLOT(changeLabel(int)));
	connect(clearButton, SIGNAL(clicked()),
			bandLabel, SLOT(clearLabelPixels()));

	initGraphsegUI();

	connect(ignoreButton, SIGNAL(toggled(bool)),
			markButton, SLOT(setDisabled(bool)));
	connect(ignoreButton, SIGNAL(toggled(bool)),
			nonmarkButton, SLOT(setDisabled(bool)));

	connect(addButton, SIGNAL(clicked()),
			this, SLOT(addToLabel()));
	connect(remButton, SIGNAL(clicked()),
			this, SLOT(remFromLabel()));
	connect(this, SIGNAL(alterLabel(const multi_img::Mask&,bool)),
			bandLabel, SLOT(alterLabel(const multi_img::Mask&,bool)));
	connect(this, SIGNAL(drawOverlay(const multi_img::Mask&)),
			bandLabel, SLOT(drawOverlay(const multi_img::Mask&)));

	multi_img_viewer *viewer[2] = {viewIMG, viewGRAD };
	for (int i = 0; i < 2; ++i)
	{
		connect(applyButton, SIGNAL(clicked()),
				viewer[i], SLOT(rebuild()));
		connect(markButton, SIGNAL(toggled(bool)),
				viewer[i], SLOT(toggleLabeled(bool)));
		connect(nonmarkButton, SIGNAL(toggled(bool)),
				viewer[i], SLOT(toggleUnlabeled(bool)));
		connect(ignoreButton, SIGNAL(toggled(bool)),
				viewer[i], SLOT(toggleLabels(bool)));

		connect(viewer[i]->getViewport(), SIGNAL(bandSelected(int, bool)),
				this, SLOT(selectBand(int, bool)));
		connect(viewer[i]->getViewport(), SIGNAL(activated(bool)),
				this, SLOT(setActive(bool)));
		connect(viewer[i]->getViewport(), SIGNAL(activated(bool)),
				viewer[(i ? 0 : 1)], SLOT(setActive(bool)));

		connect(viewer[i]->getViewport(), SIGNAL(newOverlay()),
				this, SLOT(newOverlay()));
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
			bandLabel, SLOT(toggleSeedMode(bool)));
	connect(graphsegGoButton, SIGNAL(clicked()),
			this, SLOT(startGraphseg()));
	connect(bandLabel, SIGNAL(seedingDone(bool)),
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
		bandLabel->startGraphseg(image, conf);
	} else if (src == 1) {
		bandLabel->startGraphseg(gradient, conf);
	} else {	// currently shown band, yes I know ITS FUCKING COMPLICATED
		multi_img_viewer *viewer; const multi_img *img;
		if (activeViewer == 0) {	viewer = viewIMG;	img = &image;    }
						  else {	viewer = viewGRAD;	img = &gradient; }
		int band = viewer->getViewport()->selection;
		multi_img i((*img)[band], img->minval, img->maxval);
		bandLabel->startGraphseg(i, conf);
	}
}

const QPixmap* ViewerWindow::getBand(int dim, bool grad)
{
	// select variables according to which set is asked for
	std::vector<QPixmap*> &v = (grad ? gbands : ibands);

	if (!v[dim]) {
		const multi_img &m = (grad ? gradient : image);
		// create here
		QImage img = m.export_qt(dim);
		v[dim] = new QPixmap(QPixmap::fromImage(img));
	}
	return v[dim];
}

void ViewerWindow::selectBand(int dim, bool grad)
{
	bandLabel->setPixmap(*getBand(dim, grad));
	const multi_img &m = (grad ? gradient : image);
	std::string banddesc = m.meta[dim].str();
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

	float src_aspect = image.width/(float)image.height;
	float dest_aspect = bandLabel->width()/(float)bandLabel->height();
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
	QVector<QColor> &col = bandLabel->markerColors;
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
