#include "viewerwindow.h"

#include <QPainter>
#include <QIcon>
#include <iostream>

ViewerWindow::ViewerWindow(const multi_img &image, const multi_img &gradient, QWidget *parent)
	: QMainWindow(parent), image(image), gradient(gradient), activeViewer(0),
	  islices(image.size(), NULL), gslices(gradient.size(), NULL),
	  labels(image.height, image.width, (uchar)0)
{
	setupUi(this);
	sliceButton->hide();

	/* setup labeling stuff first */
	QVector<QColor> &labelcolors = sliceLabel->markerColors;
	sliceLabel->labels = labels;
	createMarkers();
	selectSlice(0, false);

	/* setup viewers, do setImage() last */
	viewIMG->labels = viewGRAD->labels = labels;
	viewIMG->labelcolors = viewGRAD->labelcolors = &labelcolors;
	viewIMG->setImage(image);
	viewGRAD->setImage(gradient, true);
	viewIMG->setActive(false);

	/* signals & slots */
	connect(sliceDock, SIGNAL(visibilityChanged(bool)),
			sliceButton, SLOT(setHidden(bool)));
	connect(sliceButton, SIGNAL(clicked()),
			sliceDock, SLOT(show()));

	connect(sliceDock, SIGNAL(topLevelChanged(bool)),
			this, SLOT(reshapeDock(bool)));

	connect(markerSelector, SIGNAL(currentIndexChanged(int)),
			sliceLabel, SLOT(changeLabel(int)));
	connect(clearButton, SIGNAL(clicked()),
			sliceLabel, SLOT(clearLabelPixels()));
	connect(ignoreButton, SIGNAL(toggled(bool)),
			markButton, SLOT(setDisabled(bool)));
	connect(ignoreButton, SIGNAL(toggled(bool)),
			nonmarkButton, SLOT(setDisabled(bool)));

	connect(addButton, SIGNAL(clicked()),
			this, SLOT(addToLabel()));
	connect(remButton, SIGNAL(clicked()),
			this, SLOT(remFromLabel()));
	connect(this, SIGNAL(alterLabel(const cv::Mat_<uchar>&,bool)),
			sliceLabel, SLOT(alterLabel(const cv::Mat_<uchar>&,bool)));
	connect(this, SIGNAL(drawOverlay(const cv::Mat_<uchar>&)),
			sliceLabel, SLOT(drawOverlay(const cv::Mat_<uchar>&)));

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

		connect(viewer[i]->getViewport(), SIGNAL(sliceSelected(int, bool)),
				this, SLOT(selectSlice(int, bool)));
		connect(viewer[i]->getViewport(), SIGNAL(activated(bool)),
				this, SLOT(setActive(bool)));
		connect(viewer[i]->getViewport(), SIGNAL(activated(bool)),
				viewer[(i ? 0 : 1)], SLOT(setActive(bool)));

		connect(viewer[i]->getViewport(), SIGNAL(newOverlay()),
				this, SLOT(newOverlay()));
	}
}

const QPixmap* ViewerWindow::getSlice(int dim, bool grad)
{
	// select variables according to which set is asked for
	std::vector<QPixmap*> &v = (grad ? gslices : islices);
	const multi_img &m = (grad ? gradient : image);

	if (!v[dim]) {
		// create here
		QImage img = m.export_qt(dim);
		v[dim] = new QPixmap(QPixmap::fromImage(img));
	}
	return v[dim];
}

void ViewerWindow::selectSlice(int dim, bool grad)
{
	sliceLabel->setPixmap(*getSlice(dim, grad));
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
	float dest_aspect = sliceLabel->width()/(float)sliceLabel->height();
	// we force the dock aspect ratio to fit slice image aspect ratio.
	// this is not 100% correct
	if (src_aspect > dest_aspect) {
		sliceDock->resize(sliceDock->width(), sliceDock->width()/src_aspect);
	} else
		sliceDock->resize(sliceDock->height()*src_aspect, sliceDock->height());
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
	QVector<QColor> &col = sliceLabel->markerColors;
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
