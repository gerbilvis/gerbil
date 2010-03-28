#include "viewerwindow.h"

#include <QPainter>
#include <QIcon>
#include <iostream>

ViewerWindow::ViewerWindow(const multi_img &image, const multi_img &gradient, QWidget *parent)
	: QMainWindow(parent), image(image), gradient(gradient),
	  islices(image.size(), NULL), gslices(gradient.size(), NULL),
	  labels(image.width, image.height, QImage::Format_Indexed8)
{
	setupUi(this);
	sliceButton->hide();

	/* setup labeling stuff first */
	QVector<QColor> &labelcolors = sliceLabel->markerColors;
	labels.setNumColors(labelcolors.size());
	labels.fill(0);
	sliceLabel->labels = &labels;
	createMarkers();
	selectSlice(0, false);

	/* setup viewers, do setImage() last */
	viewIMG->labels = viewGRAD->labels = &labels;
	viewIMG->labelcolors = viewGRAD->labelcolors = &labelcolors;
	viewIMG->setImage(image);
	viewGRAD->setImage(gradient, true);

	/* signals & slots */
	connect(sliceDock, SIGNAL(visibilityChanged(bool)),
			sliceButton, SLOT(setHidden(bool)));
	connect(sliceButton, SIGNAL(clicked()),
			sliceDock, SLOT(show()));

	connect(sliceDock, SIGNAL(topLevelChanged(bool)),
			this, SLOT(reshapeDock(bool)));

	connect(viewIMG->getViewport(), SIGNAL(sliceSelected(int, bool)),
			this, SLOT(selectSlice(int, bool)));
	connect(viewGRAD->getViewport(), SIGNAL(sliceSelected(int, bool)),
			this, SLOT(selectSlice(int, bool)));

	connect(markerSelector, SIGNAL(currentIndexChanged(int)),
			sliceLabel, SLOT(changeLabel(int)));
	connect(clearButton, SIGNAL(clicked()),
			sliceLabel, SLOT(clearLabelPixels()));

	connect(applyButton, SIGNAL(clicked()),
			viewIMG, SLOT(rebuild()));
	connect(applyButton, SIGNAL(clicked()),
			viewGRAD, SLOT(rebuild()));
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

void ViewerWindow::reshapeDock(bool floating)
{
	if (!floating)
		return;

	const QPixmap *p = sliceLabel->pixmap();
	float src_aspect = p->width()/(float)p->height();
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

