#include "viewerwindow.h"

#include <iostream>

ViewerWindow::ViewerWindow(const multi_img &image, const multi_img &gradient, QWidget *parent)
	: QMainWindow(parent), slices(image.size(), NULL), image(image)
{
	setupUi(this);
	viewIMG->setImage(image);
	viewGRAD->setImage(gradient, true);

	sliceButton->hide();
	connect(sliceDock, SIGNAL(visibilityChanged(bool)),
			sliceButton, SLOT(setHidden(bool)));
	connect(sliceButton, SIGNAL(clicked()),
			sliceDock, SLOT(show()));

	connect(sliceDock, SIGNAL(topLevelChanged(bool)),
			this, SLOT(reshapeDock(bool)));

	connect(viewIMG->getViewport(), SIGNAL(sliceSelected(int)),
			this, SLOT(selectSlice(int)));

	selectSlice(0);
}

const QPixmap* ViewerWindow::getSlice(int dim)
{
	if (!slices[dim]) {
		// create here
		QImage img = image.export_qt(dim);
		slices[dim] = new QPixmap(QPixmap::fromImage(img));
	}
	return slices[dim];
}

void ViewerWindow::selectSlice(int dim)
{
	sliceLabel->setPixmap(*getSlice(dim));
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
