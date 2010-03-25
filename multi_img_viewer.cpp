#include "multi_img_viewer.h"
#include <cv.h>
#include <iostream>

using namespace std;

multi_img_viewer::multi_img_viewer(QWidget *parent)
	: QWidget(parent), image(NULL)
{
	setupUi(this);
}

void multi_img_viewer::setImage(const multi_img &img, bool gradient)
{
	if (!image) {
		connect(binSlider, SIGNAL(valueChanged(int)), this, SLOT(rebuild(int)));
		viewport->addSet(&unlabled);
	}
	image = &img;
	viewport->gradient = gradient;
	viewport->dimensionality = img.size();
	rebuild(binSlider->value());
}

void multi_img_viewer::rebuild(int bins)
{
	assert(image);
	binLabel->setText(QString("%1 bins").arg(bins));
	createBins(bins);
	viewport->nbins = bins;
	viewport->repaint();
}

void multi_img_viewer::createBins(int nbins)
{
	int dim = image->size();
	double minval = image->minval, maxval = image->maxval;
	double binsize = (maxval - minval)/(double)nbins;

	/* note that this is quite inefficient because of cache misses */
	cv::MatConstIterator_<double> it[dim];
	register int d;
	for (d = 0; d < dim; ++d)
		it[d] = (*image)[d].begin();

	unlabled.bins.clear();
	while (it[0] != (*image)[0].end()) {

		// create hash key and line array at once
		qlonglong hashkey = 0;
		qlonglong multiplier = 1;
		int lastpos = 0;
		QVector<QLineF> lines;
		for (d = 0; d < dim; ++d) {
			int curpos = floor((*it[d] - minval) / binsize);
			// minval/maxval are only theoretical bounds (TODO: check this in multi_img)
			curpos = max(curpos, 0); curpos = min(curpos, nbins-1);
			hashkey += multiplier * curpos;
			if (d > 0)
				lines.push_back(QLineF(d-1, lastpos, d, curpos));
			lastpos = curpos;
			multiplier *= nbins;
		}

		// put into our set
		if (!unlabled.bins.contains(hashkey)) {
			unlabled.bins.insert(hashkey, Bin(lines, 1.f));
		} else {
			unlabled.bins[hashkey].weight += 1.f;
		}

		// increment all iterators to next pixel
		for (d = 0; d < dim; ++d)
			++it[d];
	}

	unlabled.label = QColor(255, 255, 255);
	unlabled.totalweight = image->width * image->height;
}

void multi_img_viewer::changeEvent(QEvent *e)
{
    QWidget::changeEvent(e);
    switch (e->type()) {
    case QEvent::LanguageChange:
        retranslateUi(this);
        break;
    default:
        break;
    }
}
