#include "multi_img_viewer.h"
#include <cv.h>
#include <iostream>

using namespace std;

multi_img_viewer::multi_img_viewer(QWidget *parent)
	: QWidget(parent), image(NULL), labels(NULL)
{
	setupUi(this);
}

void multi_img_viewer::setImage(const multi_img &img, bool gradient)
{
	if (!image) {
		connect(binSlider, SIGNAL(valueChanged(int)), this, SLOT(rebuild(int)));
	}
	image = &img;
	viewport->gradient = gradient;
	viewport->dimensionality = img.size();
	rebuild(binSlider->value());
}

void multi_img_viewer::rebuild(int bins)
{
	assert(image);
	if (bins > 0) {
		binLabel->setText(QString("%1 bins").arg(bins));
		viewport->nbins = bins;
	}
	createBins(viewport->nbins);
	viewport->update();
}

void multi_img_viewer::createBins(int nbins)
{
	assert(labels && labelcolors);

	int dim = image->size();
	double minval = image->minval, maxval = image->maxval;
	double binsize = (maxval - minval)/(double)nbins;

	/* note that this is quite inefficient because of cache misses */
	cv::MatConstIterator_<double> it[dim];
	register int d;
	for (d = 0; d < dim; ++d)
		it[d] = (*image)[d].begin();

	vector<BinSet> &sets = viewport->sets;
	sets.clear();
	for (int i = 0; i < labelcolors->size(); ++i)
		sets.push_back(BinSet(labelcolors->at(i)));

	/* caution: dangerous assumption on the cv::Mat iterator order */
	for (int y = 0; y < labels->height(); ++y)
	  for (int x = 0; x < labels->width(); ++x) {
		// test the labeling
		int label = labels->pixelIndex(x, y);

		// create hash key and line array at once
		QByteArray hashkey;
		int lastpos = 0;
		QVector<QLineF> lines;
		for (d = 0; d < dim; ++d) {
			int curpos = floor((*it[d] - minval) / binsize);
			// minval/maxval are only theoretical bounds (TODO: check this in multi_img)
			curpos = max(curpos, 0); curpos = min(curpos, nbins-1);
			hashkey[d] = (unsigned char)curpos;
			if (d > 0)
				lines.push_back(QLineF(d-1, lastpos, d, curpos));
			lastpos = curpos;
		}

		// put into our set
		if (!sets[label].bins.contains(hashkey)) {
			sets[label].bins.insert(hashkey, Bin(lines, 1.f));
		} else {
			sets[label].bins[hashkey].weight += 1.f;
		}

		sets[label].totalweight++;

		// increment all iterators to next pixel
		for (d = 0; d < dim; ++d)
			++it[d];
	}
}

void multi_img_viewer::showLabeled(bool yes)
{
	viewport->showLabeled = yes;
	viewport->update();
}

void multi_img_viewer::showUnLabeled(bool yes)
{
	viewport->showUnlabeled = yes;
	viewport->update();
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
