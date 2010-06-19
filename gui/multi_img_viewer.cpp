#include "multi_img_viewer.h"
#include <cv.h>
#include <iostream>

using namespace std;

multi_img_viewer::multi_img_viewer(QWidget *parent)
	: QWidget(parent), image(NULL), labelcolors(NULL),
	  ignoreLabels(false)
{
	setupUi(this);
}

void multi_img_viewer::setImage(const multi_img &img, bool gradient)
{
	if (!image) {
		connect(binSlider, SIGNAL(valueChanged(int)), this, SLOT(rebuild(int)));
	}
	image = &img;
	maskholder.create(image->height, image->width);
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
	assert(labelcolors && !labels.empty());

	// make sure the whole cache is built beforehand
	image->rebuildPixels();

	int dim = image->size();
	double minval = image->minval, maxval = image->maxval;
	double binsize = (maxval - minval)/(double)nbins;

	vector<BinSet> &sets = viewport->sets;
	sets.clear();
	for (int i = 0; i < labelcolors->size(); ++i)
		sets.push_back(BinSet(labelcolors->at(i)));

	for (int y = 0; y < labels.rows; ++y) {
		uchar *lr = labels[y];
		for (int x = 0; x < labels.cols; ++x) {
			// test the labeling
			int label = (ignoreLabels ? 0 : lr[x]);
			const multi_img::Pixel& pixel = (*image)(y, x);

			// create hash key and line array at once
			QByteArray hashkey;
			int lastpos = 0;
			QVector<QLineF> lines;
			for (int d = 0; d < dim; ++d) {
				int curpos = floor((pixel[d] - minval) / binsize);
				// multi_img::minval/maxval are only theoretical bounds
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
		}
	}
/* ** statistics **
	int datapoints = 0;
	for (unsigned int i = 0; i < sets.size(); ++i)
		datapoints += sets[i].bins.count();
	cerr << (viewport->gradient? "Gradient View" : "Intensity View") << " shows ";
	cerr << datapoints << " datapoints." << endl; */
}

/* create mask of single-band user selection */
const multi_img::Mask& multi_img_viewer::createMask()
{
	double minval = image->minval, maxval = image->maxval;
	double binsize = (maxval - minval)/(double)viewport->nbins;
	int d = viewport->selection;

	maskholder.setTo(0);
	multi_img::MaskIt itm = maskholder.begin();
	multi_img::BandConstIt itb = (*image)[d].begin();
	for (; itm != maskholder.end(); ++itb, ++itm) {
		int curpos = floor((*itb - minval) / binsize);
		// minval/maxval are only theoretical bounds
		curpos = max(curpos, 0); curpos = min(curpos, viewport->nbins-1);
		if (curpos == viewport->hover)
			*itm = 1;
	}
	return maskholder;
}

void multi_img_viewer::setActive(bool who)
{
	viewport->active = (who == viewport->gradient); // yes, indeed!
	viewport->update();
}

void multi_img_viewer::toggleLabeled(bool toggle)
{
	viewport->showLabeled = toggle;
	viewport->update();
}

void multi_img_viewer::toggleUnlabeled(bool toggle)
{
	viewport->showUnlabeled = toggle;
	viewport->update();
}

void multi_img_viewer::toggleLabels(bool toggle)
{
	ignoreLabels = toggle;
	viewport->ignoreLabels = toggle;
	rebuild();
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
