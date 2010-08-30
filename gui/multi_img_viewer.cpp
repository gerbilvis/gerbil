#include "multi_img_viewer.h"
#include "viewerwindow.h"
#include <stopwatch.h>
#include <cv.h>
#include <iostream>

using namespace std;

multi_img_viewer::multi_img_viewer(QWidget *parent)
	: QWidget(parent), labelcolors(NULL), image(NULL),
	  ignoreLabels(false), illuminant(NULL), limiterMenu(this)
{
	setupUi(this);

	connect(binSlider, SIGNAL(valueChanged(int)),
			this, SLOT(rebuild(int)));
	connect(alphaSlider, SIGNAL(valueChanged(int)),
			this, SLOT(setAlpha(int)));
	connect(limiterButton, SIGNAL(toggled(bool)),
			this, SLOT(toggleLimiters(bool)));
	connect(limiterMenuButton, SIGNAL(clicked()),
			this, SLOT(showLimiterMenu()));

	setAlpha(70);
}

void multi_img_viewer::setImage(const multi_img *img, bool gradient)
{
	image = img;
	if (maskholder.rows != image->height || maskholder.cols != image->width)
		maskholder.create(image->height, image->width);

	QString title("<b>%1 Spectrum</b> [%2..%3]");
	titleLabel->setText(title.arg(gradient ? "Spectral Gradient" : "Image")
						.arg(image->minval).arg(image->maxval));

	viewport->gradient = gradient;
	viewport->dimensionality = image->size();

	/* intialize meta data */
	viewport->labels.resize(image->size());
	for (unsigned int i = 0; i < image->size(); ++i) {
		if (!image->meta[i].empty)
			viewport->labels[i].setNum(image->meta[i].center);
	}

	rebuild(binSlider->value());
}

void multi_img_viewer::setIlluminant(const std::vector<multi_img::Value> *coeffs)
{
	if (illuminant == coeffs)
		return;
	illuminant = coeffs;
	viewport->illuminant = illuminant;
	rebuild();
}

void multi_img_viewer::rebuild(int bins)
{
	if (!image)
		return;

	if (bins > 0) { // number of bins changed
		nbins = bins;
		binsize = (image->maxval - image->minval)/(multi_img::Value)(nbins-1);
		binLabel->setText(QString("%1 bins").arg(bins));
		viewport->nbins = bins;
		viewport->binsize = binsize;
		viewport->minval = image->minval;
		// reset hover value that would become inappropr.
		viewport->hover = -1;
		// reset limiters to most-lazy values
		setLimiters(0);
	}
	createBins();
	viewport->updateModelview();
	viewport->update();
}

void multi_img_viewer::createBins()
{
	assert(labelcolors && !labels.empty());

	//vole::Stopwatch s("Bin creation");

	// make sure the whole cache is built beforehand
	image->rebuildPixels();

	int dim = image->size();

	vector<BinSet> &sets = viewport->sets;
	sets.clear();
	for (int i = 0; i < labelcolors->size(); ++i)
		sets.push_back(BinSet(labelcolors->at(i), dim));

	for (int y = 0; y < labels.rows; ++y) {
		uchar *lr = labels[y];
		for (int x = 0; x < labels.cols; ++x) {
			// test the labeling
			int label = (ignoreLabels ? 0 : lr[x]);
			const multi_img::Pixel& pixel = (*image)(y, x);

			// create hash key and line array at once
			QByteArray hashkey;
			for (int d = 0; d < dim; ++d) {
				int curpos = floor((pixel[d] - image->minval) / binsize);
				/* multi_img::minval/maxval are only theoretical bounds,
				   so they could be violated */
				curpos = max(curpos, 0); curpos = min(curpos, nbins-1);
				hashkey[d] = (unsigned char)curpos;

				/* cache observed range; can be used for limiter init later */
				std::pair<int, int> &range = sets[label].boundary[d];
				range.first = std::min(range.first, curpos);
				range.second = std::max(range.second, curpos);
			}

			// put into our set
			if (!sets[label].bins.contains(hashkey)) {
				sets[label].bins.insert(hashkey, Bin(pixel));
			} else {
				sets[label].bins[hashkey].add(pixel);
			}

			sets[label].totalweight++;
		}
	}

	viewport->prepareLines();
/* ** statistics **
	int datapoints = 0;
	for (unsigned int i = 0; i < sets.size(); ++i)
		datapoints += sets[i].bins.count();
	cerr << (viewport->gradient? "Gradient View" : "Intensity View") << " shows ";
	cerr << datapoints << " datapoints." << endl;
*/
}

/* create mask of single-band user selection */
void multi_img_viewer::fillMaskSingle(int dim, int sel)
{
	maskholder.setTo(0);

	multi_img::MaskIt itm = maskholder.begin();
	multi_img::BandConstIt itb = (*image)[dim].begin();
	for (; itm != maskholder.end(); ++itb, ++itm) {
		int curpos = floor((*itb - image->minval) / binsize);
		if (curpos == sel)
			*itm = 1;
	}
}

void multi_img_viewer::fillMaskLimiters(const std::vector<std::pair<int, int> >& l)
{
	maskholder.setTo(1);

	for (int y = 0; y < image->height; ++y) {
		uchar *row = maskholder[y];
		for (int x = 0; x < image->width; ++x) {
			const multi_img::Pixel& p = (*image)(y, x);
			for (unsigned int i = 0; i < image->size(); ++i) {
				int curpos = floor((p[i] - image->minval) / binsize);
				if (curpos < l[i].first || curpos > l[i].second) {
					row[x] = 0;
					break;
				}
			}
		}
	}
}

const multi_img::Mask& multi_img_viewer::createMask()
{
	//vole::Stopwatch s("Mask creation");
	if (viewport->limiterMode)
		fillMaskLimiters(viewport->limiters);
	else
		fillMaskSingle(viewport->selection, viewport->hover);
	return maskholder;
}

void multi_img_viewer::overlay(int x, int y)
{
	const multi_img::Pixel &pixel = (*image)(y, x);
	QVector<QLineF> &lines = viewport->overlayLines;
	lines.clear();

	qreal lastpos = 0.;
	for (unsigned int d = 0; d < image->size(); ++d) {
		qreal curpos = (pixel[d] - image->minval) / binsize;
		if (illuminant)
			curpos *= (*illuminant)[d];
		if (d > 0)
			lines.push_back(QLineF(d-1, lastpos, d, curpos));
		lastpos = curpos;
	}

	viewport->overlayMode = true;
	viewport->repaint();
	viewport->overlayMode = false;
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

void multi_img_viewer::toggleLimiters(bool toggle)
{
	viewport->limiterMode = toggle;
	viewport->repaint();
	emit newOverlay();
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

void multi_img_viewer::setAlpha(int alpha)
{
	viewport->useralpha = (float)alpha/100.f;
	alphaLabel->setText(QString::fromUtf8("α: %1").arg(viewport->useralpha, 0, 'f', 2));
	viewport->update();
}

void multi_img_viewer::createLimiterMenu()
{
	QAction *tmp;
	tmp = limiterMenu.addAction("No limits");
	tmp->setData(0);
	tmp = limiterMenu.addAction("Limit from current highlight");
	tmp->setData(-1);
	limiterMenu.addSeparator();
	for (int i = 1; i < labelcolors->size(); ++i) {
		tmp = limiterMenu.addAction(ViewerWindow::colorIcon((*labelcolors)[i]),
													  "Limit by label");
		tmp->setData(i);
	}
}

void multi_img_viewer::showLimiterMenu()
{
	if (limiterMenu.isEmpty())
		createLimiterMenu();
	QAction *a = limiterMenu.exec(limiterMenuButton->mapToGlobal(QPoint(0, 0)));
	if (!a)
		return;

	int choice = a->data().toInt(); assert(choice < labelcolors->size());
	setLimiters(choice);
	if (!limiterButton->isChecked())
		limiterButton->toggle();	// change button state, toggleLimiters()
	else
		toggleLimiters(true);
}

void multi_img_viewer::setLimiters(int label)
{
	if (label < 1) {	// not label
		viewport->limiters.assign(image->size(), make_pair(0, nbins-1));
		if (label == -1) {	// use hover data
			int b = viewport->selection;
			int h = viewport->hover;
			viewport->limiters[b] = std::make_pair(h, h);
		}
	} else {
		if (viewport->sets[label].totalweight > 0.f) { // label holds data
			// use range from this label
			const std::vector<std::pair<int, int> > &b = viewport->sets[label].boundary;
			viewport->limiters.assign(b.begin(), b.end());
		} else
			setLimiters(0);
	}
}
