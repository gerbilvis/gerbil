/*
	Copyright(c) 2012 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "multi_img_viewer.h"
#include "viewerwindow.h"
#include <stopwatch.h>
#include <cv.h>
#include <iostream>
#include <QThread>

using namespace std;

multi_img_viewer::multi_img_viewer(QWidget *parent)
	: QWidget(parent), image(NULL), illuminant(NULL),
	  ignoreLabels(false), limiterMenu(this)
{
	setupUi(this);

	connect(binSlider, SIGNAL(valueChanged(int)),
			this, SLOT(rebuild(int)));
	connect(alphaSlider, SIGNAL(valueChanged(int)),
			this, SLOT(setAlpha(int)));
	connect(alphaSlider, SIGNAL(sliderPressed()),
			viewport, SLOT(startNoHQ()));
	connect(alphaSlider, SIGNAL(sliderReleased()),
			viewport, SLOT(endNoHQ()));
	connect(limiterButton, SIGNAL(toggled(bool)),
			this, SLOT(toggleLimiters(bool)));
	connect(limiterMenuButton, SIGNAL(clicked()),
			this, SLOT(showLimiterMenu()));

	connect(rgbButton, SIGNAL(toggled(bool)),
			viewport, SLOT(toggleRGB(bool)));

	connect(viewport, SIGNAL(newOverlay(int)),
			this, SLOT(updateMask(int)));

	setAlpha(alphaSlider->value());

	connect(topBar, SIGNAL(toggleFold()),
			this, SLOT(toggleFold()));

}

void multi_img_viewer::toggleFold()
{
	if (!payload->isHidden()) {
		emit folding();
		payload->setHidden(true);
		topBar->fold();
		setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
	} else {
		emit folding();
		payload->setShown(true);
		topBar->unfold();
		QSizePolicy pol(QSizePolicy::Preferred, QSizePolicy::Expanding);
		pol.setVerticalStretch(1);
		setSizePolicy(pol);
	}
}

void multi_img_viewer::setImage(const multi_img *img, representation type)
{
	if (type != IMG)
		rgbButton->setVisible(false);

	image = img;
	maskholder = multi_img::Mask(image->height, image->width, (uchar)0);

	QString title;
	if (type == IMG)
		title = QString("<b>Image Spectrum</b> [%1..%2]");
	if (type == GRAD)
		title = QString("<b>Spectral Gradient Spectrum</b> [%1..%2]");
	if (type == IMGPCA)
		title = QString("<b>Image PCA</b> [%1..%2]");
	if (type == GRADPCA)
		title = QString("<b>Spectral Gradient PCA</b> [%1..%2]");

	topBar->setTitle(title.arg(image->minval).arg(image->maxval));

	viewport->type = type;
	viewport->dimensionality = image->size();

	/* intialize meta data */
	viewport->labels.resize(image->size());
	for (unsigned int i = 0; i < image->size(); ++i) {
		if (!image->meta[i].empty)
			viewport->labels[i].setNum(image->meta[i].center);
	}

	rebuild(binSlider->value());
}

void multi_img_viewer::setIlluminant(
		const std::vector<multi_img::Value> *coeffs, bool for_real)
{
	if (viewport->type != IMG)
		return;

	if (for_real) {
		// only set it to true, never to false again
		viewport->illuminant_correction = true;

		illuminant = coeffs;
		rebuild();
	} else {
		viewport->illuminant = coeffs;
		viewport->update();
	}
}

void multi_img_viewer::rebuild(int bins)
{
	if (!image)
		return;

	if (bins > 0 || bins == -1) { // number of bins or binning changed
		if (bins > 0) {
			nbins = bins;
			binLabel->setText(QString("%1 bins").arg(bins));
		}
		binsize = (image->maxval - image->minval)/(multi_img::Value)(nbins-1);
		viewport->reset(nbins, binsize, image->minval);
	}
	createBins();
	viewport->prepareLines();
	viewport->update();
}

void multi_img_viewer::createBins()
{
	assert(!labelColors.empty() && !labels.empty());

	// vole::Stopwatch watch;

	// make sure the whole cache is built beforehand
	image->rebuildPixels();

	int dim = image->size();

	vector<BinSet> &sets = viewport->sets;
	vector<std::pair<int, QByteArray> > &shuffleIdx = viewport->shuffleIdx;
	sets.clear(); shuffleIdx.clear();
	for (int i = 0; i < labelColors.size(); ++i)
		sets.push_back(BinSet(labelColors[i], dim));

	for (int y = 0; y < labels.rows; ++y) {
		short *lr = labels[y];
		for (int x = 0; x < labels.cols; ++x) {
			// test the labeling
			int label = (ignoreLabels ? 0 : lr[x]);
			const multi_img::Pixel& pixel = (*image)(y, x);
			BinSet &s = sets[label];

			// create hash key and line array at once
			QByteArray hashkey;
			for (int d = 0; d < dim; ++d) {
				int pos = floor(curpos(pixel[d], d));
				/* multi_img::minval/maxval are only theoretical bounds,
				   so they could be violated */
				pos = max(pos, 0); pos = min(pos, nbins-1);
				hashkey[d] = (unsigned char)pos;

				/* cache observed range; can be used for limiter init later */
				std::pair<int, int> &range = s.boundary[d];
				range.first = std::min<int>(range.first, pos);
				range.second = std::max<int>(range.second, pos);
			}

			// put into our set
			if (!s.bins.contains(hashkey)) {
				s.bins.insert(hashkey, Bin(pixel));
				// add to initial shuffle index
				shuffleIdx.push_back(make_pair(label, hashkey));
			} else {
				s.bins[hashkey].add(pixel);
			}

			sets[label].totalweight++;
		}
	}
	//watch.print_reset(" - bin creation");

	/* normalize means & calculate color */
	for (unsigned int i = 0; i < sets.size(); ++i) {
		BinSet &s = sets[i];
		QHash<QByteArray, Bin>::iterator it;
		for (it = s.bins.begin(); it != s.bins.end(); ++it) {
			Bin &b = it.value();
			for (int d = 0; d < dim; ++d)
				b.means[d] /= b.weight;
			cv::Vec3f color = image->bgr(b.means);
			b.rgb = QColor(color[2]*255, color[1]*255, color[0]*255);
		}
	}
	//watch.print_reset(" - bin normalization + RGB");

	/* shuffle indices with fisher-yates */
	std::random_shuffle(shuffleIdx.begin(), shuffleIdx.end());
	//watch.print(" - index shuffle");

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
		int pos = floor(curpos(*itb, dim));
		if (pos == sel)
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
			for (unsigned int d = 0; d < image->size(); ++d) {
				int pos = floor(curpos(p[d], d));
				if (pos < l[d].first || pos > l[d].second) {
					row[x] = 0;
					break;
				}
			}
		}
	}
}

void multi_img_viewer::updateMaskLimiters(
		const std::vector<std::pair<int, int> >& l, int dim)
{
	for (int y = 0; y < image->height; ++y) {
		uchar *mrow = maskholder[y];
		const multi_img::Value *brow = (*image)[dim][y];
		for (int x = 0; x < image->width; ++x) {
			int pos = floor(curpos(brow[x], dim));
			if (pos < l[dim].first || pos > l[dim].second) {
				mrow[x] = 0;
			} else if (mrow[x] == 0) { // we need to do exhaustive test
				mrow[x] = 1;
				const multi_img::Pixel& p = (*image)(y, x);
				for (unsigned int d = 0; d < image->size(); ++d) {
					int pos = floor(curpos(p[d], d));
					if (pos < l[d].first || pos > l[d].second) {
						mrow[x] = 0;
						break;
					}
				}
			}
		}
	}
}

void multi_img_viewer::updateMask(int dim)
{
	if (viewport->limiterMode) {		
		if (maskValid && dim > -1)
			updateMaskLimiters(viewport->limiters, dim);
		else
			fillMaskLimiters(viewport->limiters);
		maskValid = true;
	} else {
		fillMaskSingle(viewport->selection, viewport->hover);
	}
}

void multi_img_viewer::overlay(int x, int y)
{
	const multi_img::Pixel &pixel = (*image)(y, x);
	QPolygonF &points = viewport->overlayPoints;
	points.resize(image->size());

	for (unsigned int d = 0; d < image->size(); ++d)
		points[d] = QPointF(d, curpos(pixel[d], d));

	viewport->overlayMode = true;
	viewport->repaint();
	viewport->overlayMode = false;
}

void multi_img_viewer::setAlpha(int alpha)
{
	viewport->useralpha = (float)alpha/100.f;
	alphaLabel->setText(QString::fromUtf8("α: %1").arg(viewport->useralpha, 0, 'f', 2));
	viewport->update();
}

void multi_img_viewer::createLimiterMenu()
{
	limiterMenu.clear();
	QAction *tmp;
	tmp = limiterMenu.addAction("No limits");
	tmp->setData(0);
	tmp = limiterMenu.addAction("Limit from current highlight");
	tmp->setData(-1);
	limiterMenu.addSeparator();
	for (int i = 1; i < labelColors.size(); ++i) {
		tmp = limiterMenu.addAction(ViewerWindow::colorIcon(labelColors[i]),
													  "Limit by label");
		tmp->setData(i);
	}
}

void multi_img_viewer::showLimiterMenu()
{
	QAction *a = limiterMenu.exec(limiterMenuButton->mapToGlobal(QPoint(0, 0)));
	if (!a)
		return;

	int choice = a->data().toInt(); assert(choice < labelColors.size());
	viewport->setLimiters(choice);
	if (!limiterButton->isChecked()) {
		limiterButton->toggle();	// change button state AND toggleLimiters()
	} else {
		toggleLimiters(true);
	}
}

void multi_img_viewer::updateLabelColors(const QVector<QColor> &colors, bool changed)
{
	labelColors = colors;
	createLimiterMenu();
	if (changed)
		rebuild();
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
	viewport->activate();
	updateMask(-1);
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
