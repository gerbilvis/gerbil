#ifndef MULTI_IMG_VIEWER_H
#define MULTI_IMG_VIEWER_H

#include "ui_multi_img_viewer.h"
#include "viewport.h"
#include "multi_img.h"

#include <vector>
#include <QMenu>
#include <cv.h>

class multi_img_viewer : public QWidget, private Ui::multi_img_viewer {
    Q_OBJECT
public:
	multi_img_viewer(QWidget *parent = 0);

	const Viewport* getViewport() { return viewport; }
	const multi_img::Mask& getMask() { return maskholder; }

	/* translate image value to value in our coordinate system */
	inline multi_img::Value curpos(multi_img::Value val, int dim) {
		multi_img::Value curpos = (val - image->minval) / binsize;
		if (illuminant)
			curpos /= (*illuminant)[dim];
		return curpos;
	}

	const multi_img *image;
	multi_img::Mask labels;
	const QVector<QColor> *labelColors;

public slots:
	void rebuild(int bins = 0);
	void updateMask(int dim);
	void setImage(const multi_img *image, bool gradient = false);
	void setIlluminant(const std::vector<multi_img::Value> *, bool for_real);
	void toggleLabeled(bool toggle);
	void toggleUnlabeled(bool toggle);
	void toggleLabels(bool toggle);
	void toggleLimiters(bool toggle);
	void setActive(bool who);
	void setAlpha(int);
	void overlay(int x, int y);
	void showLimiterMenu();

signals:
	void newOverlay();

protected:
    void changeEvent(QEvent *e);
	void createBins();

	// helpers for createMask
	void fillMaskSingle(int dim, int sel);
	void fillMaskLimiters(const std::vector<std::pair<int, int> >& limits);
	void updateMaskLimiters(const std::vector<std::pair<int, int> >&, int dim);

	const std::vector<multi_img::Value> *illuminant;
	bool ignoreLabels;

private:
	void createLimiterMenu();

	multi_img::Mask maskholder;
	bool maskValid;
	// current number of bins shown
	int nbins;
	// respective data range of each bin
	multi_img::Value binsize;
	QMenu limiterMenu;
};

#endif // MULTI_IMG_VIEWER_H
