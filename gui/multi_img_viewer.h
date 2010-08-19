#ifndef MULTI_IMG_VIEWER_H
#define MULTI_IMG_VIEWER_H

#include "ui_multi_img_viewer.h"
#include "viewport.h"
#include "multi_img.h"

#include <vector>
#include <cv.h>

class multi_img_viewer : public QWidget, private Ui::multi_img_viewer {
    Q_OBJECT
public:
	multi_img_viewer(QWidget *parent = 0);

	const Viewport* getViewport() { return viewport; }
	const multi_img::Mask& createMask();

	multi_img::Mask labels;
	const QVector<QColor> *labelcolors;

public slots:
	void rebuild(int bins = 0);
	void setImage(const multi_img *image, bool gradient = false);
	void setIlluminant(const std::vector<multi_img::Value> *);
	void toggleLabeled(bool toggle);
	void toggleUnlabeled(bool toggle);
	void toggleLabels(bool toggle);
	void toggleLimiters(bool toggle);
	void setActive(bool who);
	void setAlpha(int);
	void overlay(int x, int y);

signals:
	void newOverlay();

protected:
    void changeEvent(QEvent *e);
	void createBins();

	// helpers for createMask
	void fillMaskSingle(int dim, int sel);
	void fillMaskLimiters(const std::vector<std::pair<int, int> >& limits);

	const multi_img *image;
	bool ignoreLabels;

	const std::vector<multi_img::Value> *illuminant;

private:
	multi_img::Mask maskholder;
	// current number of bins shown
	int nbins;
	// respective data range of each bin
	multi_img::Value binsize;
};

#endif // MULTI_IMG_VIEWER_H
