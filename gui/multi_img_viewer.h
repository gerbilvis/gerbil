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
	void toggleLabeled(bool toggle);
	void toggleUnlabeled(bool toggle);
	void toggleLabels(bool toggle);
	void setActive(bool who);
	void setAlpha(int);
	void overlay(int x, int y);

protected:
    void changeEvent(QEvent *e);
	void createBins(int bins);

	const multi_img *image;
	bool ignoreLabels;

	std::vector<float> illuminant;

private:
	multi_img::Mask maskholder;
};

#endif // MULTI_IMG_VIEWER_H
