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

	const QWidget* getViewport() { return viewport; }
	const cv::Mat_<uchar>& createMask();

	cv::Mat_<uchar> labels;
	const QVector<QColor> *labelcolors;

public slots:
	void rebuild(int bins = 0);
	void setImage(const multi_img &image, bool gradient = false);
	void toggleLabeled(bool toggle);
	void toggleUnlabeled(bool toggle);
	void toggleLabels(bool toggle);
	void setActive(bool who);

protected:
    void changeEvent(QEvent *e);
	void createBins(int bins);

	const multi_img *image;
	bool ignoreLabels;

private:
	cv::Mat_<uchar> maskholder;
};

#endif // MULTI_IMG_VIEWER_H
