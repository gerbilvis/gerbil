#ifndef VIEWERWINDOW_H
#define VIEWERWINDOW_H

#include "ui_viewerwindow.h"
#include "multi_img.h"

#include <vector>
#include <QPixmap>
#include <QImage>
#include <QVector>
#include <cv.h>

class ViewerWindow : public QMainWindow, private Ui::ViewerWindow {
    Q_OBJECT
public:
	ViewerWindow(const multi_img &image, const multi_img &gradient, QWidget *parent = 0);

	const QPixmap* getSlice(int dim, bool gradient);

	static QIcon colorIcon(const QColor& color);

public slots:
	void reshapeDock(bool floating);
	void selectSlice(int dim, bool gradient);
	void addToLabel()   { labelmask(false); }
	void remFromLabel() { labelmask(true); }
	void setActive(bool gradient);

signals:
	void alterLabel(const cv::Mat_<uchar> &mask, bool negative);

protected:
    void changeEvent(QEvent *e);

	// helper functions
	void createMarkers();
	void labelmask(bool negative);

	// slices from both image and gradient
	std::vector<QPixmap*> islices, gslices;
	// pixel label holder
	cv::Mat_<uchar> labels;
	const multi_img &image, &gradient;
	int activeViewer; // 0: IMG, 1: GRAD
};

#endif // VIEWERWINDOW_H
