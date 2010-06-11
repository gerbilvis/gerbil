#ifndef VIEWERWINDOW_H
#define VIEWERWINDOW_H

#include "ui_viewerwindow.h"
#include <multi_img.h>

#include <vector>
#include <QPixmap>
#include <QImage>
#include <QVector>
#include <QMenu>
#include <cv.h>

class ViewerWindow : public QMainWindow, private Ui::ViewerWindow {
    Q_OBJECT
public:
	ViewerWindow(const multi_img &image, const multi_img &gradient, QWidget *parent = 0);

	const QPixmap* getBand(int dim, bool gradient);

	static QIcon colorIcon(const QColor& color);

public slots:
	void reshapeDock(bool floating);
	void selectBand(int dim, bool gradient);
	void addToLabel()   { labelmask(false); }
	void remFromLabel() { labelmask(true); }
	void setActive(bool gradient);
	void newOverlay();
	void startGraphseg();

signals:
	void alterLabel(const multi_img::Mask &mask, bool negative);
	void drawOverlay(const multi_img::Mask &mask);

protected:
    void changeEvent(QEvent *e);

	// helper functions
	void createMarkers();
	void labelmask(bool negative);

	// bands from both image and gradient
	std::vector<QPixmap*> ibands, gbands;
	// pixel label holder
	cv::Mat_<uchar> labels;
	const multi_img &image, &gradient;
	int activeViewer; // 0: IMG, 1: GRAD

private:
	void initGraphsegUI();
};

#endif // VIEWERWINDOW_H
