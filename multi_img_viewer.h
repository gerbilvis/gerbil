#ifndef MULTI_IMG_VIEWER_H
#define MULTI_IMG_VIEWER_H

#include "ui_multi_img_viewer.h"
#include "viewport.h"
#include "multi_img.h"

#include <vector>

class multi_img_viewer : public QWidget, private Ui::multi_img_viewer {
    Q_OBJECT
public:
	multi_img_viewer(QWidget *parent = 0);

public slots:
	void rebuild(int bins);
	void setImage(const multi_img &image, bool gradient = false);

protected:
    void changeEvent(QEvent *e);

	void createBins(int bins);
	const multi_img *image;
	BinSet unlabled;
};

#endif // MULTI_IMG_VIEWER_H
