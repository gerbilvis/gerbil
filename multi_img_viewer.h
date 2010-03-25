#ifndef MULTI_IMG_VIEWER_H
#define MULTI_IMG_VIEWER_H

#include "ui_multi_img_viewer.h"
#include "viewport.h"
#include "multi_img.h"

#include <vector>

class multi_img_viewer : public QMainWindow, private Ui::multi_img_viewer {
    Q_OBJECT
public:
	multi_img_viewer(const multi_img& img, QWidget *parent = 0);

	void createBins(int bins);

public slots:
	void rebuild(int bins);

protected:
    void changeEvent(QEvent *e);

	const multi_img& image;
	BinSet unlabled;
};

#endif // MULTI_IMG_VIEWER_H
