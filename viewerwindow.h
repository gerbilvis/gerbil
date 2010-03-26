#ifndef VIEWERWINDOW_H
#define VIEWERWINDOW_H

#include "ui_viewerwindow.h"
#include "multi_img.h"

#include <vector>
#include <QPixmap>

class ViewerWindow : public QMainWindow, private Ui::ViewerWindow {
    Q_OBJECT
public:
	ViewerWindow(const multi_img &image, const multi_img &gradient, QWidget *parent = 0);

	const QPixmap* getSlice(int dim);

public slots:
	void reshapeDock(bool floating);
	void selectSlice(int dim);

protected:
    void changeEvent(QEvent *e);

	std::vector<QPixmap*> slices;
	const multi_img &image;
};

#endif // VIEWERWINDOW_H
