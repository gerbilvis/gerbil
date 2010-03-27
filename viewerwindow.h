#ifndef VIEWERWINDOW_H
#define VIEWERWINDOW_H

#include "ui_viewerwindow.h"
#include "multi_img.h"

#include <vector>
#include <QPixmap>
#include <QImage>
#include <QVector>

class ViewerWindow : public QMainWindow, private Ui::ViewerWindow {
    Q_OBJECT
public:
	ViewerWindow(const multi_img &image, const multi_img &gradient, QWidget *parent = 0);

	const QPixmap* getSlice(int dim, bool gradient);

	static QIcon colorIcon(const QColor& color);

public slots:
	void reshapeDock(bool floating);
	void selectSlice(int dim, bool gradient);

protected:
    void changeEvent(QEvent *e);

	// helper functions
	void createMarkers();

	// slices from both image and gradient
	std::vector<QPixmap*> islices, gslices;
	// pixel label holder
	QImage labels;
	const multi_img &image, &gradient;
};

#endif // VIEWERWINDOW_H
