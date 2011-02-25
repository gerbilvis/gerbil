#ifndef VIEWERWINDOW_H
#define VIEWERWINDOW_H

#include "ui_viewerwindow.h"
#include <multi_img.h>
#include <labeling.h>
#include <illuminant.h>
#include <graphseg.h>

#include <vector>
#include <QPixmap>
#include <QImage>
#include <QVector>
#include <QMenu>
#include <cv.h>

class ViewerWindow : public QMainWindow, private Ui::ViewerWindow {
    Q_OBJECT
public:
	ViewerWindow(multi_img *image, QWidget *parent = 0);

	const QPixmap* getBand(int dim, bool gradient);
	const inline Illuminant & getIlluminant(int temp);
	const inline std::vector<multi_img::Value> & getIlluminantC(int temp);

	static QIcon colorIcon(const QColor& color);

public slots:
	void reshapeDock(bool floating);
	void selectBand(int dim, bool gradient);
	void addToLabel()   { labelmask(false); }
	void remFromLabel() { labelmask(true); }
	void setActive(int id); // id 0: viewIMG, 1: viewGRAD
	void newOverlay();
	void startGraphseg();
	void applyIlluminant();
	void setI1(int index);
	void setI1Visible(bool);

	void normTargetChanged();
	void normModeSelected(int mode);
	void applyNormUserRange(bool update = true);
	void clampNormUserRange();

	void loadLabeling();
	void loadSeeds();
	void saveLabeling();
	// add new label (color)
	void createLabel();

	void ROITrigger();
	void ROIDecision(QAbstractButton *sender);
	void ROISelection(const QRect &roi);

signals:
	void alterLabel(const multi_img::Mask &mask, bool negative);
	void newLabelColors(const QVector<QColor> &colors, bool changed);
	void drawOverlay(const multi_img::Mask &mask);
	void seedingDone(bool yeah = false);

protected:
    void changeEvent(QEvent *e);

	/* helper functions */
	void applyROI();
	void labelmask(bool negative);
	// returns true if updates were triggered, false if not (trigger yourself!)
	bool setLabelColors(const std::vector<cv::Vec3b> &colors);
	void setLabels(const vole::Labeling &labeling);

	void runGraphseg(const multi_img& input, const vole::GraphSegConfig &config);

	// multispectral image and gradient
	multi_img *full_image, *image, *gradient;
	// current region of interest
	cv::Rect roi;
	// bands from both image and gradient
	std::vector<QPixmap*> ibands, gbands;
	// label colors
	QVector<QColor> labelColors;

	// rgb pixmap
	QPixmap full_rgb, rgb;
	multi_img_viewer *activeViewer; // 0: IMG, 1: GRAD

	enum normMode {
		NORM_OBSERVED = 0,
		NORM_THEORETICAL = 1,
		NORM_FIXED = 2
	};
	normMode normIMG, normGRAD;
	std::pair<multi_img::Value, multi_img::Value> normIMGRange, normGRADRange;

private:
	void initUI();
	void initGraphsegUI();
	void initIlluminantUI();
	void initNormalizationUI();
	void updateBand();
	void updateRGB(bool full);
	void buildIlluminant(int temp);

	// calculates norm range
	std::pair<multi_img::Value, multi_img::Value>
			getNormRange(normMode mode, int target,
						 std::pair<multi_img::Value, multi_img::Value> cur);

	// updates target's norm range member variable
	void setNormRange(int target);
	// updates target data range acc. to norm range member variable
	void updateImageRange(int target);

	// cache for illumination coefficients
	typedef std::map<int, std::pair<
			Illuminant, std::vector<multi_img::Value> > > Illum_map;
	Illum_map illuminants;
};

#endif // VIEWERWINDOW_H
