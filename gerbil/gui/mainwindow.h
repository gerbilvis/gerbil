/*
	Copyright(c) 2012 Johannes Jordan <johannes.jordan@cs.fau.de>.
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "ui_mainwindow.h"
#include <shared_data.h>
#include <background_task.h>
#include <background_task_queue.h>
#include "multi_img_viewer.h"  //FIXME REMOVE
#include <multi_img.h>
#include <multi_img_tasks.h>
#include <labeling.h>
#include <illuminant.h>
#include <progress_observer.h>
#include "commandrunner.h"
#ifdef WITH_SEG_MEANSHIFT
#include <meanshift_shell.h>
#endif
#ifdef WITH_SEG_MEDIANSHIFT
#include <medianshift_shell.h>
#endif
#ifdef WITH_SEG_PROBSHIFT
#include <probshift_shell.h>
#endif
#include <graphseg.h>

#include <vector>
#include <QPixmap>
#include <QImage>
#include <QVector>
#include <QMenu>
#include <opencv2/core/core.hpp>

class MainWindow : public QMainWindow, private Ui::MainWindow {
    Q_OBJECT
public:
	MainWindow(BackgroundTaskQueue &queue, multi_img_base *image,
				 QString labelfile = QString(), bool limitedMode = false,
				 QWidget *parent = 0);

	// -> viewerContainer
//	const QPixmap* getBand(representation type, int dim);
	const inline Illuminant & getIlluminant(int temp);
	const inline std::vector<multi_img::Value> & getIlluminantC(int temp);

	static QIcon colorIcon(const QColor& color);
	BackgroundTaskQueue &queue;

	/* BEGIN methods for viewercontainer integration */
	bool haveImagePCA();
	bool haveGradientPCA();
	/* END methods for viewercontainer integration */

public slots:
	void setGUIEnabled(bool enable, TaskType tt = TT_NONE);
//	void imgCalculationComplete(bool success);
//	void gradCalculationComplete(bool success);
//	void imgPcaCalculationComplete(bool success);
//	void gradPcaCalculationComplete(bool success);
//	void disconnectViewer(int viewer);
	void toggleViewer(bool enable, representation viewer);
//	void finishViewerRefresh(int viewer);
	void finishROIChange(bool success);
	void finishNormRangeImgChange(bool success);
	void finishNormRangeGradChange(bool success);
	void finishGraphSeg(bool success);
	void finishTask(bool success);

	void reshapeDock(bool floating);
	void selectBand(representation type, int dim);
	void addToLabel()   { labelmask(false); }
	void remFromLabel() { labelmask(true); }
	void setActive(int id); // id mapping see initUI()
	void newOverlay();

	void startGraphseg();

	void startUnsupervisedSeg(bool findKL = false);
	void startFindKL();
	void segmentationFinished();
	void segmentationApply(std::map<std::string, boost::any>);

	void applyIlluminant();
	void setI1(int index);
	void setI1Visible(bool);
	void bandsSliderMoved(int b);
	void toggleLabels(bool toggle);

	void usMethodChanged(int idx);
	void usInitMethodChanged(int idx);
	void usBandwidthMethodChanged(const QString &current);
	void unsupervisedSegCancelled();

	void normTargetChanged(bool usecurrent = false);
	void normModeSelected(int mode, bool targetchange = false, bool usecurrent = false);
	void normModeFixed();
	void applyNormUserRange();
	void clampNormUserRange();

	void loadLabeling(QString filename = "");
	void loadSeeds();
	void saveLabeling();
	// add new label (color)
	void createLabel();

	void ROIDecision(QAbstractButton *sender);
	void ROISelection(const QRect &roi);

	void openContextMenu();

	void screenshot();

	void updateRGB(bool success);
//	void refreshLabelsInViewers();
public slots: // introduced by ViewerContainer
	// FIXME was void updateBand() -> still breaks compile -> all updateBand() calls
	// need to move from MainWindow to ViewerContainer
	void updateBand(representation repr, int selection);
	void imageResetNeeded(representation repr);
signals:
	void clearLabel();
	void alterLabel(const multi_img::Mask &mask, bool negative);
	void newLabelColors(const QVector<QColor> &colors, bool changed);
	void drawOverlay(const multi_img::Mask &mask);
	void seedingDone(bool yeah = false);
signals:
	/* BEGIN signals for viewercontainer integration */

	// TODO this needs to be emitted in all the right places.
	void roiChanged(cv::Rect roi);
	/* END signals for viewercontainer integration */

protected:
	void changeEvent(QEvent *e);

	/* helper functions */
	void applyROI(bool reuse);
	void labelmask(bool negative);
	// returns true if updates were triggered, false if not (trigger yourself!)
	bool setLabelColors(const std::vector<cv::Vec3b> &colors);
	void setLabels(const vole::Labeling &labeling);

	void runGraphseg(SharedMultiImgPtr input, const vole::GraphSegConfig &config);

	// FIXME SharedMultiImgPtr s are to be moved into model classes

	// multispectral image and gradient
    // FIXME rename
    SharedMultiImgPtr image_lim;

    SharedMultiImgPtr image, gradient, imagepca, gradientpca;

	// current region of interest
	cv::Rect roi;
	// bands from all representations (image, gradient, PCA..)
	std::vector<std::vector<QPixmap*> > bands;
	// label colors
	QVector<QColor> labelColors;
	// full image labels and roi scoped labels
	cv::Mat1s full_labels, labels;

	// rgb pixmap
	QPixmap full_rgb, rgb;
	qimage_ptr full_rgb_temp; // QPixmap cannot be directly shared between threads

	// viewers
	//std::vector<multi_img_viewer*> viewers;
	//multi_img_viewer *activeViewer;

	MultiImg::NormMode normIMG, normGRAD;
	data_range_ptr normIMGRange, normGRADRange;

protected slots:
	void labelflush();

private:
	void initUI();
	void initGraphsegUI();
	void initIlluminantUI();
#ifdef WITH_SEG_MEANSHIFT
	void initUnsupervisedSegUI();
#endif
	void initNormalizationUI();
	//void updateBand();
	void buildIlluminant(int temp);

	// cache for illumination coefficients
	typedef std::map<int, std::pair<
			Illuminant, std::vector<multi_img::Value> > > Illum_map;
	Illum_map illuminants;

	CommandRunner *usRunner;

	QMenu *contextMenu;

	TaskType runningTask;

	boost::shared_ptr<multi_img::Mask> graphsegResult;

	bool limitedMode;

	QString startupLabelFile;
};

#endif // MAINWINDOW_H
