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
#include "docks/roidock.h"
#include <shared_data.h>
#include <background_task.h>
#include <background_task_queue.h>
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

class IllumDock;
class IllumModel;
class Controller;

class MainWindow : public QMainWindow, private Ui::MainWindow {
    Q_OBJECT
public:
	MainWindow(bool limitedMode = false);

	const inline Illuminant & getIlluminant(int temp);
	const inline std::vector<multi_img::Value> & getIlluminantC(int temp);
	// TODO: used by Controller; hack until we have a resp. vc-controller
	ViewerContainer* getViewerContainer() { return viewerContainer; }

	static QIcon colorIcon(const QColor& color);

public slots:
	void changeBand(QPixmap band, QString desc);

	void setLabelMatrix(cv::Mat1s matrix);
	void processLabelingChange(const QVector<QColor> &colors, bool changed);
	void processRGB(QPixmap rgb);

	void setGUIEnabled(bool enable, TaskType tt = TT_NONE);
	void finishGraphSeg(bool success);

	void reshapeDock(bool floating);
	void clearLabelOrSeeds();
	void addToLabel();
	void remFromLabel();

	void startGraphseg();

	void startUnsupervisedSeg(bool findKL = false);
	void startFindKL();
	void segmentationFinished();
	void segmentationApply(std::map<std::string, boost::any>);

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

	void loadSeeds();
	void selectLabel(int index);

	void initiateROIChange();

	void openContextMenu();

	void screenshot();

	// DEBUG
	void debugRequestGUIEnabled(bool enable, TaskType tt);

signals:
	void clearLabelRequested(short index);
	void alterLabelRequested(short index, const cv::Mat1b &mask, bool negative);
	void rgbRequested();

	void seedingDone(bool yeah = false);
	// signal new ROI to viewerContainer (TODO needs to go into MODEL)
	void roiChanged(cv::Rect roi);

protected:
	void changeEvent(QEvent *e);

	void runGraphseg(SharedMultiImgPtr input, const vole::GraphSegConfig &config);

private:
	void initUI(Controller *chief);
	void initGraphsegUI();
#ifdef WITH_SEG_MEANSHIFT
	void initUnsupervisedSegUI();
#endif
	void initNormalizationUI();

	IllumDock* illumDock;
	IllumModel* illumModel;

#ifdef WITH_SEG_MEANSHIFT
	CommandRunner *usRunner;
#endif

	QMenu *contextMenu;
	ROIDock *roiDock;

	/* this hack sucks actually! we would get rid of this by using commandrunner
	 * for graphseg, but I start to tend against it
	 */
	boost::shared_ptr<cv::Mat1s> graphsegResult;

	bool limitedMode;
};

#endif // MAINWINDOW_H
