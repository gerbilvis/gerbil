#ifndef VIEWERCONTAINER_H
#define VIEWERCONTAINER_H

#include <QWidget>

#include "multi_img_viewer.h"
#include <background_task_queue.h>

//namespace Ui {
//class ViewersWidget;
//}

class ViewerContainer : public QWidget
{
    Q_OBJECT
protected:
    typedef QList<multi_img_viewer*> ViewerList;
    typedef QMultiMap<representation, multi_img_viewer*> ViewerMultiMap;
    
public:
    explicit ViewerContainer(QWidget *parent = 0);
    ~ViewerContainer();

    void setTaskQueue(BackgroundTaskQueue *taskQueue); // TODO impl
    void initUi();

    //void clearBinSets(const std::vector<cv::Rect> &sub, const cv::Rect &roi);

	// FIXME not sure if it is a good idea to share the temp bin sets between
	// multiple viewers of the same representation. For now it doesn't matter,
	// since we have only one viewer for each representation.
    void addImage(representation repr, sets_ptr temp, const std::vector<cv::Rect> &regions, cv::Rect roi);

	// MODEL
	// For now these are copied from MainWindow. To be removed when refactored
	// into model classes.
	SharedMultiImgPtr *image, *gradient, *imagepca, *gradientpca;

public slots:
	void setGUIEnabled(bool enable, TaskType tt);
	void toggleViewer(bool enable, representation repr);
	void newROI(cv::Rect roi);

	void imgCalculationComplete(bool success);
	void gradCalculationComplete(bool success);
	void imgPcaCalculationComplete(bool success);
	void gradPcaCalculationComplete(bool success);
	void finishViewerRefresh(int viewer);
	void finishTask(bool success);
signals:
	// pass through signals
	void viewportsKillHover();
	void viewersOverlay(int, int);
	void viewersSubPixels(const std::map<std::pair<int, int>, short> &);
	void viewersAddPixels(const std::map<std::pair<int, int>, short> &);
	void viewersToggleLabeled(bool);
	void viewersToggleUnlabeled(bool);

	void setViewportActive(int);

signals:
	// TODO these need to be either wired to mainwindow or removed/broken up
	// to be processed in ViewerContainer.
	void viewPortBandSelected(representation, int);
	void viewerSetGUIEnabled(bool, TaskType);
	void viewerToggleViewer(bool , representation);
	void viewerFinishTask(bool);
	void viewerNewOverlay();
	void viewPortNewOverlay(int);
	void viewPortAddSelection();
	void viewPortRemSelection();

signals:
	// new signals to break-up coupling between MainWindow and ViewerContainer

	void bandUpdateNeeded(representation repr, int selection);
	void imageResetNeeded(representation repr);

	// TODO still unhandled by MainWindow
	void requestGUIEnabled(bool enable, TaskType tt);

protected:
	ViewerList vl;
	ViewerMultiMap vm;
	BackgroundTaskQueue *taskQueue;
	multi_img_viewer *activeViewer;
	cv::Rect roi;

	QLayout *vLayout;

private:
    /*! \brief Add a multi_img_viewer to the widget.
     *
     * The viewer will be inserted at the bottom of the vertical layout.
     * Needs to be called before signal/slot wiring in initUi() is done.
     */
    multi_img_viewer *createViewer(representation repr);

    void toggleViewerEnable(representation repr);
    void toggleViewerDisable(representation repr);
//private:
//    Ui::ViewersWidget *ui;
};

#endif // VIEWERCONTAINER_H