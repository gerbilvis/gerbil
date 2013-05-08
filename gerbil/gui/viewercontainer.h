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

public slots:
	void viewportsKillHover(); // TODO impl
	void viewersOverlay(int, int); // TODO impl
	void viewersSubPixels(const std::map<std::pair<int, int>, short> &); // TODO impl
	void viewersAddPixels(const std::map<std::pair<int, int>, short> &); // TODO impl
	void setViewportActive(int); // TODO impl

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

protected:
    ViewerList vl;
    ViewerMultiMap vm;
    BackgroundTaskQueue *taskQueue;

    QLayout *vLayout;

private:
    /*! \brief Add a multi_img_viewer to the widget.
     *
     * The viewer will be inserted at the bottom of the vertical layout.
     * Needs to be called before signal/slot wiring in initUi() is done.
     */
    multi_img_viewer *createViewer(representation repr);
//private:
//    Ui::ViewersWidget *ui;
};

#endif // VIEWERCONTAINER_H
