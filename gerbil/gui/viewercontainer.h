#ifndef VIEWERCONTAINER_H
#define VIEWERCONTAINER_H

#include <QWidget>
#include <QVector>

#include "multi_img_viewer.h"
#include <background_task_queue.h>

//namespace Ui {
//class ViewersWidget;
//}

// TODO
// * check if bands can be removed from MainWindow altogether

class ViewerContainer : public QWidget
{
    Q_OBJECT
protected:
    typedef QList<multi_img_viewer*> ViewerList;
	typedef QMap<representation, multi_img_viewer*> ViewerMap;
    
public:
    explicit ViewerContainer(QWidget *parent = 0);
    ~ViewerContainer();

	void setTaskQueue(BackgroundTaskQueue *taskQueue);
    void initUi();

    //void clearBinSets(const std::vector<cv::Rect> &sub, const cv::Rect &roi);



	void setLabels(cv::Mat1s labels);
	void addImage(representation repr, sets_ptr temp, const std::vector<cv::Rect> &regions, cv::Rect roi);
	void subImage(representation repr, sets_ptr temp, const std::vector<cv::Rect> &regions, cv::Rect roi);
	void setImage(representation repr, SharedMultiImgPtr image, cv::Rect roi);

	void toggleLabels(bool toggle);
	void updateLabelColors(QVector<QColor> labelColors, bool changed);

	void updateBinning(representation repr, int bins);

	// MODEL
	// For now these are shared with MainWindow. To be removed when refactored
	// into model classes.
	SharedMultiImgPtr *image, *gradient, *imagepca, *gradientpca;
	std::vector<std::vector<QPixmap*> > *bands; 	// MODEL
	cv::Mat1s *labels; // MODEL

	void disconnectAllViewers();
	void updateViewerBandSelections(int numbands);
	size_t size() const;
	// TODO name consistently with viewer prefix
	const QPixmap* getBand(representation repr, int dim);
	int getSelection(representation repr);
	SharedMultiImgPtr getViewerImage(representation repr);
	representation getActiveRepresentation() const;
	void setIlluminant(representation repr, const std::vector<multi_img::Value> &illuminant, bool for_real);

//	void subViewersLabelMask(sets_ptr temp, const cv::Mat1b &mask);
	//	void addViewersLabelMask(sets_ptr temp, const cv::Mat1b &mask);
	void labelmask(bool negative);

public slots:
	void setGUIEnabled(bool enable, TaskType tt);
	void toggleViewer(bool enable, representation repr);
	void newROI(cv::Rect roi);

	void newOverlay();
	void setActiveViewer(int repri);

	void imgCalculationComplete(bool success);
	void gradCalculationComplete(bool success);
	void imgPcaCalculationComplete(bool success);
	void gradPcaCalculationComplete(bool success);

	// TODO rename (reconnectViewer ?)
	void finishViewerRefresh(representation repr);
	void disconnectViewer(representation repr);

	void finishTask(bool success);
	void finishNormRangeImgChange(bool success);
	void finishNormRangeGradChange(bool success);

	void labelflush(bool seedModeEnabled, short curLabel);
	void refreshLabelsInViewers();

    // TODO
	void viewersHighlight(short);
signals:
	// pass through signals to viewers/viewports
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
	void viewportBandSelected(representation, int);
	// replaced by requestGUIEnabled
	//void viewerSetGUIEnabled(bool, TaskType);

	//void viewerToggleViewer(bool , representation);
	//void viewerFinishTask(bool);
//	void viewportNewOverlay(int);
	void viewportAddSelection();
	void viewportRemSelection();

	void normTargetChanged(bool useCurrent);
	void drawOverlay(const multi_img::Mask &mask);
	void alterLabel(const multi_img::Mask &mask, bool negative);
	void clearLabel();

signals:
	// new signals to break-up coupling between MainWindow and ViewerContainer

	void bandUpdateNeeded(representation repr, int selection);
	void imageResetNeeded(representation repr);

	// TODO still unhandled by MainWindow
	void requestGUIEnabled(bool enable, TaskType tt);

protected:
	ViewerMap vm;
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
