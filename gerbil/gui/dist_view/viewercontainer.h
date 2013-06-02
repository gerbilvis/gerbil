#ifndef VIEWERCONTAINER_H
#define VIEWERCONTAINER_H

#include "multi_img_viewer.h"
#include <illuminant.h>
#include <background_task_queue.h>

#include <QWidget>
#include <QVector>
#include <QLayout>

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

	void addImage(representation repr, sets_ptr temp, const std::vector<cv::Rect> &regions, cv::Rect roi);
	sets_ptr subImage(representation repr, const std::vector<cv::Rect> &regions, cv::Rect roi);
	void setImage(representation repr, SharedMultiImgPtr image, cv::Rect roi);

	void toggleLabels(bool toggle);
	void updateLabelColors(QVector<QColor> colors, bool changed);

	void updateBinning(representation repr, int bins);

	// MODEL
	// To be removed when refactored into model classes.
	SharedMultiImgPtr image, gradient, imagepca, gradientpca;

	void disconnectAllViewers();
	void updateViewerBandSelections(int numbands);
	int getSelection(representation repr);
	SharedMultiImgPtr getViewerImage(representation repr);
	representation getActiveRepresentation() const;
	const cv::Mat1b getHighlightMask() const;

public slots:
	void setLabelMatrix(cv::Mat1s matrix);
	void setGUIEnabled(bool enable, TaskType tt);
	void toggleViewer(bool enable, representation repr);
	void newROI(cv::Rect roi);

	void newOverlay();
	void setActiveViewer(int repri);

	// TODO rename (reconnectViewer ?)
	void connectViewer(representation repr);
	void disconnectViewer(representation repr);

	void finishTask(bool success);
	void finishNormRangeImgChange(bool success);
	void finishNormRangeGradChange(bool success);

	void updateLabelsPartially(cv::Mat1b mask, cv::Mat1s old);
	void updateLabels();
	void newIlluminant(cv::Mat1f illum);
	void showIlluminationCurve(bool show);
	void setIlluminantApplied(bool applied);

signals:
	// pass through signals to viewers/viewports
	void viewportsKillHover();
	void viewersOverlay(int, int);
	void viewersSubPixels(const std::map<std::pair<int, int>, short> &);
	void viewersAddPixels(const std::map<std::pair<int, int>, short> &);
	void viewersToggleLabeled(bool);
	void viewersToggleUnlabeled(bool);
	void viewersHighlight(short);

	void setViewportActive(int);

	void viewportAddSelection();
	void viewportRemSelection();

	void normTargetChanged(bool useCurrent);
	void drawOverlay(const cv::Mat1b &mask);

	// new signals to break-up coupling between MainWindow and ViewerContainer

	void bandSelected(representation type, int selection);
	void requestGUIEnabled(bool enable, TaskType tt);

protected:
	ViewerMap vm;
	BackgroundTaskQueue *taskQueue;
	multi_img_viewer *activeViewer;
	cv::Rect roi;

private:
    /*! \brief Add a multi_img_viewer to the widget.
     *
     * The viewer will be inserted at the bottom of the vertical layout.
     * Needs to be called before signal/slot wiring in initUi() is done.
     */
    multi_img_viewer *createViewer(representation repr);

    void enableViewer(representation repr);
    void disableViewer(representation repr);
};

#endif // VIEWERCONTAINER_H
