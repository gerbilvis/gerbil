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
    
public:
    explicit ViewerContainer(QWidget *parent = 0);
    ~ViewerContainer();

	void setTaskQueue(BackgroundTaskQueue *taskQueue);
    void initUi();

	void addImage(representation::t type, sets_ptr temp, const std::vector<cv::Rect> &regions, cv::Rect roi);
	sets_ptr subImage(representation::t type, const std::vector<cv::Rect> &regions, cv::Rect roi);
	void setImage(representation::t type, SharedMultiImgPtr image, cv::Rect roi);

	void toggleLabels(bool toggle);
	void updateLabels(const cv::Mat1s &labels,
					  const QVector<QColor>& colors = QVector<QColor>(),
					  bool colorsChanged = false);

	void updateBinning(representation::t type, int bins);

	void disconnectAllViewers();
	void updateViewerBandSelections(int numbands);
	int getSelection(representation::t type);
	SharedMultiImgPtr getViewerImage(representation::t type);
	representation::t getActiveRepresentation() const;
	const cv::Mat1b getHighlightMask() const;

public slots:
	void setGUIEnabled(bool enable, TaskType tt);
	void toggleViewer(bool enable, representation::t type);
	void newROI(cv::Rect roi);

	void newOverlay();
	void setActiveViewer(int repri);

	// TODO rename (reconnectViewer ?)
	void connectViewer(representation::t type);
	void disconnectViewer(representation::t type);

	void finishTask(bool success);
	void finishNormRangeImgChange(bool success);
	void finishNormRangeGradChange(bool success);

	void updateLabelsPartially(const cv::Mat1s &labels, const cv::Mat1b &mask);

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

	void bandSelected(representation::t type, int bandId);
	void setGUIEnabledRequested(bool enable, TaskType tt);

protected:
	QMap<representation::t, multi_img_viewer*> map;
	BackgroundTaskQueue *taskQueue;
	multi_img_viewer *activeViewer;
	cv::Rect roi;

private:
    /*! \brief Add a multi_img_viewer to the widget.
     *
     * The viewer will be inserted at the bottom of the vertical layout.
     * Needs to be called before signal/slot wiring in initUi() is done.
     */
	multi_img_viewer *createViewer(representation::t type);

	void enableViewer(representation::t type);
	void disableViewer(representation::t type);
};

#endif // VIEWERCONTAINER_H
